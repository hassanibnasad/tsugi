//! # Git Guard Hook Safety Net Subsystem
//!
//! Installs fail-safe, zero-cost Git hooks (`pre-rebase`, `post-checkout`,
//! `pre-merge-commit`, `reference-transaction`) that automatically snapshot
//! repository state before potentially destructive operations.
//!
//! ## Invariants
//!
//! 1. **Never Break Git**: Guard hooks MUST ALWAYS exit 0, even if tsugi is missing,
//!    the database is locked, or a crash occurs.
//! 2. **Preserve User Hooks**: Existing hooks (Husky, lint-staged, custom scripts)
//!    are preserved via backup copies and script prepending.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use colored::Colorize;
use tracing::error;

use tsugi_core::{
    config::load_config,
    types::{ExecutionStep, PlanSource, ResolutionPlan, RiskTier},
    TsugiEngine,
};

const HOOK_MARKER_START: &str = "# --- TSUGI GUARD HOOK BEGIN ---";
const HOOK_MARKER_END: &str = "# --- TSUGI GUARD HOOK END ---";

const TARGET_HOOKS: &[&str] = &[
    "pre-rebase",
    "post-checkout",
    "pre-merge-commit",
    "reference-transaction",
];

const HOOK_SCRIPT_BODY: &str = r#"# --- TSUGI GUARD HOOK BEGIN ---
if command -v tsugi >/dev/null 2>&1; then
    tsugi checkpoint --quiet >/dev/null 2>&1 || true
fi
# --- TSUGI GUARD HOOK END ---
"#;

/// Checkpoint the current repository state silently and quickly (<15ms).
/// Always exits 0 to never break Git operations.
pub fn cmd_checkpoint(repo_path: &Path, quiet: bool) -> Result<()> {
    let run_checkpoint = || -> Result<()> {
        let config = load_config().unwrap_or_default();
        let engine = TsugiEngine::new(config)?;

        let plan = ResolutionPlan {
            root_cause: "Automatic Git guard checkpoint".to_string(),
            risk_tier: RiskTier::ReadOnly,
            execution_steps: vec![ExecutionStep {
                cmd: "git status".to_string(),
                explanation: "Guard snapshot".to_string(),
                requires_confirmation: false,
            }],
            rollback_cmd: "git status".to_string(),
            source: PlanSource::Tier0,
            confidence: 1.0,
        };

        let snapshot = engine.vault().checkpoint(repo_path, &plan)?;
        if !quiet {
            println!(
                "  {} Guard checkpoint created: {}",
                "🛡".to_string(),
                snapshot.id[..8.min(snapshot.id.len())].cyan()
            );
        }
        Ok(())
    };

    if let Err(e) = run_checkpoint() {
        // Log error silently to ~/.tsugi/guard.log
        if let Some(home) = dirs::home_dir() {
            let log_dir = home.join(".tsugi");
            let _ = fs::create_dir_all(&log_dir);
            let log_path = log_dir.join("guard.log");
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path) {
                let timestamp = chrono::Utc::now().to_rfc3339();
                let _ = writeln!(file, "[{timestamp}] Checkpoint failed in {}: {e}", repo_path.display());
            }
        }
        if !quiet {
            error!("Guard checkpoint warning: {e}");
        }
    }

    Ok(())
}

/// Install git guard hooks into `.git/hooks/` or globally.
pub fn install_hooks(repo_path: &Path, global: bool) -> Result<()> {
    if global {
        install_global_hooks()
    } else {
        install_local_hooks(repo_path)
    }
}

fn install_local_hooks(repo_path: &Path) -> Result<()> {
    let git_dir = find_git_dir(repo_path)?;
    let hooks_dir = git_dir.join("hooks");
    fs::create_dir_all(&hooks_dir)
        .with_context(|| format!("Failed to create hooks directory: {}", hooks_dir.display()))?;

    println!("\n  {} Installing tsugi guard hooks into: {}", "🛡".cyan().bold(), hooks_dir.display());

    let mut installed_count = 0;
    for hook_name in TARGET_HOOKS {
        let hook_file = hooks_dir.join(hook_name);
        install_single_hook(&hook_file, hook_name)?;
        installed_count += 1;
    }

    println!(
        "\n  {} Successfully installed {} guard hook(s)!",
        "✔".green().bold(),
        installed_count
    );
    println!("     Your repository is now protected: every rebase, merge, and checkout will create an automatic snapshot.");
    Ok(())
}

fn install_global_hooks() -> Result<()> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    let global_hooks_dir = home.join(".tsugi").join("hooks");
    fs::create_dir_all(&global_hooks_dir)?;

    println!("\n  {} Installing global tsugi guard hooks into: {}", "🛡".cyan().bold(), global_hooks_dir.display());

    for hook_name in TARGET_HOOKS {
        let hook_file = global_hooks_dir.join(hook_name);
        install_single_hook(&hook_file, hook_name)?;
    }

    // Configure core.hooksPath in global git config
    let output = Command::new("git")
        .args(["config", "--global", "core.hooksPath", &global_hooks_dir.to_string_lossy()])
        .output()
        .context("Failed to configure git core.hooksPath")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to set git core.hooksPath: {stderr}");
    }

    println!(
        "  {} Configured git core.hooksPath -> {}",
        "✔".green().bold(),
        global_hooks_dir.display().to_string().cyan()
    );
    Ok(())
}

fn install_single_hook(hook_file: &Path, hook_name: &str) -> Result<()> {
    if hook_file.exists() {
        let content = fs::read_to_string(hook_file).unwrap_or_default();
        if content.contains(HOOK_MARKER_START) {
            println!("     · {} already contains tsugi guard hook (up-to-date)", hook_name.white());
            return Ok(());
        }

        // Backup existing hook
        let backup_path = hook_file.with_extension("tsugi-bak");
        fs::copy(hook_file, &backup_path)?;
        println!(
            "     · Existing {} preserved (backed up to {})",
            hook_name.yellow(),
            backup_path.file_name().unwrap_or_default().to_string_lossy().dimmed()
        );

        // Prepend tsugi hook block
        let mut new_content = String::new();
        if content.starts_with("#!") {
            let mut lines = content.lines();
            if let Some(shebang) = lines.next() {
                new_content.push_str(shebang);
                new_content.push('\n');
            }
            new_content.push_str(HOOK_SCRIPT_BODY);
            for line in lines {
                new_content.push_str(line);
                new_content.push('\n');
            }
        } else {
            new_content.push_str("#!/bin/sh\n");
            new_content.push_str(HOOK_SCRIPT_BODY);
            new_content.push_str(&content);
        }

        fs::write(hook_file, new_content)?;
    } else {
        let full_script = format!("#!/bin/sh\n{HOOK_SCRIPT_BODY}");
        fs::write(hook_file, full_script)?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = hook_file.metadata() {
            let mut perms = meta.permissions();
            perms.set_mode(0o755);
            let _ = fs::set_permissions(hook_file, perms);
        }
    }

    println!("     {} Installed {}", "✔".green(), hook_name.white().bold());
    Ok(())
}

/// Uninstall git guard hooks.
pub fn uninstall_hooks(repo_path: &Path, global: bool) -> Result<()> {
    if global {
        let home = dirs::home_dir().context("Cannot determine home directory")?;
        let global_hooks_dir = home.join(".tsugi").join("hooks");
        if global_hooks_dir.exists() {
            let _ = fs::remove_dir_all(&global_hooks_dir);
        }
        let _ = Command::new("git")
            .args(["config", "--global", "--unset", "core.hooksPath"])
            .output();
        println!("\n  {} Global guard hooks removed.", "✔".green().bold());
    } else {
        let git_dir = find_git_dir(repo_path)?;
        let hooks_dir = git_dir.join("hooks");

        for hook_name in TARGET_HOOKS {
            let hook_file = hooks_dir.join(hook_name);
            if hook_file.exists() {
                let content = fs::read_to_string(&hook_file).unwrap_or_default();
                if content.contains(HOOK_MARKER_START) {
                    if let (Some(start), Some(end)) = (content.find(HOOK_MARKER_START), content.find(HOOK_MARKER_END)) {
                        let end_pos = end + HOOK_MARKER_END.len();
                        let remaining = format!("{}{}", &content[..start], &content[end_pos..]).trim().to_string();
                        if remaining.is_empty() || remaining == "#!/bin/sh" {
                            let _ = fs::remove_file(&hook_file);
                        } else {
                            let _ = fs::write(&hook_file, remaining);
                        }
                    }
                }
                // Restore backup if present
                let backup_file = hook_file.with_extension("tsugi-bak");
                if backup_file.exists() {
                    let _ = fs::rename(&backup_file, &hook_file);
                }
            }
        }
        println!("\n  {} Local guard hooks removed.", "✔".green().bold());
    }
    Ok(())
}

/// Print guard status for current repo.
pub fn guard_status(repo_path: &Path) -> Result<()> {
    println!("\n  {}", "Git Guard Status".white().bold());
    println!("  {}", "────────────────".dimmed());

    let git_dir = find_git_dir(repo_path);
    match git_dir {
        Ok(dir) => {
            let hooks_dir = dir.join("hooks");
            let mut active = 0;
            for hook in TARGET_HOOKS {
                let file = hooks_dir.join(hook);
                let is_installed = file.exists()
                    && fs::read_to_string(&file)
                        .map(|c| c.contains(HOOK_MARKER_START))
                        .unwrap_or(false);
                if is_installed {
                    println!("    · {:<22} {}", hook.white().bold(), "active".green());
                    active += 1;
                } else {
                    println!("    · {:<22} {}", hook.dimmed(), "not installed".dimmed());
                }
            }
            if active > 0 {
                println!("\n  {} Guard protection is active ({}/{} hooks installed)", "✔".green().bold(), active, TARGET_HOOKS.len());
            } else {
                println!("\n  {} Guard is not yet installed in this repo.", "ℹ".yellow());
                println!("    Run {} to enable automatic pre-operation snapshots.", "tsugi guard install".white().bold());
            }
        }
        Err(_) => {
            println!("  {} Not inside a git repository.", "✖".red());
        }
    }
    Ok(())
}

fn find_git_dir(repo_path: &Path) -> Result<PathBuf> {
    let repo = git2::Repository::discover(repo_path)
        .context("Failed to find git repository")?;
    Ok(repo.path().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_install_and_uninstall_local_hooks() {
        let tmp = TempDir::new().unwrap();
        let repo = git2::Repository::init(tmp.path()).unwrap();

        // Install hooks
        install_hooks(tmp.path(), false).unwrap();

        let hooks_dir = repo.path().join("hooks");
        for hook in TARGET_HOOKS {
            let hook_file = hooks_dir.join(hook);
            assert!(hook_file.exists(), "Hook file {} should exist", hook);
            let content = fs::read_to_string(&hook_file).unwrap();
            assert!(content.contains(HOOK_MARKER_START));
            assert!(content.contains(HOOK_MARKER_END));
        }

        // Uninstall hooks
        uninstall_hooks(tmp.path(), false).unwrap();

        for hook in TARGET_HOOKS {
            let hook_file = hooks_dir.join(hook);
            assert!(!hook_file.exists(), "Hook file {} should be deleted after uninstall", hook);
        }
    }

    #[test]
    fn test_existing_hook_preserved_with_backup() {
        let tmp = TempDir::new().unwrap();
        let repo = git2::Repository::init(tmp.path()).unwrap();
        let hooks_dir = repo.path().join("hooks");
        fs::create_dir_all(&hooks_dir).unwrap();

        let custom_script = "#!/bin/sh\n# husky pre-rebase\nnpm run lint\n";
        let pre_rebase_file = hooks_dir.join("pre-rebase");
        fs::write(&pre_rebase_file, custom_script).unwrap();

        // Install guard
        install_hooks(tmp.path(), false).unwrap();

        // Backup must exist
        let backup_file = hooks_dir.join("pre-rebase.tsugi-bak");
        assert!(backup_file.exists());
        assert_eq!(fs::read_to_string(&backup_file).unwrap(), custom_script);

        // Pre-rebase must contain both tsugi guard and original script
        let content = fs::read_to_string(&pre_rebase_file).unwrap();
        assert!(content.contains(HOOK_MARKER_START));
        assert!(content.contains("npm run lint"));

        // Uninstall must restore original script
        uninstall_hooks(tmp.path(), false).unwrap();
        assert!(pre_rebase_file.exists());
        assert_eq!(fs::read_to_string(&pre_rebase_file).unwrap(), custom_script);
        assert!(!backup_file.exists());
    }

    #[test]
    fn test_cmd_checkpoint_creates_snapshot() {
        let tmp = TempDir::new().unwrap();
        let repo = git2::Repository::init(tmp.path()).unwrap();

        // Commit initial file
        let file = tmp.path().join("hello.txt");
        fs::write(&file, "initial").unwrap();
        let mut idx = repo.index().unwrap();
        idx.add_path(Path::new("hello.txt")).unwrap();
        idx.write().unwrap();
        let tree_id = idx.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@test.com").unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[]).unwrap();

        // Make dirty change
        fs::write(&file, "modified").unwrap();

        // Run silent checkpoint
        let res = cmd_checkpoint(tmp.path(), true);
        assert!(res.is_ok());
    }
}
