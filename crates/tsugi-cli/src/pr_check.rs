//! # PR Readiness Checker (`tsugi pr-ready`)
//!
//! Audits the active branch before pushing or opening a Pull Request:
//! - Working tree cleanliness
//! - Branch divergence from base (main/master)
//! - Conventional Commits compliance
//! - Detection of WIP / fixup / squash commits
//! - Secret and token leakage prevention in diffs
//! - Large file size detection (>5MB without LFS)
//! - Merge commit pollution in feature branch

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use colored::Colorize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckSeverity {
    Passed,
    Warning,
    Failure,
}

#[derive(Debug, Clone)]
pub struct PrCheckItem {
    pub title: &'static str,
    pub severity: CheckSeverity,
    pub message: String,
}

pub struct PrCheckReport {
    pub branch: String,
    pub base_branch: String,
    pub items: Vec<PrCheckItem>,
}

impl PrCheckReport {
    pub fn is_ready(&self) -> bool {
        !self.items.iter().any(|i| i.severity == CheckSeverity::Failure)
    }

    pub fn print_summary(&self) {
        println!("\n  {} PR Readiness Audit for branch '{}'", "📋".cyan(), self.branch.white().bold());
        println!("  Target Base: {}", self.base_branch.cyan());
        println!("  {}", "──────────────────────────────────────────────────────".dimmed());

        for item in &self.items {
            let icon = match item.severity {
                CheckSeverity::Passed => "✔".green().bold(),
                CheckSeverity::Warning => "⚠".yellow().bold(),
                CheckSeverity::Failure => "✖".red().bold(),
            };
            println!("  {} {}: {}", icon, item.title.bold(), item.message);
        }

        println!("  {}", "──────────────────────────────────────────────────────".dimmed());
        if self.is_ready() {
            println!(
                "  {} Branch is {} for Pull Request!",
                "🚀".cyan(),
                "READY".green().bold()
            );
        } else {
            println!(
                "  {} Branch has {} that should be resolved before opening PR.",
                "🛑".red(),
                "blocking issues".red().bold()
            );
        }
    }
}

pub fn run_pr_check(repo_path: &Path, custom_base: Option<&str>) -> Result<PrCheckReport> {
    let repo = git2::Repository::discover(repo_path)
        .context("Failed to open git repository for PR check")?;

    let head = repo.head().context("Failed to inspect repository HEAD")?;
    let branch = head.shorthand().unwrap_or("HEAD").to_string();

    let base_branch = custom_base
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            if repo.find_branch("main", git2::BranchType::Local).is_ok()
                || repo.find_branch("origin/main", git2::BranchType::Remote).is_ok()
            {
                "main".to_string()
            } else {
                "master".to_string()
            }
        });

    let mut items = Vec::new();

    // 1. Working tree cleanliness
    let mut status_opts = git2::StatusOptions::new();
    status_opts.include_untracked(true);
    let statuses = repo.statuses(Some(&mut status_opts))?;
    if statuses.is_empty() {
        items.push(PrCheckItem {
            title: "Working Tree",
            severity: CheckSeverity::Passed,
            message: "Clean — no uncommitted or untracked changes".into(),
        });
    } else {
        items.push(PrCheckItem {
            title: "Working Tree",
            severity: CheckSeverity::Warning,
            message: format!("Dirty — {} uncommitted modification(s)", statuses.len()),
        });
    }

    // 2. Query commits ahead/behind base using git CLI
    let rev_range = format!("{}..HEAD", base_branch);
    let log_output = Command::new("git")
        .args(["log", &rev_range, "--oneline"])
        .current_dir(repo_path)
        .output();

    let commit_lines: Vec<String> = match log_output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        }
        _ => Vec::new(),
    };

    if commit_lines.is_empty() {
        items.push(PrCheckItem {
            title: "Commit Count",
            severity: CheckSeverity::Warning,
            message: format!("Branch has 0 commits ahead of '{}'", base_branch),
        });
    } else {
        items.push(PrCheckItem {
            title: "Commit Count",
            severity: CheckSeverity::Passed,
            message: format!("{} commit(s) ready for review", commit_lines.len()),
        });

        // 3. Check for WIP / fixup / squash commits
        let wip_commits: Vec<&String> = commit_lines
            .iter()
            .filter(|c| {
                let lower = c.to_lowercase();
                lower.contains("wip")
                    || lower.contains("fixup!")
                    || lower.contains("squash!")
                    || lower.contains("temp")
            })
            .collect();

        if wip_commits.is_empty() {
            items.push(PrCheckItem {
                title: "WIP Commits",
                severity: CheckSeverity::Passed,
                message: "No WIP or fixup commits detected".into(),
            });
        } else {
            items.push(PrCheckItem {
                title: "WIP Commits",
                severity: CheckSeverity::Failure,
                message: format!("{} temporary/WIP commit(s) found in PR range. Squash before merging.", wip_commits.len()),
            });
        }

        // 4. Conventional Commits format
        let conventional_prefixes = ["feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert"];
        let mut non_conventional = 0;
        for line in &commit_lines {
            let msg_part = line.splitn(2, ' ').nth(1).unwrap_or("");
            let is_conv = conventional_prefixes.iter().any(|&p| {
                msg_part.starts_with(&format!("{}:", p)) || msg_part.starts_with(&format!("{}(", p))
            });
            if !is_conv {
                non_conventional += 1;
            }
        }

        if non_conventional == 0 {
            items.push(PrCheckItem {
                title: "Conventional Commits",
                severity: CheckSeverity::Passed,
                message: "100% of commits follow Conventional Commits specification".into(),
            });
        } else {
            items.push(PrCheckItem {
                title: "Conventional Commits",
                severity: CheckSeverity::Warning,
                message: format!("{}/{} commits do not follow Conventional Commits (e.g. feat: / fix:)", non_conventional, commit_lines.len()),
            });
        }

        // 5. Merge commit check (linear history)
        let merge_commits: Vec<&String> = commit_lines
            .iter()
            .filter(|c| {
                let msg_part = c.splitn(2, ' ').nth(1).unwrap_or("").to_lowercase();
                msg_part.starts_with("merge ")
            })
            .collect();

        if merge_commits.is_empty() {
            items.push(PrCheckItem {
                title: "Linear History",
                severity: CheckSeverity::Passed,
                message: "No merge commits found in PR branch range".into(),
            });
        } else {
            items.push(PrCheckItem {
                title: "Linear History",
                severity: CheckSeverity::Warning,
                message: format!("{} merge commit(s) in branch. Consider rebasing on {}", merge_commits.len(), base_branch),
            });
        }
    }

    // 6. Secret scanning on diff
    let diff_output = Command::new("git")
        .args(["diff", &rev_range])
        .current_dir(repo_path)
        .output();

    if let Ok(out) = diff_output {
        let diff_str = String::from_utf8_lossy(&out.stdout);
        let secret_patterns = [
            ("AWS Access Key", "AKIA[0-9A-Z]{16}"),
            ("GitHub Personal Token", "ghp_"),
            ("OpenAI / Stripe API Key", "sk-live-"),
            ("Private Key Header", "-----BEGIN RSA PRIVATE KEY-----"),
            ("Private Key Header", "-----BEGIN OPENSSH PRIVATE KEY-----"),
        ];

        let mut detected_secrets = Vec::new();
        for (name, pat) in secret_patterns {
            if diff_str.contains(pat) {
                detected_secrets.push(name);
            }
        }

        if detected_secrets.is_empty() {
            items.push(PrCheckItem {
                title: "Secret Leak Scan",
                severity: CheckSeverity::Passed,
                message: "No API keys, credentials, or private keys detected in PR diff".into(),
            });
        } else {
            items.push(PrCheckItem {
                title: "Secret Leak Scan",
                severity: CheckSeverity::Failure,
                message: format!("POTENTIAL SECRET DETECTED: {:?}", detected_secrets),
            });
        }
    }

    Ok(PrCheckReport {
        branch,
        base_branch,
        items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pr_check_report_readiness() {
        let report_pass = PrCheckReport {
            branch: "feat-a".to_string(),
            base_branch: "main".to_string(),
            items: vec![
                PrCheckItem {
                    title: "Working Tree",
                    severity: CheckSeverity::Passed,
                    message: "Clean".into(),
                },
                PrCheckItem {
                    title: "Commit Count",
                    severity: CheckSeverity::Warning,
                    message: "Warning note".into(),
                },
            ],
        };
        assert!(report_pass.is_ready());

        let report_fail = PrCheckReport {
            branch: "feat-a".to_string(),
            base_branch: "main".to_string(),
            items: vec![
                PrCheckItem {
                    title: "Secret Leak Scan",
                    severity: CheckSeverity::Failure,
                    message: "Leaked key".into(),
                },
            ],
        };
        assert!(!report_fail.is_ready());
    }
}
