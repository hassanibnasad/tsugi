//! # Interactive Conflict Resolution UI
//!
//! Presents a 3-option quick-choice menu when merge or rebase conflicts are detected:
//! - [1] Keep Ours (HEAD)
//! - [2] Keep Theirs (Incoming branch)
//! - [3] Let tsugi AI synthesize resolution

use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use colored::Colorize;

use tsugi_core::types::ConflictEntry;

/// User choice for conflict resolution.
#[derive(Debug, PartialEq, Eq)]
pub enum ConflictChoice {
    KeepOurs,
    KeepTheirs,
    LetAiResolve,
    Cancel,
}

/// Prompt the user with a quick-choice resolution menu for a conflicting file.
pub fn prompt_conflict_resolution(conflict: &ConflictEntry) -> Result<ConflictChoice> {
    println!("\n  {} {}", "Active Conflict Detected in:".yellow().bold(), conflict.path.white().bold());

    if let Some(ref hunk) = conflict.primary_hunk {
        println!("  {}", "┌── Conflicting Changes ───────────────────────".dimmed());
        println!("  {} <<<<<<< Ours ({})", "│".dimmed(), hunk.ours_branch.green().bold());
        for line in hunk.ours_content.iter().take(8) {
            println!("  {}   {}", "│".dimmed(), line.green());
        }
        if hunk.ours_content.len() > 8 {
            println!("  {}   {}", "│".dimmed(), format!("... ({} more lines)", hunk.ours_content.len() - 8).dimmed());
        }
        println!("  {} =======", "│".dimmed());
        for line in hunk.theirs_content.iter().take(8) {
            println!("  {}   {}", "│".dimmed(), line.red());
        }
        if hunk.theirs_content.len() > 8 {
            println!("  {}   {}", "│".dimmed(), format!("... ({} more lines)", hunk.theirs_content.len() - 8).dimmed());
        }
        println!("  {} >>>>>>> Theirs ({})", "│".dimmed(), hunk.theirs_branch.red().bold());
        println!("  {}", "└──".dimmed());
    }

    println!("\n  How would you like to resolve this conflict?");
    println!("    {} Keep Ours ({})", "[1]".cyan().bold(), "checkout --ours".dimmed());
    println!("    {} Keep Theirs ({})", "[2]".cyan().bold(), "checkout --theirs".dimmed());
    println!("    {} Let tsugi AI resolve", "[3]".cyan().bold());
    println!("    {} Skip / Cancel", "[4]".dimmed());

    print!("\n  Select option [1-4]: ");
    io::stdout().flush().ok();

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    match input.trim() {
        "1" => Ok(ConflictChoice::KeepOurs),
        "2" => Ok(ConflictChoice::KeepTheirs),
        "3" => Ok(ConflictChoice::LetAiResolve),
        _ => Ok(ConflictChoice::Cancel),
    }
}

/// Execute checkout --ours or checkout --theirs and stage the resolved file.
pub fn resolve_with_checkout(repo_path: &Path, file_path: &str, keep_ours: bool) -> Result<()> {
    let side_flag = if keep_ours { "--ours" } else { "--theirs" };
    let side_name = if keep_ours { "Ours (HEAD)" } else { "Theirs (Incoming)" };

    let checkout_out = Command::new("git")
        .args(["checkout", side_flag, "--", file_path])
        .current_dir(repo_path)
        .output()
        .with_context(|| format!("Failed to run git checkout {side_flag}"))?;

    if !checkout_out.status.success() {
        let stderr = String::from_utf8_lossy(&checkout_out.stderr);
        anyhow::bail!("git checkout {side_flag} failed: {stderr}");
    }

    let add_out = Command::new("git")
        .args(["add", file_path])
        .current_dir(repo_path)
        .output()
        .context("Failed to run git add")?;

    if !add_out.status.success() {
        let stderr = String::from_utf8_lossy(&add_out.stderr);
        anyhow::bail!("git add failed: {stderr}");
    }

    println!(
        "\n  {} Resolved {} by keeping {} and staged.",
        "✔".green().bold(),
        file_path.white().bold(),
        side_name.cyan().bold()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_resolve_with_checkout_ours_and_theirs() {
        let tmp = TempDir::new().unwrap();
        let repo = git2::Repository::init(tmp.path()).unwrap();
        let file_path = tmp.path().join("conflict.txt");
        std::fs::write(&file_path, "base\n").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("conflict.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@test.com").unwrap();
        let base_commit = repo.commit(Some("HEAD"), &sig, &sig, "Base", &tree, &[]).unwrap();

        let base_obj = repo.find_object(base_commit, None).unwrap();
        let _ = repo.branch("b1", base_obj.as_commit().unwrap(), false).unwrap();

        // Master change
        std::fs::write(&file_path, "master change\n").unwrap();
        index.add_path(Path::new("conflict.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let master_parent = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Master", &tree, &[&master_parent]).unwrap();

        // B1 change
        repo.set_head("refs/heads/b1").unwrap();
        repo.checkout_head(Some(git2::build::CheckoutBuilder::new().force())).unwrap();
        std::fs::write(&file_path, "branch change\n").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("conflict.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let b1_parent = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "B1", &tree, &[&b1_parent]).unwrap();

        // Back to master and merge
        repo.set_head("refs/heads/master").unwrap();
        repo.checkout_head(Some(git2::build::CheckoutBuilder::new().force())).unwrap();

        let b1_annotated = repo.find_annotated_commit(repo.resolve_reference_from_short_name("b1").unwrap().target().unwrap()).unwrap();
        repo.merge(&[&b1_annotated], None, None).unwrap();
        assert!(repo.index().unwrap().has_conflicts());

        // Test resolve with checkout ours
        let res = resolve_with_checkout(tmp.path(), "conflict.txt", true);
        assert!(res.is_ok());

        let resolved_content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(resolved_content.trim(), "master change");
    }
}
