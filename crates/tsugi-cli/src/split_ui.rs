//! # Interactive Split Commit Terminal Formatter & UI
//!
//! Visualizes proposed atomic commit clusters and provides an interactive
//! approval prompt before progressive staging and committing.

use std::io::{self, Write};

use colored::Colorize;
use tsugi_core::types::SplitCommitPlan;

/// Pretty-print the proposed split commit plan to the terminal.
pub fn print_split_plan(plan: &SplitCommitPlan) {
    println!("\n  {}", "Proposed Commit Split Plan".white().bold());
    println!("  {}", "──────────────────────────".dimmed());
    println!("  {} {}\n", "Summary:".cyan(), plan.summary);

    let total = plan.commits.len();
    for (idx, commit) in plan.commits.iter().enumerate() {
        let tag = format!("[Commit {}/{}]", idx + 1, total);
        println!("  {} {}", tag.cyan().bold(), commit.message.white().bold());
        println!("     {} {}", "Rationale:".dimmed(), commit.explanation.dimmed());
        println!("     {} {} file(s):", "Files:".dimmed(), commit.files.len());
        for f in &commit.files {
            println!("       · {}", f.green());
        }
        println!();
    }

    if !plan.unassigned_files.is_empty() {
        println!(
            "  {} {}",
            "⚠".yellow().bold(),
            "The following files will remain uncommitted:".yellow()
        );
        for f in &plan.unassigned_files {
            println!("     · {}", f.yellow());
        }
        println!();
    }
}

/// Prompt the user to confirm execution of the split plan.
pub fn prompt_split_confirmation() -> io::Result<bool> {
    print!(
        "  {} ",
        "Proceed with progressive commits? [y/N]:".yellow().bold()
    );
    io::stdout().flush().ok();

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(matches!(input.trim().to_lowercase().as_str(), "y" | "yes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsugi_core::types::ProposedCommit;

    #[test]
    fn test_print_split_plan_smoke() {
        let plan = SplitCommitPlan {
            summary: "Test split".to_string(),
            commits: vec![ProposedCommit {
                commit_type: "feat".to_string(),
                scope: Some("core".to_string()),
                message: "feat(core): test".to_string(),
                explanation: "test explanation".to_string(),
                files: vec!["file1.rs".to_string()],
            }],
            unassigned_files: vec!["file2.log".to_string()],
        };

        // Smoke test ensuring formatter runs without panic
        print_split_plan(&plan);
    }
}
