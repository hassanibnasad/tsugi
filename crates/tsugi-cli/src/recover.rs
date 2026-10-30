//! # Git Lost Commit & State Recovery (`tsugi recover`)
//!
//! Hunts for accidentally lost, detached, or reset commits using a combination
//! of `git reflog` archaeology and `git fsck --unreachable`.

use std::path::Path;
use std::process::Command;

use anyhow::Result;
use colored::Colorize;

#[derive(Debug, Clone)]
pub struct LostCommit {
    pub sha_short: String,
    #[allow(dead_code)]
    pub sha_full: String,
    pub message: String,
    pub author: String,
    pub date: String,
    pub source: &'static str,
}

/// Locate dangling or abandoned commits in the repository.
pub fn find_lost_commits(repo_path: &Path, limit: usize) -> Result<Vec<LostCommit>> {
    let mut commits = Vec::new();
    let mut seen_shas = std::collections::HashSet::new();

    // 1. Reflog archaeology: examine commits in reflog that are no longer HEAD or branches
    let reflog_out = Command::new("git")
        .args(["reflog", "-n", "60", "--pretty=format:%h|%H|%an|%cr|%s"])
        .current_dir(repo_path)
        .output();

    if let Ok(out) = reflog_out {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 5 {
                let sha_short = parts[0].to_string();
                let sha_full = parts[1].to_string();
                let author = parts[2].to_string();
                let date = parts[3].to_string();
                let message = parts[4..].join("|");

                // Filter out standard checkout switches, keep resets/commits
                if !seen_shas.contains(&sha_short) {
                    seen_shas.insert(sha_short.clone());
                    commits.push(LostCommit {
                        sha_short,
                        sha_full,
                        message,
                        author,
                        date,
                        source: "reflog",
                    });
                }
            }
        }
    }

    // 2. Fsck unreachable search
    let fsck_out = Command::new("git")
        .args(["fsck", "--unreachable", "--no-reflogs"])
        .current_dir(repo_path)
        .output();

    if let Ok(out) = fsck_out {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            if line.starts_with("unreachable commit ") {
                let sha = line.trim_start_matches("unreachable commit ").trim();
                let sha_short = if sha.len() >= 7 { &sha[..7] } else { sha };

                if !seen_shas.contains(sha_short) {
                    seen_shas.insert(sha_short.to_string());

                    // Inspect commit info
                    let show_out = Command::new("git")
                        .args(["log", "-1", "--pretty=format:%an|%cr|%s", sha])
                        .current_dir(repo_path)
                        .output();

                    if let Ok(s_out) = show_out {
                        let text = String::from_utf8_lossy(&s_out.stdout);
                        let p: Vec<&str> = text.split('|').collect();
                        let author = p.first().copied().unwrap_or("Unknown").to_string();
                        let date = p.get(1).copied().unwrap_or("Unknown").to_string();
                        let message = p.get(2..).map(|s| s.join("|")).unwrap_or_else(|| "No message".into());

                        commits.push(LostCommit {
                            sha_short: sha_short.to_string(),
                            sha_full: sha.to_string(),
                            message,
                            author,
                            date,
                            source: "dangling object",
                        });
                    }
                }
            }
        }
    }

    commits.truncate(limit);
    Ok(commits)
}

/// Print formatted list of lost commits with recovery commands.
pub fn print_recovery_list(commits: &[LostCommit]) {
    if commits.is_empty() {
        println!("\n  {} No lost or dangling commits found. Repository graph is intact.", "✔".green().bold());
        return;
    }

    println!("\n  {} Found {} recoverable commit candidate(s):", "🔍".cyan(), commits.len().to_string().cyan().bold());
    println!("  {}", "─────────────────────────────────────────────────────────────".dimmed());

    for (idx, c) in commits.iter().enumerate() {
        println!(
            "  [{}] {} ({}) — {}",
            (idx + 1).to_string().cyan().bold(),
            c.sha_short.yellow().bold(),
            c.date.dimmed(),
            c.message.white().bold()
        );
        println!(
            "      Author: {} | Source: {}",
            c.author.dimmed(),
            c.source.cyan()
        );
        println!(
            "      Recover: {}",
            format!("git switch -c recovery/{} {}", c.sha_short, c.sha_short).green()
        );
        println!();
    }

    println!("  {} Run the green command above to resurrect any commit into a branch.", "💡".cyan());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_empty_recovery_list_does_not_panic() {
        print_recovery_list(&[]);
    }

    #[test]
    fn test_print_recovery_list_items() {
        let items = vec![LostCommit {
            sha_short: "abc1234".to_string(),
            sha_full: "abc1234567890abcdef".to_string(),
            message: "lost test commit".to_string(),
            author: "Tester".to_string(),
            date: "1 hour ago".to_string(),
            source: "reflog",
        }];
        print_recovery_list(&items);
    }
}
