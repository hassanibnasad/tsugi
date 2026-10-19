//! # Command Executor
//!
//! Executes the resolved Git commands from a [`ResolutionPlan`], one step at a time,
//! with real-time output streaming to the terminal.
//!
//! The executor respects confirmation requirements — dangerous steps are paused
//! until the user explicitly approves them.

use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

use colored::Colorize;
use tracing::{debug, error};

use crate::types::{ResolutionPlan, TsugiError};

/// Execute a resolution plan step by step.
///
/// Returns a list of commands that were actually executed (for audit logging).
pub fn execute_plan(
    plan: &ResolutionPlan,
    repo_path: &Path,
    auto_confirm: bool,
) -> Result<Vec<String>, TsugiError> {
    let mut executed_commands = Vec::new();

    for (i, step) in plan.execution_steps.iter().enumerate() {
        let step_num = i + 1;
        let total = plan.execution_steps.len();

        // Skip comment-only steps (used by Tier-0 for manual conflict resolution hints)
        if step.cmd.starts_with('#') {
            println!(
                "\n{} {} {}",
                format!("[{step_num}/{total}]").dimmed(),
                "📝".to_string(),
                step.explanation.yellow()
            );
            continue;
        }

        println!(
            "\n{} {} {}",
            format!("[{step_num}/{total}]").dimmed(),
            "▶".green().bold(),
            step.cmd.white().bold()
        );
        println!("  {}", step.explanation.dimmed());

        // Handle confirmation for dangerous steps
        if step.requires_confirmation && !auto_confirm {
            print!(
                "\n  {} ",
                "Execute this command? [y/N/s(kip)]:"
                    .yellow()
                    .bold()
            );
            io::stdout().flush().ok();

            let mut input = String::new();
            io::stdin().read_line(&mut input).map_err(|e| {
                TsugiError::ExecutionFailed {
                    cmd: step.cmd.clone(),
                    reason: format!("Failed to read user input: {e}"),
                }
            })?;

            match input.trim().to_lowercase().as_str() {
                "y" | "yes" => {}
                "s" | "skip" => {
                    println!("  {} Skipped", "⏭".dimmed());
                    continue;
                }
                _ => {
                    println!("  {} Aborted by user", "✖".red().bold());
                    return Err(TsugiError::ExecutionFailed {
                        cmd: step.cmd.clone(),
                        reason: "Aborted by user".to_string(),
                    });
                }
            }
        }

        // Execute the command
        let result = execute_git_command(&step.cmd, repo_path);

        match result {
            Ok(output) => {
                if !output.is_empty() {
                    // Indent the output for visual clarity
                    for line in output.lines() {
                        println!("  {}", line.dimmed());
                    }
                }
                println!("  {} Done", "✔".green().bold());
                executed_commands.push(step.cmd.clone());
            }
            Err(e) => {
                error!("Command failed: {}", step.cmd);
                println!("  {} Failed: {}", "✖".red().bold(), e);

                // Offer rollback
                println!(
                    "\n  {} {}",
                    "Rollback available:".yellow().bold(),
                    plan.rollback_cmd.white()
                );
                print!(
                    "  {} ",
                    "Run rollback? [Y/n]:".yellow().bold()
                );
                io::stdout().flush().ok();

                let mut input = String::new();
                io::stdin().read_line(&mut input).ok();

                match input.trim().to_lowercase().as_str() {
                    "n" | "no" => {
                        println!("  {} Rollback skipped", "⏭".dimmed());
                    }
                    _ => {
                        println!("  {} Running rollback...", "↩".yellow());
                        if let Ok(output) = execute_git_command(&plan.rollback_cmd, repo_path) {
                            if !output.is_empty() {
                                for line in output.lines() {
                                    println!("  {}", line.dimmed());
                                }
                            }
                            println!("  {} Rollback complete", "✔".green().bold());
                        } else {
                            println!(
                                "  {} Rollback failed — run `tsugi undo` for full restore",
                                "✖".red().bold()
                            );
                        }
                    }
                }

                return Err(TsugiError::ExecutionFailed {
                    cmd: step.cmd.clone(),
                    reason: e.to_string(),
                });
            }
        }
    }

    Ok(executed_commands)
}

/// Execute a single Git command and return its stdout.
fn execute_git_command(cmd: &str, repo_path: &Path) -> Result<String, TsugiError> {
    debug!("Executing: {cmd} in {}", repo_path.display());

    // Parse the command string into program + args
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return Err(TsugiError::ExecutionFailed {
            cmd: cmd.to_string(),
            reason: "Empty command".to_string(),
        });
    }

    let output = Command::new(parts[0])
        .args(&parts[1..])
        .current_dir(repo_path)
        .output()
        .map_err(|e| TsugiError::ExecutionFailed {
            cmd: cmd.to_string(),
            reason: format!("Failed to spawn process: {e}"),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        // Some Git commands output useful info to stderr even on success codes
        let error_msg = if !stderr.is_empty() {
            stderr.trim().to_string()
        } else {
            format!("Exit code: {}", output.status)
        };

        return Err(TsugiError::ExecutionFailed {
            cmd: cmd.to_string(),
            reason: error_msg,
        });
    }

    // Git often writes progress/info to stderr even on success
    if !stderr.is_empty() {
        debug!("stderr (non-fatal): {}", stderr.trim());
    }

    Ok(stdout)
}
