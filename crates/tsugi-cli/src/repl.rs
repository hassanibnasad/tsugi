//! # Conversational REPL Git Co-Pilot Loop (`tsugi chat` / interactive mode)
//!
//! Provides an interactive terminal session where developers converse naturally
//! with tsugi. Maintains an in-memory rolling session context window (<2MB RAM)
//! so users can iterate on complex Git states conversationally.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use colored::Colorize;
use tsugi_core::{
    explain::explain_command,
    formatter,
    types::{HeadState, SessionTurn},
    PlanOutcome, TerminalObserver, TsugiEngine,
};

const MAX_SESSION_TURNS: usize = 5;

pub struct ReplSession {
    repo_path: PathBuf,
    engine: TsugiEngine,
    turns: VecDeque<SessionTurn>,
}

impl ReplSession {
    pub fn new(repo_path: PathBuf, engine: TsugiEngine) -> Self {
        Self {
            repo_path,
            engine,
            turns: VecDeque::with_capacity(MAX_SESSION_TURNS),
        }
    }

    pub fn start(&mut self) -> Result<()> {
        println!("{}", "次 tsugi — Enterprise Git Co-Pilot (Interactive Session)".cyan().bold());
        println!("{}", "Type your request in plain English. Type 'help' for commands, 'exit' to quit.\n".dimmed());

        // Probe initial repository state and show quick summary
        if let Ok(state) = self.engine.probe(&self.repo_path) {
            let branch_name = match state.head {
                HeadState::Attached { ref branch, .. } => branch.cyan().bold(),
                HeadState::Detached { ref commit_sha } => format!("detached at {}", &commit_sha[..7.min(commit_sha.len())]).yellow().bold(),
                HeadState::Unborn { ref branch } => format!("unborn ({})", branch).dimmed(),
            };
            println!("  {} Repository: {} (Branch: {})", "📁".dimmed(), self.repo_path.display().to_string().bold(), branch_name);
            if state.working_tree.has_changes() {
                println!("  {} Working tree has uncommitted modifications", "⚠".yellow());
            } else {
                println!("  {} Working tree is clean", "✔".green());
            }
            if let Some(ref op) = state.ongoing_operation {
                println!("  {} In-progress operation: {}", "⚡".yellow().bold(), op.to_string().yellow().bold());
            }
            if !state.conflicts.is_empty() {
                println!("  {} Active merge/rebase conflicts in {} file(s)", "💥".red().bold(), state.conflicts.len());
            }
        }
        println!();

        loop {
            print!("{} ", "tsugi ›".cyan().bold());
            io::stdout().flush().ok();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                break;
            }

            let query = input.trim();
            if query.is_empty() {
                continue;
            }

            match query.to_lowercase().as_str() {
                "exit" | "quit" | "q" => {
                    println!("  {} Session ended.", "👋".dimmed());
                    break;
                }
                "help" | "?" => {
                    self.print_help();
                    continue;
                }
                "status" => {
                    if let Ok(state) = self.engine.probe(&self.repo_path) {
                        formatter::print_diagnosis(&state);
                    }
                    continue;
                }
                "undo" => {
                    match self.engine.undo(&self.repo_path) {
                        Ok(Some(snap)) => {
                            println!("  {} Restored repository to pre-fix state ({})", "✔".green().bold(), snap.head_sha[..7.min(snap.head_sha.len())].yellow());
                        }
                        Ok(None) => println!("  {} No snapshots available to restore.", "ℹ".cyan()),
                        Err(e) => eprintln!("  {} Undo failed: {}", "✖".red(), e),
                    }
                    continue;
                }
                "pr-ready" | "pr" => {
                    let _ = crate::pr_check::run_pr_check(&self.repo_path, None)
                        .map(|r| r.print_summary());
                    continue;
                }
                "recover" => {
                    if let Ok(lost) = crate::recover::find_lost_commits(&self.repo_path, 10) {
                        crate::recover::print_recovery_list(&lost);
                    }
                    continue;
                }
                _ => {}
            }

            // Handle explain prefix
            if query.starts_with("explain") {
                let target = query.trim_start_matches("explain").trim();
                let explanation = explain_command(target, None, self.engine.config())?;
                println!("\n  {} {}\n", "💡".cyan(), explanation);
                continue;
            }

            // Regular conversational Git request
            self.handle_turn(query)?;
        }

        Ok(())
    }

    fn handle_turn(&mut self, query: &str) -> Result<()> {
        let state = self.engine.probe(&self.repo_path)?;

        let history: Vec<SessionTurn> = self.turns.iter().cloned().collect();
        let plan_outcome = self.engine.plan_with_history(&state, query, &history)?;

        match plan_outcome {
            PlanOutcome::Ready(plan) => {
                println!();
                formatter::print_plan(&plan);

                print!("\n  {} ", "Execute this resolution plan? [y/N/explain]:".yellow().bold());
                io::stdout().flush().ok();

                let mut answer = String::new();
                io::stdin().read_line(&mut answer)?;
                let ans = answer.trim().to_lowercase();

                if ans == "explain" {
                    println!("\n  {} Detailed Plan Breakdown:", "📖".cyan());
                    for (i, step) in plan.execution_steps.iter().enumerate() {
                        let exp = explain_command(&step.cmd, Some(&state), self.engine.config())?;
                        println!("    [{}] `{}`\n        -> {}\n        -> {}", i + 1, step.cmd.green(), step.explanation.dimmed(), exp.cyan());
                    }

                    print!("\n  {} ", "Proceed with execution now? [y/N]:".yellow().bold());
                    io::stdout().flush().ok();
                    answer.clear();
                    io::stdin().read_line(&mut answer)?;
                }

                if matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
                    println!("\n{}", "─── Executing ─────────────────────────────".dimmed());
                    let observer = TerminalObserver::new(true);
                    match self.engine.execute(&self.repo_path, &plan, query, &observer) {
                        Ok(executed) => {
                            println!("\n  {} Resolution complete ({} steps executed)!", "✔".green().bold(), executed.len());
                            self.record_turn(query, &plan.root_cause, true);
                        }
                        Err(e) => {
                            eprintln!("\n  {} Execution error: {}", "✖".red().bold(), e);
                            self.record_turn(query, &plan.root_cause, false);
                        }
                    }
                } else {
                    println!("  {} Execution skipped.", "✖".dimmed());
                    self.record_turn(query, &plan.root_cause, false);
                }
            }
            PlanOutcome::SetupNeeded => {
                formatter::print_setup_needed();
            }
        }

        println!();
        Ok(())
    }

    fn record_turn(&mut self, query: &str, response_summary: &str, executed: bool) {
        if self.turns.len() >= MAX_SESSION_TURNS {
            self.turns.pop_front();
        }
        self.turns.push_back(SessionTurn {
            user_prompt: query.to_string(),
            assistant_response: response_summary.to_string(),
            executed,
        });
    }

    fn print_help(&self) {
        println!("\n  {} Interactive Co-Pilot Capabilities:", "💬".cyan().bold());
        println!("    {} Diagnose & resolve rebase, merge, or tracking issues", "• Describe problem:".bold());
        println!("    {} Explain what a command does", "• explain <cmd>:   ".bold());
        println!("    {} Audit active branch before opening PR", "• pr-ready:        ".bold());
        println!("    {} Hunt for lost or dangling commits", "• recover:         ".bold());
        println!("    {} Show repository health status", "• status:          ".bold());
        println!("    {} Undo last mutating tsugi action", "• undo:            ".bold());
        println!("    {} Exit the interactive session\n", "• exit / quit:     ".bold());
    }
}
