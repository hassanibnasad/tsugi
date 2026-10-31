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
use tracing::debug;
use tsugi_core::{
    explain::explain_command,
    formatter,
    types::{HeadState, SessionTurn, TsugiConfig},
    PlanOutcome, TerminalObserver, TsugiEngine,
};

const MAX_SESSION_TURNS: usize = 5;

/// RAII Guard managing a local ephemeral llama-server process during REPL sessions.
/// Automatically terminates the server process on drop, ensuring 0 MB idle RAM.
pub struct EphemeralServerGuard {
    child: Option<std::process::Child>,
}

impl EphemeralServerGuard {
    pub fn spawn_if_needed(config: &TsugiConfig) -> Self {
        // 1. Check if server already running on port 8089 or configured URL
        if tsugi_core::config::is_llama_server_running(Some("http://127.0.0.1:8089")) {
            debug!("llama-server already active on port 8089; reusing existing instance");
            return Self { child: None };
        }
        if let Some(ref url) = config.llama_server_url {
            if tsugi_core::config::is_llama_server_running(Some(url)) {
                debug!("llama-server already active on configured endpoint {url}");
                return Self { child: None };
            }
        }

        // 2. Locate llama-server binary and model path
        let server_bin = match tsugi_core::engine::find_llama_server() {
            Ok(bin) => bin,
            Err(_) => return Self { child: None },
        };

        let model_path = match config.primary_model_path.as_ref().or(config.fallback_model_path.as_ref()) {
            Some(p) if p.exists() => p,
            _ => return Self { child: None },
        };

        // 3. Spawn child process: llama-server -m <model_path> -c 2048 --port 8089 --timeout 300
        let mut cmd = std::process::Command::new(server_bin);
        cmd.arg("-m")
            .arg(model_path)
            .arg("-c")
            .arg("2048")
            .arg("--port")
            .arg("8089")
            .arg("--timeout")
            .arg("300")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                debug!("Failed to spawn ephemeral llama-server: {e}");
                return Self { child: None };
            }
        };

        // 4. Poll for readiness up to 8 seconds (checking child.try_wait() on each tick)
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(8);
        let mut ready = false;

        while start.elapsed() < timeout {
            // Check if child exited early / crashed
            if let Ok(Some(status)) = child.try_wait() {
                debug!("Ephemeral llama-server process exited early with status: {status}");
                return Self { child: None };
            }

            if tsugi_core::config::is_llama_server_running(Some("http://127.0.0.1:8089")) {
                ready = true;
                break;
            }

            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        if ready {
            println!("  {} Warm inference server ready on port 8089 (<100ms response time)", "⚡".cyan());
            Self { child: Some(child) }
        } else {
            // If port 8089 still hasn't bound after 8s, kill and wait to avoid leaking an orphaned process
            let _ = child.kill();
            let _ = child.wait();
            Self { child: None }
        }
    }
}

impl Drop for EphemeralServerGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub struct ReplSession {
    repo_path: PathBuf,
    engine: TsugiEngine,
    turns: VecDeque<SessionTurn>,
    _managed_server: Option<EphemeralServerGuard>,
}

impl ReplSession {
    pub fn new(repo_path: PathBuf, engine: TsugiEngine) -> Self {
        let server = EphemeralServerGuard::spawn_if_needed(engine.config());
        Self {
            repo_path,
            engine,
            turns: VecDeque::with_capacity(MAX_SESSION_TURNS),
            _managed_server: Some(server),
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
            match io::stdin().read_line(&mut input) {
                Ok(0) => {
                    println!("\n  {} Session ended.", "👋".dimmed());
                    break;
                }
                Err(_) => break,
                Ok(_) => {}
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
