//! # Command Executor
//!
//! Executes the resolved Git commands from a [`ResolutionPlan`], one step at a time,
//! reporting progress and requesting confirmations across the [`ExecutionObserver`] seam.

use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

use colored::Colorize;
use tracing::{debug, error};

use crate::types::{ExecutionStep, ResolutionPlan, TsugiError};

/// Outcome of a confirmation request for a command step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmationResult {
    /// Proceed with executing the step.
    Proceed,
    /// Skip this step and proceed to subsequent steps.
    Skip,
    /// Abort plan execution immediately.
    Abort,
}

/// Seam separating command execution from presentation and user interaction.
pub trait ExecutionObserver: Send + Sync {
    /// Called when plan execution starts.
    fn on_plan_start(&self, _plan: &ResolutionPlan) {}

    /// Called before executing an individual step.
    fn on_step_start(&self, _step_num: usize, _total: usize, _step: &ExecutionStep) {}

    /// Called when a command step produces output.
    fn on_step_output(&self, _line: &str) {}

    /// Request user confirmation for dangerous or flagged steps.
    fn request_confirmation(&self, _step: &ExecutionStep) -> ConfirmationResult {
        ConfirmationResult::Proceed
    }

    /// Called when a step completes.
    fn on_step_complete(&self, _step_num: usize, _step: &ExecutionStep, _success: bool) {}

    /// Called when the entire plan finishes executing.
    fn on_plan_complete(&self, _plan: &ResolutionPlan, _executed_count: usize) {}
}

/// Terminal observer displaying ANSI-colored steps and reading confirmation from stdin.
#[derive(Debug, Clone)]
pub struct TerminalObserver {
    auto_confirm: bool,
}

impl TerminalObserver {
    pub fn new(auto_confirm: bool) -> Self {
        Self { auto_confirm }
    }
}

impl ExecutionObserver for TerminalObserver {
    fn on_step_start(&self, step_num: usize, total: usize, step: &ExecutionStep) {
        if step.cmd.starts_with('#') {
            println!(
                "\n{} {} {}",
                format!("[{step_num}/{total}]").dimmed(),
                "📝",
                step.explanation.yellow()
            );
        } else {
            println!(
                "\n{} {} {}",
                format!("[{step_num}/{total}]").dimmed(),
                "▶".green().bold(),
                step.cmd.white().bold()
            );
            println!("  {}", step.explanation.dimmed());
        }
    }

    fn on_step_output(&self, line: &str) {
        println!("  {}", line.dimmed());
    }

    fn request_confirmation(&self, step: &ExecutionStep) -> ConfirmationResult {
        if self.auto_confirm || !step.requires_confirmation {
            return ConfirmationResult::Proceed;
        }

        print!(
            "\n  {} ",
            "Execute this command? [y/N/s(kip)]:".yellow().bold()
        );
        io::stdout().flush().ok();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return ConfirmationResult::Abort;
        }

        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => ConfirmationResult::Proceed,
            "s" | "skip" => ConfirmationResult::Skip,
            _ => ConfirmationResult::Abort,
        }
    }

    fn on_step_complete(&self, _step_num: usize, step: &ExecutionStep, success: bool) {
        if !step.cmd.starts_with('#') {
            if success {
                println!("  {} Done", "✔".green().bold());
            } else {
                println!("  {} Failed", "✖".red().bold());
            }
        }
    }
}

/// In-memory observer capturing execution events, suitable for automated tests.
#[derive(Debug, Default, Clone)]
pub struct CapturingObserver {
    pub started_steps: Arc<Mutex<Vec<String>>>,
    pub output_lines: Arc<Mutex<Vec<String>>>,
    pub completed_steps: Arc<Mutex<Vec<String>>>,
    auto_confirm: bool,
}

impl CapturingObserver {
    pub fn new(auto_confirm: bool) -> Self {
        Self {
            started_steps: Arc::new(Mutex::new(Vec::new())),
            output_lines: Arc::new(Mutex::new(Vec::new())),
            completed_steps: Arc::new(Mutex::new(Vec::new())),
            auto_confirm,
        }
    }
}

impl ExecutionObserver for CapturingObserver {
    fn on_step_start(&self, _step_num: usize, _total: usize, step: &ExecutionStep) {
        self.started_steps.lock().unwrap().push(step.cmd.clone());
    }

    fn on_step_output(&self, line: &str) {
        self.output_lines.lock().unwrap().push(line.to_string());
    }

    fn request_confirmation(&self, _step: &ExecutionStep) -> ConfirmationResult {
        if self.auto_confirm {
            ConfirmationResult::Proceed
        } else {
            ConfirmationResult::Abort
        }
    }

    fn on_step_complete(&self, _step_num: usize, step: &ExecutionStep, success: bool) {
        if success {
            self.completed_steps.lock().unwrap().push(step.cmd.clone());
        }
    }
}

/// Execute a resolution plan with an observer adapter.
pub fn execute_plan_with_observer(
    plan: &ResolutionPlan,
    repo_path: &Path,
    observer: &dyn ExecutionObserver,
) -> Result<Vec<String>, TsugiError> {
    let mut executed_commands = Vec::new();
    let total = plan.execution_steps.len();

    observer.on_plan_start(plan);

    for (i, step) in plan.execution_steps.iter().enumerate() {
        let step_num = i + 1;
        observer.on_step_start(step_num, total, step);

        // Skip comments
        if step.cmd.starts_with('#') {
            continue;
        }

        // Handle confirmation
        if step.requires_confirmation {
            match observer.request_confirmation(step) {
                ConfirmationResult::Proceed => {}
                ConfirmationResult::Skip => {
                    continue;
                }
                ConfirmationResult::Abort => {
                    return Err(TsugiError::ExecutionFailed {
                        cmd: step.cmd.clone(),
                        reason: "Aborted by user".to_string(),
                    });
                }
            }
        }

        // Execute command
        match execute_git_command(&step.cmd, repo_path) {
            Ok(output) => {
                if !output.is_empty() {
                    for line in output.lines() {
                        observer.on_step_output(line);
                    }
                }
                observer.on_step_complete(step_num, step, true);
                executed_commands.push(step.cmd.clone());
            }
            Err(e) => {
                error!("Command failed: {}", step.cmd);
                observer.on_step_complete(step_num, step, false);
                return Err(TsugiError::ExecutionFailed {
                    cmd: step.cmd.clone(),
                    reason: e.to_string(),
                });
            }
        }
    }

    observer.on_plan_complete(plan, executed_commands.len());
    Ok(executed_commands)
}

/// Convenience entry point using the interactive TerminalObserver.
pub fn execute_plan(
    plan: &ResolutionPlan,
    repo_path: &Path,
    auto_confirm: bool,
) -> Result<Vec<String>, TsugiError> {
    let observer = TerminalObserver::new(auto_confirm);
    execute_plan_with_observer(plan, repo_path, &observer)
}

/// Execute a single Git command and return its stdout.
fn execute_git_command(cmd: &str, repo_path: &Path) -> Result<String, TsugiError> {
    debug!("Executing: {cmd} in {}", repo_path.display());

    let parts = shlex::split(cmd).ok_or_else(|| TsugiError::ExecutionFailed {
        cmd: cmd.to_string(),
        reason: "Failed to parse command arguments: mismatched quotes or escapes".to_string(),
    })?;

    if parts.is_empty() {
        return Err(TsugiError::ExecutionFailed {
            cmd: cmd.to_string(),
            reason: "Empty command".to_string(),
        });
    }

    if parts[0] != "git" {
        return Err(TsugiError::ExecutionFailed {
            cmd: cmd.to_string(),
            reason: format!("Refusing to execute non-git command: '{}'", parts[0]),
        });
    }

    let output = Command::new(&parts[0])
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

    if !stderr.is_empty() {
        debug!("stderr (non-fatal): {}", stderr.trim());
    }

    Ok(stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PlanSource, RiskTier};
    use tempfile::TempDir;

    #[test]
    fn test_capturing_observer_executes_steps() {
        let tmp = TempDir::new().unwrap();
        // Initialize git repo so git status works
        let _repo = git2::Repository::init(tmp.path()).unwrap();

        let plan = ResolutionPlan {
            root_cause: "Test plan".to_string(),
            risk_tier: RiskTier::ReadOnly,
            execution_steps: vec![
                ExecutionStep {
                    cmd: "git status".to_string(),
                    explanation: "Check git status".to_string(),
                    requires_confirmation: false,
                },
                ExecutionStep {
                    cmd: "# comment step".to_string(),
                    explanation: "Documentation".to_string(),
                    requires_confirmation: false,
                },
            ],
            rollback_cmd: "none".to_string(),
            source: PlanSource::Tier0,
            confidence: 1.0,
        };

        let observer = CapturingObserver::new(true);
        let executed = execute_plan_with_observer(&plan, tmp.path(), &observer).unwrap();

        assert_eq!(executed.len(), 1);
        assert_eq!(executed[0], "git status");
        assert_eq!(observer.started_steps.lock().unwrap().len(), 2);
        assert_eq!(observer.completed_steps.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_capturing_observer_confirmation_abort() {
        let tmp = TempDir::new().unwrap();
        let plan = ResolutionPlan {
            root_cause: "Dangerous plan".to_string(),
            risk_tier: RiskTier::MutatingDangerous,
            execution_steps: vec![ExecutionStep {
                cmd: "git reset --hard".to_string(),
                explanation: "Hard reset".to_string(),
                requires_confirmation: true,
            }],
            rollback_cmd: "none".to_string(),
            source: PlanSource::Tier0,
            confidence: 1.0,
        };

        // auto_confirm = false -> should abort
        let observer = CapturingObserver::new(false);
        let result = execute_plan_with_observer(&plan, tmp.path(), &observer);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_git_command_rejects_non_git() {
        let tmp = TempDir::new().unwrap();
        let plan = ResolutionPlan {
            root_cause: "Malicious plan".to_string(),
            risk_tier: RiskTier::MutatingDangerous,
            execution_steps: vec![ExecutionStep {
                cmd: "rm -rf /".to_string(),
                explanation: "Delete files".to_string(),
                requires_confirmation: false,
            }],
            rollback_cmd: "none".to_string(),
            source: PlanSource::Tier0,
            confidence: 1.0,
        };

        let observer = CapturingObserver::new(true);
        let result = execute_plan_with_observer(&plan, tmp.path(), &observer);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-git"));
    }

    #[test]
    fn test_execute_git_command_handles_quoted_arguments() {
        let tmp = TempDir::new().unwrap();
        let repo = git2::Repository::init(tmp.path()).unwrap();

        // Create an initial commit
        let test_file = tmp.path().join("dummy.txt");
        std::fs::write(&test_file, "initial").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("dummy.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[]).unwrap();

        // Modify file to stash
        std::fs::write(&test_file, "modified content").unwrap();

        let plan = ResolutionPlan {
            root_cause: "Stash test".to_string(),
            risk_tier: RiskTier::MutatingSafe,
            execution_steps: vec![ExecutionStep {
                cmd: "git stash push -m \"tsugi safety backup message\"".to_string(),
                explanation: "Stash with spaces in quotes".to_string(),
                requires_confirmation: false,
            }],
            rollback_cmd: "git stash pop".to_string(),
            source: PlanSource::Tier0,
            confidence: 1.0,
        };

        let observer = CapturingObserver::new(true);
        let executed = execute_plan_with_observer(&plan, tmp.path(), &observer).unwrap();
        assert_eq!(executed.len(), 1);
    }
}
