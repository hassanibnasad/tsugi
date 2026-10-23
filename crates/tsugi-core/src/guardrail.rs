//! # Layer 4a: Deterministic Policy Guardrail
//!
//! Scans every command in a [`ResolutionPlan`] before execution. Destructive operations
//! are intercepted and either:
//! - **Rewritten** to safer equivalents (e.g., `--force` → `--force-with-lease`)
//! - **Blocked** with an explanation if no safe equivalent exists
//! - **Flagged** for mandatory user confirmation
//!
//! This layer is purely deterministic — it uses regex/string matching, not AI inference.
//! It is the last line of defense between the model's output and the user's repository.

use tracing::{info, warn};

use crate::types::{ResolutionPlan, RiskTier, TsugiConfig, TsugiError};

/// Policy rules applied to every command before execution.
struct PolicyRule {
    /// Pattern to match in the command string.
    pattern: &'static str,
    /// Action to take when matched.
    action: PolicyAction,
    /// Human-readable reason for the policy.
    reason: &'static str,
}

enum PolicyAction {
    /// Block the command entirely.
    Block,
    /// Force user confirmation before execution.
    RequireConfirmation,
    /// Escalate the risk tier.
    EscalateRisk(RiskTier),
}

/// The complete set of safety policies.
static POLICIES: &[PolicyRule] = &[
    // ── Hard Reset ────────────────────────────────────────────────────
    PolicyRule {
        pattern: "git reset --hard",
        action: PolicyAction::RequireConfirmation,
        reason: "git reset --hard will permanently discard uncommitted changes",
    },
    // ── Clean ─────────────────────────────────────────────────────────
    PolicyRule {
        pattern: "git clean -fd",
        action: PolicyAction::RequireConfirmation,
        reason: "git clean will permanently delete untracked files",
    },
    PolicyRule {
        pattern: "git clean -fx",
        action: PolicyAction::RequireConfirmation,
        reason: "git clean -fx will delete untracked AND ignored files (including build artifacts)",
    },
    // ── Branch Deletion ──────────────────────────────────────────────
    PolicyRule {
        pattern: "git branch -D",
        action: PolicyAction::RequireConfirmation,
        reason: "Force-deleting a branch may lose unmerged commits",
    },
    // ── Checkout with discard ────────────────────────────────────────
    PolicyRule {
        pattern: "git checkout -- .",
        action: PolicyAction::RequireConfirmation,
        reason: "This will discard ALL unstaged changes in every tracked file",
    },
    // ── Force push to main/master ────────────────────────────────────
    PolicyRule {
        pattern: "git push --force-with-lease origin main",
        action: PolicyAction::EscalateRisk(RiskTier::HistoryRewrite),
        reason: "Force-pushing to main/master rewrites shared history",
    },
    PolicyRule {
        pattern: "git push --force-with-lease origin master",
        action: PolicyAction::EscalateRisk(RiskTier::HistoryRewrite),
        reason: "Force-pushing to main/master rewrites shared history",
    },
    // ── Stash drop ───────────────────────────────────────────────────
    PolicyRule {
        pattern: "git stash drop",
        action: PolicyAction::RequireConfirmation,
        reason: "Dropping a stash permanently deletes the stashed changes",
    },
    // ── Reflog expire ────────────────────────────────────────────────
    PolicyRule {
        pattern: "git reflog expire",
        action: PolicyAction::Block,
        reason: "Expiring reflog entries removes the safety net for recovery",
    },
    // ── GC prune ─────────────────────────────────────────────────────
    PolicyRule {
        pattern: "git gc --prune",
        action: PolicyAction::Block,
        reason: "Pruning garbage collection removes unreachable objects needed for recovery",
    },
    // ── Filter-branch / filter-repo ──────────────────────────────────
    PolicyRule {
        pattern: "git filter-branch",
        action: PolicyAction::Block,
        reason: "filter-branch rewrites entire history — use BFG Repo-Cleaner or git-filter-repo instead",
    },
];

/// Check if a space-separated pattern appears as a contiguous token sequence in the command.
fn matches_command_pattern(cmd: &str, pattern: &str) -> bool {
    let cmd_tokens = shlex::split(cmd).unwrap_or_else(|| cmd.split_whitespace().map(|s| s.to_string()).collect());
    let pat_tokens = shlex::split(pattern).unwrap_or_else(|| pattern.split_whitespace().map(|s| s.to_string()).collect());
    if pat_tokens.is_empty() || cmd_tokens.is_empty() || pat_tokens.len() > cmd_tokens.len() {
        return false;
    }
    cmd_tokens
        .windows(pat_tokens.len())
        .any(|window| window == pat_tokens.as_slice())
}

/// Deterministically derive the risk tier and confirmation requirement from parsed command tokens.
pub fn derive_command_risk(cmd_tokens: &[String]) -> (RiskTier, bool, Option<String>) {
    if cmd_tokens.is_empty() || cmd_tokens[0] != "git" {
        return (RiskTier::ReadOnly, false, None);
    }

    if cmd_tokens.len() == 1 {
        return (RiskTier::ReadOnly, false, None);
    }

    let subcmd = cmd_tokens[1].as_str();

    match subcmd {
        // Read-only inspection commands
        "status" | "diff" | "log" | "show" | "rev-parse" | "cat-file" | "ls-files" => {
            (RiskTier::ReadOnly, false, None)
        }

        // Branch operations
        "branch" => {
            let is_force_delete = cmd_tokens.iter().any(|t| t == "-D")
                || (cmd_tokens.iter().any(|t| t == "-d" || t == "--delete")
                    && cmd_tokens.iter().any(|t| t == "-f" || t == "--force"));
            if is_force_delete {
                (
                    RiskTier::MutatingDangerous,
                    true,
                    Some("Force-deleting a branch can permanently lose unmerged commits".into()),
                )
            } else if cmd_tokens.len() == 2
                || cmd_tokens.iter().any(|t| t == "--list" || t == "-a" || t == "-r")
            {
                (RiskTier::ReadOnly, false, None)
            } else {
                (RiskTier::MutatingSafe, false, None)
            }
        }

        // Clean: any non-dry-run clean is dangerous
        "clean" => {
            let is_dry_run = cmd_tokens.iter().any(|t| t == "-n" || t == "--dry-run");
            if is_dry_run {
                (RiskTier::ReadOnly, false, None)
            } else {
                let deletes_ignored = cmd_tokens.iter().any(|t| {
                    t == "-x" || t == "-X" || (t.starts_with('-') && (t.contains('x') || t.contains('X')))
                });
                let reason = if deletes_ignored {
                    "git clean with -x will permanently delete untracked AND ignored files (including build artifacts, keys, and environment files)"
                } else {
                    "git clean will permanently delete untracked files from disk"
                };
                (RiskTier::MutatingDangerous, true, Some(reason.into()))
            }
        }

        // Reset
        "reset" => {
            let is_hard_or_merge = cmd_tokens.iter().any(|t| t == "--hard" || t == "--merge");
            if is_hard_or_merge {
                (
                    RiskTier::MutatingDangerous,
                    true,
                    Some("git reset --hard/--merge will permanently discard uncommitted changes".into()),
                )
            } else {
                (RiskTier::MutatingSafe, false, None)
            }
        }

        // Push
        "push" => {
            let is_force = cmd_tokens.iter().any(|t| {
                t == "--force"
                    || t == "-f"
                    || t == "--force-with-lease"
                    || (t.starts_with('+') && t.len() > 1 && !t.starts_with("++"))
            });
            let is_delete = cmd_tokens.iter().any(|t| t == "--delete" || (t.starts_with(':') && t.len() > 1));
            let targets_main = cmd_tokens.iter().any(|t| {
                t == "main" || t == "master" || t.ends_with("/main") || t.ends_with("/master")
            });

            if is_force || is_delete {
                let reason = if targets_main {
                    "Force-pushing or deleting remote main/master branch rewrites shared history"
                } else {
                    "Force-pushing rewrites remote branch history"
                };
                (RiskTier::HistoryRewrite, true, Some(reason.into()))
            } else {
                (RiskTier::MutatingSafe, false, None)
            }
        }

        // Checkout & restore
        "checkout" => {
            let is_force_discard = cmd_tokens.iter().any(|t| t == "-f" || t == "--force" || t == "--discard-changes")
                || (cmd_tokens.iter().any(|t| t == "--") && cmd_tokens.iter().any(|t| t == "." || t == "*"));
            if is_force_discard {
                (
                    RiskTier::MutatingDangerous,
                    true,
                    Some("git checkout with force or discard flag will permanently overwrite working tree changes".into()),
                )
            } else {
                (RiskTier::MutatingSafe, false, None)
            }
        }

        "restore" => {
            let is_staged_only = cmd_tokens.iter().any(|t| t == "--staged" || t == "-S")
                && !cmd_tokens.iter().any(|t| t == "--worktree" || t == "-W");
            if is_staged_only {
                (RiskTier::MutatingSafe, false, None)
            } else {
                let is_all = cmd_tokens.iter().any(|t| t == "." || t == "*");
                if is_all {
                    (
                        RiskTier::MutatingDangerous,
                        true,
                        Some("git restore . will discard all unstaged changes across the repository".into()),
                    )
                } else {
                    (RiskTier::MutatingSafe, false, None)
                }
            }
        }

        // Stash
        "stash" => {
            let is_drop_or_clear = cmd_tokens.iter().any(|t| t == "drop" || t == "clear");
            if is_drop_or_clear {
                (
                    RiskTier::MutatingDangerous,
                    true,
                    Some("Dropping or clearing stashes permanently deletes stashed changes".into()),
                )
            } else if cmd_tokens.len() == 2 && cmd_tokens.iter().any(|t| t == "list") {
                (RiskTier::ReadOnly, false, None)
            } else {
                (RiskTier::MutatingSafe, false, None)
            }
        }

        // Blocked dangerous commands
        "filter-branch" | "filter-repo" => {
            (RiskTier::HistoryRewrite, true, Some("filter-branch rewrites entire repository history".into()))
        }

        // Default mutating commands
        "add" | "commit" | "switch" | "merge" | "rebase" | "cherry-pick" | "revert" | "tag" | "fetch" | "pull" => {
            (RiskTier::MutatingSafe, false, None)
        }

        _ => (RiskTier::MutatingSafe, false, None),
    }
}

/// Apply all safety policies to a resolution plan.
///
/// This function:
/// 1. Scans every command against the policy rule set and command risk analyzer.
/// 2. Rewrites unsafe commands (including --force, -f, and +refspec) to --force-with-lease.
/// 3. Marks dangerous commands as requiring confirmation.
/// 4. Blocks commands that should never be auto-executed.
/// 5. Deterministically derives and escalates the risk tier based on actual commands.
///
/// Returns the modified plan, or an error if a command is blocked.
pub fn apply_guardrails(
    mut plan: ResolutionPlan,
    config: &TsugiConfig,
) -> Result<ResolutionPlan, TsugiError> {
    // Derive plan risk tier deterministically from parsed commands rather than trusting model output
    let mut highest_risk = RiskTier::ReadOnly;
    let mut modified_steps = Vec::with_capacity(plan.execution_steps.len());

    for mut step in plan.execution_steps {
        // Skip comment steps
        if step.cmd.trim_start().starts_with('#') {
            modified_steps.push(step);
            continue;
        }

        // Validate that the command is a git command using shlex
        let mut cmd_tokens = match shlex::split(&step.cmd) {
            Some(tokens) if !tokens.is_empty() => tokens,
            Some(_) => {
                return Err(TsugiError::PolicyViolation(
                    "Empty command step encountered".to_string(),
                ));
            }
            None => {
                return Err(TsugiError::PolicyViolation(format!(
                    "Command '{}' contains invalid quoting or escaping",
                    step.cmd
                )));
            }
        };

        if cmd_tokens[0] != "git" {
            return Err(TsugiError::PolicyViolation(format!(
                "Command '{}' was blocked: only 'git' commands are permitted",
                step.cmd
            )));
        }

        // Handle force-push rewriting on token level (flags and +refspec)
        if cmd_tokens.len() >= 2 && cmd_tokens[0] == "git" && cmd_tokens[1] == "push" {
            let mut rewritten = false;
            let mut has_lease = false;
            let mut new_tokens = Vec::with_capacity(cmd_tokens.len() + 1);

            for tok in &cmd_tokens {
                if tok == "--force" || tok == "-f" {
                    rewritten = true;
                } else if tok == "--force-with-lease" {
                    has_lease = true;
                    new_tokens.push(tok.clone());
                } else if tok.starts_with('+') && tok.len() > 1 && !tok.starts_with("++") {
                    rewritten = true;
                    new_tokens.push(tok.trim_start_matches('+').to_string());
                } else {
                    new_tokens.push(tok.clone());
                }
            }

            if rewritten {
                if !has_lease {
                    if let Some(pos) = new_tokens.iter().position(|t| t == "push") {
                        new_tokens.insert(pos + 1, "--force-with-lease".to_string());
                    } else {
                        new_tokens.push("--force-with-lease".to_string());
                    }
                }
                let original = step.cmd.clone();
                step.cmd = shlex::try_join(new_tokens.iter().map(|s| s.as_str()))
                    .unwrap_or_else(|_| new_tokens.join(" "));
                info!(
                    "Guardrail: rewrote '{}' → '{}'. Reason: Replaced force-push with --force-with-lease to protect shared history",
                    original, step.cmd
                );
                step.explanation = format!(
                    "{} [GUARDRAIL: Replaced force-push with --force-with-lease to protect shared history]",
                    step.explanation
                );
                cmd_tokens = new_tokens;
            }
        }

        // Derive risk tier and confirmation requirements deterministically from command tokens
        let (derived_tier, requires_confirm, reason) = derive_command_risk(&cmd_tokens);
        if derived_tier.as_u8() > highest_risk.as_u8() {
            highest_risk = derived_tier;
        }

        if requires_confirm {
            step.requires_confirmation = true;
            if let Some(ref r) = reason {
                if config.require_confirmation_for_dangerous {
                    step.explanation = format!("{} [⚠ REQUIRES CONFIRMATION: {}]", step.explanation, r);
                }
                warn!("Guardrail: flagged '{}' for confirmation. Reason: {}", step.cmd, r);
            }
        }

        // Check against static policies for blocked commands
        for rule in POLICIES {
            if matches_command_pattern(&step.cmd, rule.pattern) {
                match &rule.action {
                    PolicyAction::Block => {
                        return Err(TsugiError::PolicyViolation(format!(
                            "Command '{}' was blocked by safety policy: {}",
                            step.cmd, rule.reason
                        )));
                    }

                    PolicyAction::RequireConfirmation => {
                        step.requires_confirmation = true;
                        if config.require_confirmation_for_dangerous {
                            step.explanation = format!(
                                "{} [⚠ REQUIRES CONFIRMATION: {}]",
                                step.explanation, rule.reason
                            );
                        }
                        warn!("Guardrail: flagged '{}' for confirmation. Reason: {}", step.cmd, rule.reason);
                    }

                    PolicyAction::EscalateRisk(tier) => {
                        if tier.as_u8() > highest_risk.as_u8() {
                            highest_risk = *tier;
                        }
                        step.requires_confirmation = true;
                        step.explanation = format!(
                            "{} [⚠ RISK ESCALATED to {}: {}]",
                            step.explanation, tier, rule.reason
                        );
                        warn!("Guardrail: escalated risk for '{}'. Reason: {}", step.cmd, rule.reason);
                    }
                }
            }
        }

        modified_steps.push(step);
    }

    plan.execution_steps = modified_steps;
    plan.risk_tier = highest_risk;

    Ok(plan)
}

// Implement comparison for RiskTier ordering
impl RiskTier {
    fn as_u8(self) -> u8 {
        match self {
            RiskTier::ReadOnly => 0,
            RiskTier::MutatingSafe => 1,
            RiskTier::MutatingDangerous => 2,
            RiskTier::HistoryRewrite => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn test_plan(steps: Vec<ExecutionStep>) -> ResolutionPlan {
        ResolutionPlan {
            root_cause: "test".to_string(),
            risk_tier: RiskTier::MutatingSafe,
            execution_steps: steps,
            rollback_cmd: "git reset".to_string(),
            source: PlanSource::Tier0,
            confidence: 1.0,
        }
    }

    fn test_step(cmd: &str) -> ExecutionStep {
        ExecutionStep {
            cmd: cmd.to_string(),
            explanation: "test".to_string(),
            requires_confirmation: false,
        }
    }

    #[test]
    fn test_force_push_rewrite() {
        let plan = test_plan(vec![test_step("git push --force origin feature")]);
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config).unwrap();
        assert!(result.execution_steps[0].cmd.contains("--force-with-lease"));
        assert!(!result.execution_steps[0].cmd.contains("--force "));
    }

    #[test]
    fn test_hard_reset_requires_confirmation() {
        let plan = test_plan(vec![test_step("git reset --hard HEAD~3")]);
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config).unwrap();
        assert!(result.execution_steps[0].requires_confirmation);
    }

    #[test]
    fn test_filter_branch_blocked() {
        let plan = test_plan(vec![test_step("git filter-branch --prune-empty")]);
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TsugiError::PolicyViolation(_)));
    }

    #[test]
    fn test_safe_commands_pass_through() {
        let plan = test_plan(vec![
            test_step("git rebase --continue"),
            test_step("git add src/file.rs"),
            test_step("git commit -m \"fix merge\""),
        ]);
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config).unwrap();
        assert_eq!(result.execution_steps.len(), 3);
        assert!(!result.execution_steps[0].requires_confirmation);
        assert!(!result.execution_steps[1].requires_confirmation);
        assert!(!result.execution_steps[2].requires_confirmation);
    }

    #[test]
    fn test_non_git_command_blocked() {
        let plan = test_plan(vec![test_step("bash -c 'evil command'")]);
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TsugiError::PolicyViolation(_)));
    }

    #[test]
    fn test_comment_step_allowed() {
        let plan = test_plan(vec![
            test_step("# This is a manual instruction"),
            test_step("git status"),
        ]);
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clean_fdx_requires_confirmation_and_escalates_risk() {
        let mut plan = test_plan(vec![test_step("git clean -fdx")]);
        plan.risk_tier = RiskTier::ReadOnly; // Model mistakenly reported ReadOnly
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config).unwrap();
        assert_eq!(result.risk_tier, RiskTier::MutatingDangerous);
        assert!(result.execution_steps[0].requires_confirmation);
    }

    #[test]
    fn test_push_plus_refspec_rewritten_and_escalates() {
        let mut plan = test_plan(vec![test_step("git push origin +main")]);
        plan.risk_tier = RiskTier::MutatingSafe;
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config).unwrap();
        assert_eq!(result.risk_tier, RiskTier::HistoryRewrite);
        assert!(result.execution_steps[0].cmd.contains("--force-with-lease"));
        assert!(!result.execution_steps[0].cmd.contains("+main"));
        assert!(result.execution_steps[0].requires_confirmation);
    }

    #[test]
    fn test_derived_risk_overrides_model_read_only() {
        let mut plan = test_plan(vec![test_step("git reset --hard HEAD~1")]);
        plan.risk_tier = RiskTier::ReadOnly;
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config).unwrap();
        assert_eq!(result.risk_tier, RiskTier::MutatingDangerous);
        assert!(result.execution_steps[0].requires_confirmation);
    }

    #[test]
    fn test_branch_delete_force_requires_confirmation() {
        let mut plan = test_plan(vec![test_step("git branch -D feature")]);
        plan.risk_tier = RiskTier::MutatingSafe;
        let config = TsugiConfig::default();
        let result = apply_guardrails(plan, &config).unwrap();
        assert_eq!(result.risk_tier, RiskTier::MutatingDangerous);
        assert!(result.execution_steps[0].requires_confirmation);
    }
}
