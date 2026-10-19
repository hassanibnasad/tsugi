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
    /// Rewrite the command to a safer equivalent.
    Rewrite(&'static str, &'static str),
    /// Block the command entirely.
    Block,
    /// Force user confirmation before execution.
    RequireConfirmation,
    /// Escalate the risk tier.
    EscalateRisk(RiskTier),
}

/// The complete set of safety policies.
static POLICIES: &[PolicyRule] = &[
    // ── Force Push ────────────────────────────────────────────────────
    PolicyRule {
        pattern: "git push --force ",
        action: PolicyAction::Rewrite("git push --force ", "git push --force-with-lease "),
        reason: "Replaced --force with --force-with-lease to prevent overwriting collaborators' work",
    },
    PolicyRule {
        pattern: "git push -f ",
        action: PolicyAction::Rewrite("git push -f ", "git push --force-with-lease "),
        reason: "Replaced -f with --force-with-lease to prevent overwriting collaborators' work",
    },
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

/// Apply all safety policies to a resolution plan.
///
/// This function:
/// 1. Scans every command against the policy rule set.
/// 2. Rewrites unsafe commands to safer equivalents where possible.
/// 3. Marks dangerous commands as requiring confirmation.
/// 4. Blocks commands that should never be auto-executed.
/// 5. Escalates the risk tier if any step triggers escalation.
///
/// Returns the modified plan, or an error if a command is blocked.
pub fn apply_guardrails(
    mut plan: ResolutionPlan,
    config: &TsugiConfig,
) -> Result<ResolutionPlan, TsugiError> {
    let mut highest_risk = plan.risk_tier;
    let mut modified_steps = Vec::with_capacity(plan.execution_steps.len());

    for mut step in plan.execution_steps {
        for rule in POLICIES {
            if step.cmd.contains(rule.pattern) {
                match &rule.action {
                    PolicyAction::Rewrite(from, to) => {
                        let original = step.cmd.clone();
                        step.cmd = step.cmd.replace(from, to);
                        info!(
                            "Guardrail: rewrote '{}' → '{}'. Reason: {}",
                            original, step.cmd, rule.reason
                        );
                        step.explanation = format!(
                            "{} [GUARDRAIL: {}]",
                            step.explanation, rule.reason
                        );
                    }

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
}
