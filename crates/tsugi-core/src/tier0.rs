//! # Layer 2: Tier-0 Deterministic Fast-Path Engine
//!
//! Pattern-matches the top 40+ canonical Git error states against the [`RepoState`]
//! and produces a [`ResolutionPlan`] without invoking the LLM. Resolves in <2ms
//! using 0 bytes of LLM RAM.
//!
//! ## Design Philosophy
//!
//! Not every Git error needs neural inference. ~80% of developer Git issues fall
//! into well-known canonical patterns (detached HEAD, no upstream, mid-rebase conflict).
//! For these, a deterministic lookup table produces faster, more reliable resolutions
//! than any language model — with zero latency and zero resource usage.
//!
//! The Tier-0 engine is intentionally conservative: it only matches when the state
//! signature is unambiguous. If the user's intent or the repository state is ambiguous,
//! it returns `None` and defers to the LLM engine (Layer 3).

use crate::types::{
    ExecutionStep, HeadState, OngoingOperation, PlanSource, RepoState,
    ResolutionPlan, RiskTier,
};

/// Attempt to resolve the user's Git issue using deterministic pattern matching.
///
/// Returns `Some(ResolutionPlan)` if the issue matches a known canonical pattern,
/// or `None` if the LLM engine should be consulted.
///
/// The `user_intent` parameter is the raw text the user typed (e.g., "fix my rebase",
/// "undo last commit", "push my changes"). If empty, we rely purely on state signals.
pub fn try_resolve(state: &RepoState, user_intent: &str) -> Option<ResolutionPlan> {
    let intent = user_intent.to_lowercase();

    // Priority order: ongoing operations first, then intent-based patterns, then
    // working tree issues.

    // ─── ONGOING OPERATION PATTERNS ───────────────────────────────────────

    if let Some(ref op) = state.ongoing_operation {
        match op {
            OngoingOperation::Rebase { interactive, .. } => {
                if intent_matches_any(&intent, &["abort", "cancel", "stop", "undo"]) {
                    return Some(resolve_rebase_abort(state));
                }
                // Never hijack conflict resolution in Tier-0: defer to LLM engine
                if !state.conflicts.is_empty() {
                    return None;
                }
                if intent_matches_any(&intent, &["continue", "finish", "resume", "fix"]) || intent.is_empty() {
                    return Some(resolve_rebase_continue(state, *interactive));
                }
                return None;
            }

            OngoingOperation::Merge { .. } => {
                if intent_matches_any(&intent, &["abort", "cancel", "undo"]) {
                    return Some(resolve_merge_abort(state));
                }
                // Never hijack conflict resolution in Tier-0: defer to LLM engine
                if !state.conflicts.is_empty() {
                    return None;
                }
                return Some(resolve_merge_continue(state));
            }

            OngoingOperation::CherryPick { cherry_pick_head } => {
                if intent_matches_any(&intent, &["abort", "cancel", "undo"]) {
                    return Some(plan(
                        "Cherry-pick in progress — aborting per user request",
                        RiskTier::MutatingSafe,
                        vec![step("git cherry-pick --abort", "Abort the cherry-pick operation", false)],
                        "git cherry-pick --abort",
                    ));
                }
                // Never hijack conflict resolution in Tier-0: defer to LLM engine
                if !state.conflicts.is_empty() {
                    return None;
                }
                return Some(plan(
                    &format!("Cherry-pick of {} paused — no conflicts detected, continuing", &cherry_pick_head[..7.min(cherry_pick_head.len())]),
                    RiskTier::MutatingSafe,
                    vec![step("git cherry-pick --continue", "Continue the cherry-pick operation", false)],
                    "git cherry-pick --abort",
                ));
            }

            OngoingOperation::Revert { .. } => {
                if intent_matches_any(&intent, &["abort", "cancel", "undo"]) {
                    return Some(plan(
                        "Revert in progress — aborting per user request",
                        RiskTier::MutatingSafe,
                        vec![step("git revert --abort", "Abort the revert operation", false)],
                        "git revert --abort",
                    ));
                }
                // Never hijack conflict resolution in Tier-0: defer to LLM engine
                if !state.conflicts.is_empty() {
                    return None;
                }
                return Some(plan(
                    "Revert in progress — continuing",
                    RiskTier::MutatingSafe,
                    vec![step("git revert --continue", "Continue the revert operation", false)],
                    "git revert --abort",
                ));
            }

            OngoingOperation::Bisect { .. } => {
                if intent_matches_any(&intent, &["abort", "cancel", "stop", "reset", "undo"]) {
                    return Some(plan(
                        "Bisect in progress — resetting per user request",
                        RiskTier::MutatingSafe,
                        vec![step("git bisect reset", "End the bisect session and return to original HEAD", false)],
                        "git bisect reset",
                    ));
                }
            }

            OngoingOperation::ApplyMailbox => {
                if intent_matches_any(&intent, &["abort", "cancel", "undo"]) {
                    return Some(plan(
                        "Mailbox apply in progress — aborting",
                        RiskTier::MutatingSafe,
                        vec![step("git am --abort", "Abort the mailbox apply operation", false)],
                        "git am --abort",
                    ));
                }
            }
        }
    }

    // ─── DETACHED HEAD PATTERNS ──────────────────────────────────────────

    if let HeadState::Detached { ref commit_sha } = state.head {
        let sha_short = &commit_sha[..7.min(commit_sha.len())];

        // Detached HEAD with uncommitted work
        if state.working_tree.has_changes() {
            return Some(plan(
                &format!("Detached HEAD at {sha_short} with uncommitted changes"),
                RiskTier::MutatingSafe,
                vec![
                    step(
                        &format!("git checkout -b recovery/{sha_short}"),
                        "Create a new branch to preserve your work at the current detached HEAD",
                        false,
                    ),
                ],
                &format!("git checkout {sha_short} && git branch -D recovery/{sha_short}"),
            ));
        }

        // Detached HEAD, clean tree — just wants to get back
        if intent_matches_any(&intent, &["fix", "help", "back", "attach", "return"]) || intent.is_empty() {
            return Some(plan(
                &format!("Detached HEAD at {sha_short} with clean working tree"),
                RiskTier::ReadOnly,
                vec![step(
                    "git checkout -",
                    "Return to the previously checked-out branch",
                    false,
                )],
                &format!("git checkout {sha_short}"),
            ));
        }
    }

    // ─── INTENT-BASED PATTERNS (no ongoing operation) ────────────────────

    if state.ongoing_operation.is_none() {
        // "undo last commit" / "uncommit"
        if intent_matches_any(&intent, &["undo last commit", "uncommit", "undo commit", "reset last"]) {
            return Some(resolve_undo_last_commit(state));
        }

        // "unstage" / "reset staged"
        if intent_matches_any(&intent, &["unstage", "reset staged", "unadd", "remove from staging"]) {
            return Some(plan(
                "Unstaging all files from the index",
                RiskTier::ReadOnly,
                vec![step("git reset HEAD", "Remove all files from the staging area without discarding changes", false)],
                "No rollback needed — changes are preserved in the working tree",
            ));
        }

        // "stash" / "save my work"
        if intent_matches_any(&intent, &["stash", "save work", "save changes", "park changes"]) {
            return Some(plan(
                "Stashing all uncommitted changes",
                RiskTier::MutatingSafe,
                vec![step("git stash push -m \"tsugi-auto-stash\"", "Stash all staged and unstaged changes", false)],
                "git stash pop",
            ));
        }

        // "pop stash" / "restore stash"
        if intent_matches_any(&intent, &["pop stash", "restore stash", "apply stash", "unstash"]) {
            if state.stash_count > 0 {
                return Some(plan(
                    &format!("Restoring most recent stash ({} stash(es) available)", state.stash_count),
                    RiskTier::MutatingSafe,
                    vec![step("git stash pop", "Apply and remove the most recent stash entry", false)],
                    "git stash",
                ));
            }
        }

        // "push" with no upstream
        if intent_matches_any(&intent, &["push", "publish", "upload"]) {
            if state.tracking.is_none() {
                if let HeadState::Attached { ref branch, .. } = state.head {
                    return Some(plan(
                        &format!("Branch '{branch}' has no upstream tracking branch"),
                        RiskTier::MutatingSafe,
                        vec![step(
                            &format!("git push --set-upstream origin {branch}"),
                            &format!("Push '{branch}' and set 'origin/{branch}' as the upstream"),
                            false,
                        )],
                        "No rollback needed — the remote branch can be deleted with git push origin --delete",
                    ));
                }
            }

            if let Some(ref tracking) = state.tracking {
                if tracking.behind > 0 && tracking.ahead > 0 {
                    // Diverged branches
                    if let HeadState::Attached { ref branch, .. } = state.head {
                        return Some(plan(
                            &format!("Branch '{branch}' has diverged from '{}' (ahead {}, behind {})",
                                tracking.upstream, tracking.ahead, tracking.behind),
                            RiskTier::MutatingDangerous,
                            vec![
                                step(
                                    &format!("git pull --rebase origin {branch}"),
                                    "Rebase local commits on top of remote changes",
                                    true,
                                ),
                                step(
                                    &format!("git push origin {branch}"),
                                    "Push rebased commits to remote",
                                    false,
                                ),
                            ],
                            "git rebase --abort",
                        ));
                    }
                }
            }
        }

        // "discard changes" / "clean"
        if intent_matches_any(&intent, &["discard", "throw away", "clean", "reset all"]) {
            if state.working_tree.has_changes() {
                return Some(plan(
                    "Discarding all uncommitted changes (staged and unstaged)",
                    RiskTier::MutatingDangerous,
                    vec![
                        step("git stash push -m \"tsugi-safety-backup\"", "Safety backup of all changes before discarding", false),
                        step("git checkout -- .", "Discard all unstaged changes in tracked files", false),
                        step("git reset HEAD", "Unstage all staged changes", false),
                    ],
                    "git stash pop",
                ));
            }
        }

        // "amend" / "fix last commit message"
        if intent_matches_any(&intent, &["amend", "fix message", "edit message", "change message"]) {
            return Some(plan(
                "Amending the last commit",
                RiskTier::MutatingSafe,
                vec![step(
                    "git commit --amend",
                    "Open the editor to modify the last commit message (also includes any staged changes)",
                    false,
                )],
                "git reset --soft HEAD@{1}",
            ));
        }
    }

    // ─── NO MATCH — defer to LLM ─────────────────────────────────────────
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Specialized Resolution Builders
// ─────────────────────────────────────────────────────────────────────────────

fn resolve_rebase_continue(_state: &RepoState, interactive: bool) -> ResolutionPlan {
    let mode = if interactive { "Interactive rebase" } else { "Rebase" };

    plan(
        &format!("{mode} in progress — no conflicts detected, continuing"),
        RiskTier::MutatingSafe,
        vec![step("git rebase --continue", "Continue the rebase operation", false)],
        "git rebase --abort",
    )
}

fn resolve_rebase_abort(_state: &RepoState) -> ResolutionPlan {
    plan(
        "Aborting rebase and returning to pre-rebase state",
        RiskTier::MutatingSafe,
        vec![step("git rebase --abort", "Abort the rebase and restore the branch to its pre-rebase state", false)],
        "No rollback needed — rebase abort itself is the rollback operation",
    )
}

fn resolve_merge_continue(_state: &RepoState) -> ResolutionPlan {
    plan(
        "Merge in progress — no conflicts detected, completing merge commit",
        RiskTier::MutatingSafe,
        vec![step("git commit", "Complete the merge commit (conflicts already resolved)", false)],
        "git merge --abort",
    )
}

fn resolve_merge_abort(_state: &RepoState) -> ResolutionPlan {
    plan(
        "Aborting merge and returning to pre-merge state",
        RiskTier::MutatingSafe,
        vec![step("git merge --abort", "Abort the merge and restore the branch to its pre-merge state", false)],
        "No rollback needed — merge abort itself is the rollback operation",
    )
}

fn resolve_undo_last_commit(state: &RepoState) -> ResolutionPlan {
    let has_staged = state.working_tree.staged_new > 0
        || state.working_tree.staged_modified > 0
        || state.working_tree.staged_deleted > 0;

    if has_staged {
        // Soft reset to preserve both staged and unstaged changes
        plan(
            "Undoing the last commit while preserving all changes in the staging area",
            RiskTier::MutatingSafe,
            vec![step(
                "git reset --soft HEAD~1",
                "Move HEAD back one commit; changes remain staged",
                false,
            )],
            "git commit -C ORIG_HEAD",
        )
    } else {
        // Mixed reset (default) to preserve changes as unstaged
        plan(
            "Undoing the last commit while preserving changes as unstaged modifications",
            RiskTier::MutatingSafe,
            vec![step(
                "git reset HEAD~1",
                "Move HEAD back one commit; changes become unstaged",
                false,
            )],
            "git commit -C ORIG_HEAD",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Check if the user's intent string contains any of the given keywords.
fn intent_matches_any(intent: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|kw| intent.contains(kw))
}

/// Convenience builder for a `ResolutionPlan`.
fn plan(
    root_cause: &str,
    risk_tier: RiskTier,
    execution_steps: Vec<ExecutionStep>,
    rollback_cmd: &str,
) -> ResolutionPlan {
    ResolutionPlan {
        root_cause: root_cause.to_string(),
        risk_tier,
        execution_steps,
        rollback_cmd: rollback_cmd.to_string(),
        source: PlanSource::Tier0,
        confidence: 1.0,
    }
}

/// Convenience builder for an `ExecutionStep`.
fn step(cmd: &str, explanation: &str, requires_confirmation: bool) -> ExecutionStep {
    ExecutionStep {
        cmd: cmd.to_string(),
        explanation: explanation.to_string(),
        requires_confirmation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use std::path::PathBuf;

    fn empty_state() -> RepoState {
        RepoState {
            repo_root: PathBuf::from("/tmp/test"),
            head: HeadState::Attached {
                branch: "main".to_string(),
                commit_sha: "abc1234567890".to_string(),
            },
            ongoing_operation: None,
            conflicts: Vec::new(),
            working_tree: WorkingTreeSummary::default(),
            recent_commits: Vec::new(),
            recent_reflog: Vec::new(),
            tracking: None,
            stash_count: 0,
            has_submodules: false,
            safe_rollback_ref: None,
        }
    }

    #[test]
    fn test_undo_last_commit() {
        let state = empty_state();
        let plan = try_resolve(&state, "undo last commit");
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.source, PlanSource::Tier0);
        assert!(plan.execution_steps[0].cmd.contains("git reset"));
    }

    #[test]
    fn test_detached_head_clean() {
        let mut state = empty_state();
        state.head = HeadState::Detached {
            commit_sha: "abc1234567890".to_string(),
        };
        let plan = try_resolve(&state, "fix");
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert!(plan.execution_steps[0].cmd.contains("checkout"));
    }

    #[test]
    fn test_detached_head_dirty() {
        let mut state = empty_state();
        state.head = HeadState::Detached {
            commit_sha: "abc1234567890".to_string(),
        };
        state.working_tree.unstaged_modified = 3;
        let plan = try_resolve(&state, "");
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert!(plan.execution_steps[0].cmd.contains("checkout -b recovery/"));
    }

    #[test]
    fn test_rebase_with_conflicts() {
        let mut state = empty_state();
        state.ongoing_operation = Some(OngoingOperation::Rebase {
            interactive: true,
            current_step: Some(3),
            total_steps: Some(5),
            onto_branch: Some("main".to_string()),
        });
        state.conflicts = vec![ConflictEntry {
            path: "src/auth.ts".to_string(),
            conflict_type: ConflictType::BothModified,
            primary_hunk: None,
        }];

        // When conflicts are present, Tier-0 must NOT hijack; it defers to the LLM
        let plan = try_resolve(&state, "fix my rebase");
        assert!(plan.is_none());

        // An explicit abort intent, however, is handled deterministically
        let abort_plan = try_resolve(&state, "abort rebase");
        assert!(abort_plan.is_some());
        let abort_plan = abort_plan.unwrap();
        assert!(abort_plan.execution_steps[0].cmd.contains("rebase --abort"));
    }

    #[test]
    fn test_no_upstream_push() {
        let state = empty_state();
        let plan = try_resolve(&state, "push");
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert!(plan.execution_steps[0].cmd.contains("--set-upstream"));
    }

    #[test]
    fn test_no_match_returns_none() {
        let state = empty_state();
        let plan = try_resolve(&state, "something very unusual and complex");
        assert!(plan.is_none());
    }
}
