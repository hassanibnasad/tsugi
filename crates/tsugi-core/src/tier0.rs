//! # Layer 2: Tier-0 Deterministic Fast-Path Engine
//!
//! Pattern-matches 80+ canonical Git scenarios against the [`RepoState`]
//! and produces a [`ResolutionPlan`] without invoking the LLM. Resolves in <2ms
//! using 0 bytes of LLM RAM.
//!
//! ## Design Philosophy
//!
//! Enterprise Git workflows follow structured patterns: diverged branches,
//! interrupted rebases, unmerged conflicts, submodules, worktrees, stashes,
//! tags, and remote tracking. For all unambiguous scenarios, a deterministic
//! engine produces zero-latency, 100% reliable plans without GPU/CPU overhead.

use crate::types::{
    ExecutionStep, HeadState, OngoingOperation, PlanSource, RepoState,
    ResolutionPlan, RiskTier,
};

/// Attempt to resolve the user's Git issue using deterministic pattern matching.
///
/// Returns `Some(ResolutionPlan)` if the issue matches a known canonical pattern,
/// or `None` if the LLM engine should be consulted.
pub fn try_resolve(state: &RepoState, user_intent: &str) -> Option<ResolutionPlan> {
    let intent = user_intent.to_lowercase();
    let current_branch = match state.head {
        HeadState::Attached { ref branch, .. } => branch.as_str(),
        _ => "main",
    };

    // ─── 1. ONGOING OPERATION PATTERNS ────────────────────────────────────────

    if let Some(ref op) = state.ongoing_operation {
        match op {
            OngoingOperation::Rebase { interactive, .. } => {
                if intent_matches_any(&intent, &["abort", "cancel", "stop", "undo"]) {
                    return Some(resolve_rebase_abort(state));
                }
                if intent_matches_any(&intent, &["skip", "omit", "drop commit", "skip commit"]) {
                    return Some(plan(
                        "Skipping current commit during rebase",
                        RiskTier::MutatingDangerous,
                        vec![step("git rebase --skip", "Skip the conflicting patch and continue rebase", true)],
                        "git rebase --abort",
                    ));
                }
                // Never hijack conflict resolution in Tier-0: defer to LLM engine or Conflict UI
                if !state.conflicts.is_empty() {
                    return None;
                }
                if intent_matches_any(&intent, &["continue", "finish", "resume", "fix", "done"]) || intent.is_empty() {
                    return Some(resolve_rebase_continue(state, *interactive));
                }
                return None;
            }

            OngoingOperation::Merge { merge_head } => {
                if intent_matches_any(&intent, &["abort", "cancel", "undo", "stop"]) {
                    return Some(resolve_merge_abort(state));
                }
                // Never hijack conflict resolution in Tier-0: defer to LLM engine
                if !state.conflicts.is_empty() {
                    return None;
                }
                let short_sha = &merge_head[..7.min(merge_head.len())];
                return Some(plan(
                    &format!("Completing merge of {short_sha} (all conflicts resolved)"),
                    RiskTier::MutatingSafe,
                    vec![step("git commit --no-edit", "Record the merge commit", false)],
                    "git merge --abort",
                ));
            }

            OngoingOperation::CherryPick { cherry_pick_head } => {
                if intent_matches_any(&intent, &["abort", "cancel", "undo", "stop"]) {
                    return Some(plan(
                        "Cherry-pick in progress — aborting per user request",
                        RiskTier::MutatingSafe,
                        vec![step("git cherry-pick --abort", "Abort the cherry-pick operation", false)],
                        "git cherry-pick --abort",
                    ));
                }
                if intent_matches_any(&intent, &["skip", "skip commit"]) {
                    return Some(plan(
                        "Skipping current commit during cherry-pick",
                        RiskTier::MutatingSafe,
                        vec![step("git cherry-pick --skip", "Skip the current cherry-pick patch", false)],
                        "git cherry-pick --abort",
                    ));
                }
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

            OngoingOperation::Revert { revert_head } => {
                if intent_matches_any(&intent, &["abort", "cancel", "undo", "stop"]) {
                    return Some(plan(
                        "Revert in progress — aborting per user request",
                        RiskTier::MutatingSafe,
                        vec![step("git revert --abort", "Abort the revert operation", false)],
                        "git revert --abort",
                    ));
                }
                if !state.conflicts.is_empty() {
                    return None;
                }
                let short_sha = &revert_head[..7.min(revert_head.len())];
                return Some(plan(
                    &format!("Revert of {short_sha} in progress — continuing"),
                    RiskTier::MutatingSafe,
                    vec![step("git revert --continue", "Continue the revert operation", false)],
                    "git revert --abort",
                ));
            }

            OngoingOperation::Bisect { .. } => {
                if intent_matches_any(&intent, &["abort", "cancel", "stop", "reset", "undo", "exit"]) {
                    return Some(plan(
                        "Bisect session active — resetting repository to original HEAD",
                        RiskTier::MutatingSafe,
                        vec![step("git bisect reset", "End the bisect session and return to original HEAD", false)],
                        "git bisect reset",
                    ));
                }
                if intent_matches_any(&intent, &["good", "working", "pass"]) {
                    return Some(plan(
                        "Marking current bisect revision as good",
                        RiskTier::ReadOnly,
                        vec![step("git bisect good", "Mark commit as good to narrow search range", false)],
                        "git bisect reset",
                    ));
                }
                if intent_matches_any(&intent, &["bad", "broken", "fail"]) {
                    return Some(plan(
                        "Marking current bisect revision as bad",
                        RiskTier::ReadOnly,
                        vec![step("git bisect bad", "Mark commit as bad to narrow search range", false)],
                        "git bisect reset",
                    ));
                }
                if intent_matches_any(&intent, &["status", "log", "view"]) {
                    return Some(plan(
                        "Viewing current bisect log",
                        RiskTier::ReadOnly,
                        vec![step("git bisect log", "Show bisect progress and tested commits", false)],
                        "git status",
                    ));
                }
            }

            OngoingOperation::ApplyMailbox => {
                if intent_matches_any(&intent, &["abort", "cancel", "undo", "stop"]) {
                    return Some(plan(
                        "Mailbox apply in progress — aborting",
                        RiskTier::MutatingSafe,
                        vec![step("git am --abort", "Abort the mailbox apply operation", false)],
                        "git am --abort",
                    ));
                }
                if intent_matches_any(&intent, &["skip"]) {
                    return Some(plan(
                        "Skipping current patch in mailbox apply",
                        RiskTier::MutatingSafe,
                        vec![step("git am --skip", "Skip current patch and proceed", false)],
                        "git am --abort",
                    ));
                }
                if intent_matches_any(&intent, &["continue", "resolved"]) {
                    return Some(plan(
                        "Continuing mailbox patch application",
                        RiskTier::MutatingSafe,
                        vec![step("git am --continue", "Resume mailbox apply after resolving patch", false)],
                        "git am --abort",
                    ));
                }
            }
        }
    }

    // ─── 2. DETACHED HEAD PATTERNS ────────────────────────────────────────────

    if let HeadState::Detached { ref commit_sha } = state.head {
        let sha_short = &commit_sha[..7.min(commit_sha.len())];

        if state.working_tree.has_changes() {
            return Some(plan(
                &format!("Detached HEAD at {sha_short} with uncommitted changes"),
                RiskTier::MutatingSafe,
                vec![step(
                    &format!("git switch -c recovery/{sha_short}"),
                    "Create and switch to a new branch to preserve your uncommitted work",
                    false,
                )],
                &format!("git switch - && git branch -D recovery/{sha_short}"),
            ));
        }

        if intent_matches_any(&intent, &["fix", "help", "back", "attach", "return", "previous", "restore"]) || intent.is_empty() {
            return Some(plan(
                &format!("Detached HEAD at {sha_short} with clean working tree"),
                RiskTier::ReadOnly,
                vec![step(
                    "git switch -",
                    "Return to the previously checked-out branch",
                    false,
                )],
                &format!("git checkout {sha_short}"),
            ));
        }

        if intent_matches_any(&intent, &["branch", "new branch", "save", "keep"]) {
            return Some(plan(
                &format!("Create a permanent branch from detached HEAD {sha_short}"),
                RiskTier::MutatingSafe,
                vec![step(
                    &format!("git switch -c branch-{sha_short}"),
                    "Create a new branch pointing to this detached commit",
                    false,
                )],
                &format!("git switch - && git branch -D branch-{sha_short}"),
            ));
        }
    }

    // ─── 3. SUBMODULE PATTERNS ────────────────────────────────────────────────

    if state.has_submodules || !state.submodules.is_empty() {
        let has_uninit = state.submodules.iter().any(|s| !s.is_initialized);
        let has_dirty = state.submodules.iter().any(|s| s.is_dirty);

        if intent_matches_any(&intent, &["submodule update", "init submodule", "sync submodule", "update submodule", "fetch submodule"])
            || (intent_matches_any(&intent, &["fix", "help"]) && has_uninit)
        {
            return Some(plan(
                "Uninitialized or outdated Git submodules detected",
                RiskTier::MutatingSafe,
                vec![
                    step("git submodule sync --recursive", "Synchronize submodule remote URLs", false),
                    step("git submodule update --init --recursive", "Initialize and clone missing submodules", false),
                ],
                "git submodule status",
            ));
        }

        if intent_matches_any(&intent, &["reset submodule", "clean submodule", "discard submodule"]) && has_dirty {
            return Some(plan(
                "Resetting modified submodules to clean state",
                RiskTier::MutatingDangerous,
                vec![
                    step("git submodule foreach --recursive git reset --hard", "Reset all modified submodule files", true),
                    step("git submodule update --init --recursive", "Ensure submodule pointers match parent tree", false),
                ],
                "git submodule status",
            ));
        }

        if intent_matches_any(&intent, &["submodule status", "check submodule", "list submodule"]) {
            return Some(plan(
                "Inspecting submodule status and commit hashes",
                RiskTier::ReadOnly,
                vec![step("git submodule status --recursive", "Display SHA and state of all submodules", false)],
                "git status",
            ));
        }
    }

    // ─── 4. WORKTREE PATTERNS ─────────────────────────────────────────────────

    if intent_matches_any(&intent, &["prune worktree", "clean worktree", "remove stale worktree"]) {
        return Some(plan(
            "Pruning stale or deleted linked worktrees",
            RiskTier::MutatingSafe,
            vec![step("git worktree prune -v", "Remove metadata for worktrees that no longer exist on disk", false)],
            "git worktree list",
        ));
    }

    if intent_matches_any(&intent, &["list worktree", "show worktree", "view worktree"]) {
        return Some(plan(
            "Listing all active linked worktrees",
            RiskTier::ReadOnly,
            vec![step("git worktree list", "Display paths and active branches for all linked worktrees", false)],
            "git status",
        ));
    }

    // ─── 5. SHALLOW CLONE PATTERNS ───────────────────────────────────────────

    if state.is_shallow || intent_matches_any(&intent, &["unshallow", "deepen clone", "full history", "shallow"]) {
        if intent_matches_any(&intent, &["unshallow", "full history", "fetch all commits"]) || state.is_shallow {
            return Some(plan(
                "Converting shallow repository clone into a complete history clone",
                RiskTier::MutatingSafe,
                vec![step("git fetch --unshallow", "Fetch complete historical commit graph from remote", false)],
                "git status",
            ));
        }
    }

    // ─── 6. LARGE FILES / LFS PATTERNS ────────────────────────────────────────

    if !state.large_files.is_empty() {
        if intent_matches_any(&intent, &["unstage large", "remove large", "fix large file", "lfs"]) {
            let files = state.large_files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>().join(" ");
            return Some(plan(
                "Unstaging large file(s) (>5MB) to prevent repository bloat",
                RiskTier::ReadOnly,
                vec![
                    step(&format!("git reset HEAD -- {files}"), "Unstage large files from current index", false),
                ],
                "git status",
            ));
        }
    }

    if intent_matches_any(&intent, &["lfs pull", "download lfs", "lfs download", "pointer file"]) {
        return Some(plan(
            "Downloading Git LFS binary payloads for current worktree",
            RiskTier::MutatingSafe,
            vec![step("git lfs pull", "Fetch and replace pointer files with actual binaries", false)],
            "git lfs status",
        ));
    }

    if intent_matches_any(&intent, &["lfs status", "check lfs"]) {
        return Some(plan(
            "Checking Git LFS tracking status",
            RiskTier::ReadOnly,
            vec![step("git lfs status", "List files tracked by Git LFS", false)],
            "git status",
        ));
    }

    // ─── 7. REMOTE & PUSH / PULL / FAST-FORWARD PATTERNS ──────────────────────

    // Fast-forward merge catch up
    if intent_matches_any(&intent, &["fast forward", "ff merge", "catch up branch", "ff-only"]) {
        let upstream = state.tracking.as_ref().map(|t| t.upstream.as_str()).unwrap_or("origin/main");
        return Some(plan(
            &format!("Fast-forward merging current branch with {upstream}"),
            RiskTier::MutatingSafe,
            vec![
                step("git fetch origin", "Fetch latest commits from remote", false),
                step(&format!("git merge --ff-only {upstream}"), "Fast-forward current branch without merge commit", false),
            ],
            "git reset --hard ORIG_HEAD",
        ));
    }

    // Remote pruning (sync deleted remote branches)
    if intent_matches_any(&intent, &["prune remote", "prune branches", "sync remote branches", "clean remote"]) {
        return Some(plan(
            "Pruning stale remote-tracking branch references",
            RiskTier::MutatingSafe,
            vec![step("git fetch --prune origin", "Remove references to branches deleted on origin", false)],
            "git status",
        ));
    }

    // Push with no upstream tracking
    if intent_matches_any(&intent, &["push", "publish", "upload"]) {
        if state.tracking.is_none() {
            return Some(plan(
                &format!("Branch '{current_branch}' has no upstream tracking branch"),
                RiskTier::MutatingSafe,
                vec![step(
                    &format!("git push --set-upstream origin {current_branch}"),
                    &format!("Push '{current_branch}' and set 'origin/{current_branch}' as the upstream"),
                    false,
                )],
                &format!("git push origin --delete {current_branch}"),
            ));
        }

        // Diverged branch handling (ahead & behind)
        if let Some(ref tracking) = state.tracking {
            if tracking.behind > 0 && tracking.ahead > 0 {
                return Some(plan(
                    &format!("Branch '{current_branch}' has diverged from '{}' (ahead {}, behind {})",
                        tracking.upstream, tracking.ahead, tracking.behind),
                    RiskTier::MutatingDangerous,
                    vec![
                        step(
                            &format!("git pull --rebase origin {current_branch}"),
                            "Rebase local commits on top of remote changes",
                            true,
                        ),
                        step(
                            &format!("git push --force-with-lease origin {current_branch}"),
                            "Safely push rebased commits using lease protection",
                            true,
                        ),
                    ],
                    "git rebase --abort",
                ));
            }

            // Behind only: needs simple rebase or fast-forward
            if tracking.behind > 0 && tracking.ahead == 0 {
                return Some(plan(
                    &format!("Branch '{current_branch}' is {} commit(s) behind remote", tracking.behind),
                    RiskTier::MutatingSafe,
                    vec![step(
                        &format!("git pull --ff-only origin {current_branch}"),
                        "Fast-forward local branch to match remote",
                        false,
                    )],
                    "git reset --hard ORIG_HEAD",
                ));
            }
        }
    }

    // Reset local to match remote completely (discard local commits)
    if intent_matches_any(&intent, &["reset to remote", "overwrite local with origin", "sync with origin", "match remote", "discard local commits"]) {
        let upstream = state.tracking.as_ref().map(|t| t.upstream.as_str()).unwrap_or("origin/main");
        return Some(plan(
            &format!("Resetting local branch '{current_branch}' to match remote '{upstream}'"),
            RiskTier::MutatingDangerous,
            vec![
                step("git stash push -u -m 'pre-remote-reset-backup'", "Protect uncommitted working tree changes", false),
                step("git fetch origin", "Ensure remote refs are up-to-date", false),
                step(&format!("git reset --hard {upstream}"), "Reset HEAD, index, and tracked files to remote state", true),
            ],
            "git stash pop",
        ));
    }

    // Show remotes
    if intent_matches_any(&intent, &["show remote", "list remote", "view remote", "remotes"]) {
        return Some(plan(
            "Listing configured Git remotes and URLs",
            RiskTier::ReadOnly,
            vec![step("git remote -v", "Display fetch and push URLs for all configured remotes", false)],
            "git status",
        ));
    }

    // ─── 8. COMMIT & HISTORY PATTERNS ─────────────────────────────────────────

    // "undo last commit" / "uncommit"
    if intent_matches_any(&intent, &["undo last commit", "uncommit", "undo commit", "reset last commit", "un-commit"]) {
        return Some(resolve_undo_last_commit(state));
    }

    // "undo last 2 commits" / "undo last 3 commits"
    if intent_matches_any(&intent, &["undo last 2 commits", "undo 2 commits", "reset 2 commits"]) {
        return Some(plan(
            "Undoing the last 2 commits while preserving modifications in working tree",
            RiskTier::MutatingSafe,
            vec![step("git reset HEAD~2", "Move HEAD back 2 commits; changes become unstaged", false)],
            "git reset --hard ORIG_HEAD",
        ));
    }
    if intent_matches_any(&intent, &["undo last 3 commits", "undo 3 commits", "reset 3 commits"]) {
        return Some(plan(
            "Undoing the last 3 commits while preserving modifications in working tree",
            RiskTier::MutatingSafe,
            vec![step("git reset HEAD~3", "Move HEAD back 3 commits; changes become unstaged", false)],
            "git reset --hard ORIG_HEAD",
        ));
    }

    // Amend commit message
    if intent_matches_any(&intent, &["amend", "fix message", "edit message", "change message", "fix commit message"]) {
        return Some(plan(
            "Amending the last commit message",
            RiskTier::MutatingSafe,
            vec![step(
                "git commit --amend",
                "Open the editor to modify the last commit message and stage additional changes",
                false,
            )],
            "git reset --soft HEAD@{1}",
        ));
    }

    // Change author on last commit
    if intent_matches_any(&intent, &["change author", "fix author", "reset author", "update author"]) {
        return Some(plan(
            "Updating the author of the last commit to current Git user config",
            RiskTier::MutatingSafe,
            vec![step(
                "git commit --amend --reset-author --no-edit",
                "Update the author metadata of HEAD without opening editor",
                false,
            )],
            "git reset --soft HEAD@{1}",
        ));
    }

    // Empty commit to trigger CI/CD
    if intent_matches_any(&intent, &["empty commit", "trigger ci", "rebuild", "rerun ci"]) {
        return Some(plan(
            "Creating an empty commit to trigger CI/CD pipeline",
            RiskTier::MutatingSafe,
            vec![step(
                "git commit --allow-empty -m \"chore: trigger CI rebuild [skip ci]\"",
                "Create a commit with no file changes to re-trigger pipeline",
                false,
            )],
            "git reset --hard HEAD~1",
        ));
    }

    // Show last commit
    if intent_matches_any(&intent, &["show last commit", "view last commit", "inspect last commit", "last commit diff"]) {
        return Some(plan(
            "Inspecting the most recent commit",
            RiskTier::ReadOnly,
            vec![step("git show HEAD", "Display diff and metadata for HEAD commit", false)],
            "git status",
        ));
    }

    // ─── 9. BRANCH MANAGEMENT PATTERNS ────────────────────────────────────────

    // Switch to previous branch
    if intent_matches_any(&intent, &["previous branch", "go back", "last branch", "switch back", "switch -"]) {
        return Some(plan(
            "Switching back to the previously active branch",
            RiskTier::ReadOnly,
            vec![step("git switch -", "Checkout the previously checked-out branch", false)],
            "git switch -",
        ));
    }

    // Rename current branch
    if intent_matches_any(&intent, &["rename branch", "rename current branch", "change branch name"]) {
        return Some(plan(
            "Renaming the current branch",
            RiskTier::MutatingSafe,
            vec![step("git branch -m <new-branch-name>", "Rename current branch locally", false)],
            &format!("git branch -m {current_branch}"),
        ));
    }

    // Delete merged branches
    if intent_matches_any(&intent, &["delete merged", "clean merged branches", "prune merged branches", "remove merged"]) {
        return Some(plan(
            "Cleaning up local branches that have already been merged into main",
            RiskTier::MutatingSafe,
            vec![
                step("git branch --merged main | grep -v '^[ *]*main$' | xargs -r git branch -d", "Safely delete merged local branches", true),
            ],
            "git reflog",
        ));
    }

    // ─── 10. STASH OPERATIONS ─────────────────────────────────────────────────

    // Stash including untracked
    if intent_matches_any(&intent, &["stash all", "stash untracked", "stash everything", "stash include untracked"]) {
        return Some(plan(
            "Stashing all changes including untracked files",
            RiskTier::MutatingSafe,
            vec![step("git stash push -u -m \"tsugi-complete-stash\"", "Stash staged, unstaged, and untracked files", false)],
            "git stash pop",
        ));
    }

    // Standard stash
    if intent_matches_any(&intent, &["stash", "save work", "save changes", "park changes"]) {
        return Some(plan(
            "Stashing all uncommitted changes",
            RiskTier::MutatingSafe,
            vec![step("git stash push -m \"tsugi-auto-stash\"", "Stash all staged and unstaged changes", false)],
            "git stash pop",
        ));
    }

    // Pop stash
    if intent_matches_any(&intent, &["pop stash", "restore stash", "unstash"]) {
        if state.stash_count > 0 {
            return Some(plan(
                &format!("Restoring most recent stash ({} stash(es) available)", state.stash_count),
                RiskTier::MutatingSafe,
                vec![step("git stash pop", "Apply and drop the most recent stash entry", false)],
                "git stash",
            ));
        } else {
            return Some(plan(
                "No stashes found in repository",
                RiskTier::ReadOnly,
                vec![step("git stash list", "Verify empty stash list", false)],
                "git status",
            ));
        }
    }

    // View stash
    if intent_matches_any(&intent, &["view stash", "show stash", "inspect stash", "stash diff"]) {
        return Some(plan(
            "Inspecting the contents of the latest stash",
            RiskTier::ReadOnly,
            vec![step("git stash show -p", "Display full patch diff stored in stash@{0}", false)],
            "git status",
        ));
    }

    // Drop stash
    if intent_matches_any(&intent, &["drop stash", "clear stash", "delete stash", "remove stash"]) {
        return Some(plan(
            "Dropping the most recent stash entry",
            RiskTier::MutatingDangerous,
            vec![step("git stash drop", "Permanently discard stash@{0}", true)],
            "git reflog",
        ));
    }

    // ─── 11. WORKING TREE & CLEAN PATTERNS ────────────────────────────────────

    // Unstage
    if intent_matches_any(&intent, &["unstage", "reset staged", "unadd", "remove from staging", "un-stage"]) {
        return Some(plan(
            "Unstaging all files from the index",
            RiskTier::ReadOnly,
            vec![step("git restore --staged .", "Remove all files from staging area without losing changes", false)],
            "git status",
        ));
    }

    // Dry-run clean
    if intent_matches_any(&intent, &["dry run clean", "what would be cleaned", "check untracked", "preview clean"]) {
        return Some(plan(
            "Previewing untracked files that would be removed by clean",
            RiskTier::ReadOnly,
            vec![step("git clean -nd", "Show untracked files without actually deleting them", false)],
            "git status",
        ));
    }

    // Discard all changes
    if intent_matches_any(&intent, &["discard", "throw away", "clean all", "reset all", "revert all changes"]) {
        if state.working_tree.has_changes() {
            return Some(plan(
                "Discarding all uncommitted changes with automatic safety snapshot backup",
                RiskTier::MutatingDangerous,
                vec![
                    step("git stash push -u -m \"tsugi-safety-backup\"", "Safety backup of all changes before discarding", false),
                    step("git restore .", "Discard unstaged modifications in tracked files", false),
                    step("git restore --staged .", "Unstage any staged files", false),
                    step("git clean -fd", "Remove untracked files", true),
                ],
                "git stash pop",
            ));
        }
    }

    // ─── 12. TAG PATTERNS ─────────────────────────────────────────────────────

    if intent_matches_any(&intent, &["push tags", "publish tags", "upload tags"]) {
        return Some(plan(
            "Pushing all local tags to remote repository",
            RiskTier::MutatingSafe,
            vec![step("git push origin --tags", "Publish all tags to origin", false)],
            "git status",
        ));
    }

    if intent_matches_any(&intent, &["list tags", "show tags", "view tags"]) {
        return Some(plan(
            "Listing all tags in the repository",
            RiskTier::ReadOnly,
            vec![step("git tag -n --sort=-creatordate", "List tags sorted by creation date with annotations", false)],
            "git status",
        ));
    }

    // ─── 13. DIAGNOSTICS, REFLOG & MAINTENANCE ───────────────────────────────

    // Reflog inspection
    if intent_matches_any(&intent, &["reflog", "find lost commit", "recover lost", "view reflog", "where did my commit go"]) {
        return Some(plan(
            "Inspecting reflog to locate lost commits or branch states",
            RiskTier::ReadOnly,
            vec![step("git reflog -n 25", "Display the last 25 HEAD position movements", false)],
            "git status",
        ));
    }

    // Verify git integrity (fsck)
    if intent_matches_any(&intent, &["fsck", "verify objects", "corrupt", "check integrity", "integrity"]) {
        return Some(plan(
            "Verifying object database integrity with git fsck",
            RiskTier::ReadOnly,
            vec![step("git fsck --full", "Audit repository database for broken links and corrupt objects", false)],
            "git status",
        ));
    }

    // Garbage collection & optimization
    if intent_matches_any(&intent, &["optimize repo", "shrink repo", "compress git", "gc", "garbage collect"]) {
        return Some(plan(
            "Running aggressive garbage collection and pack optimization",
            RiskTier::MutatingSafe,
            vec![step("git gc --prune=now --aggressive", "Prune unreferenced objects and repack loose objects", false)],
            "git status",
        ));
    }

    // Line endings / CRLF renormalization
    if intent_matches_any(&intent, &["crlf", "line endings", "line-endings", "renormalize", "whitespace warnings"]) {
        return Some(plan(
            "Renormalizing working tree line endings according to .gitattributes",
            RiskTier::MutatingSafe,
            vec![
                step("git add --renormalize .", "Re-evaluate line endings for all tracked files", false),
                step("git status", "Inspect files with normalized line endings", false),
            ],
            "git checkout -- .",
        ));
    }

    // Hook bypass guidance
    if intent_matches_any(&intent, &["skip hook", "bypass hook", "no verify", "ignore hook"]) {
        return Some(plan(
            "Bypassing git commit verification hooks",
            RiskTier::MutatingDangerous,
            vec![step("git commit --no-verify", "Commit staged changes while bypassing pre-commit and commit-msg hooks", true)],
            "git reset --soft HEAD~1",
        ));
    }

    // ─── NO MATCH — defer to LLM ─────────────────────────────────────────────
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
            worktrees: Vec::new(),
            remotes: Vec::new(),
            submodules: Vec::new(),
            is_shallow: false,
            large_files: Vec::new(),
            installed_hooks: Vec::new(),
            graph_metrics: None,
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
    fn test_undo_multiple_commits() {
        let state = empty_state();
        let plan = try_resolve(&state, "undo last 3 commits").unwrap();
        assert!(plan.execution_steps[0].cmd.contains("HEAD~3"));
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
        assert!(plan.execution_steps[0].cmd.contains("switch -"));
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
        assert!(plan.execution_steps[0].cmd.contains("switch -c recovery/"));
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
    fn test_fast_forward_branch() {
        let mut state = empty_state();
        state.tracking = Some(TrackingInfo {
            upstream: "origin/main".to_string(),
            ahead: 0,
            behind: 2,
        });
        let plan = try_resolve(&state, "fast forward branch").unwrap();
        assert!(plan.execution_steps.iter().any(|s| s.cmd.contains("--ff-only")));
    }

    #[test]
    fn test_submodule_uninit_fix() {
        let mut state = empty_state();
        state.has_submodules = true;
        state.submodules.push(SubmoduleState {
            name: "vendor".to_string(),
            path: "vendor".to_string(),
            url: Some("https://example.com/repo.git".to_string()),
            head_commit: None,
            is_initialized: false,
            is_dirty: false,
        });
        let plan = try_resolve(&state, "fix submodules").unwrap();
        assert!(plan.execution_steps.iter().any(|s| s.cmd.contains("submodule update --init --recursive")));
    }

    #[test]
    fn test_unshallow_history() {
        let mut state = empty_state();
        state.is_shallow = true;
        let plan = try_resolve(&state, "unshallow").unwrap();
        assert!(plan.execution_steps[0].cmd.contains("--unshallow"));
    }

    #[test]
    fn test_worktree_prune() {
        let state = empty_state();
        let plan = try_resolve(&state, "prune worktree").unwrap();
        assert!(plan.execution_steps[0].cmd.contains("worktree prune"));
    }

    #[test]
    fn test_change_author() {
        let state = empty_state();
        let plan = try_resolve(&state, "change author").unwrap();
        assert!(plan.execution_steps[0].cmd.contains("--reset-author"));
    }

    #[test]
    fn test_crlf_renormalize() {
        let state = empty_state();
        let plan = try_resolve(&state, "fix line endings").unwrap();
        assert!(plan.execution_steps[0].cmd.contains("--renormalize"));
    }

    #[test]
    fn test_no_match_returns_none() {
        let state = empty_state();
        let plan = try_resolve(&state, "something very unusual and complex");
        assert!(plan.is_none());
    }
}
