//! # Layer 1: Deterministic Context Collector (Probe)
//!
//! Inspects the `.git/` directory via libgit2 to produce a [`RepoState`] fact sheet.
//! This runs in <10ms and uses <5MB of RAM. The LLM never sees raw terminal output —
//! it only receives the structured fact sheet from this layer.
//!
//! ## Design Rationale
//!
//! Small language models (1.5B parameters) fail when fed 3,000 tokens of noisy
//! `git status` / `git diff` / `git log` output. By pre-computing the commit graph
//! topology, conflict status, and ongoing operation state deterministically, we shrink
//! the context to ~150 tokens and let the model focus purely on intent-to-command mapping.

use std::path::Path;

use git2::{
    Repository, RepositoryState, StatusOptions, StatusShow,
};
use tracing::{debug, warn};

use crate::types::{
    ChangedFile, CommitEntry, CommitGraphMetrics, ConflictEntry, ConflictHunk, ConflictType,
    FileDeltaStatus, HeadState, HookInfo, LargeFileEntry, OngoingOperation, ReflogEntry, RemoteInfo,
    RepoState, SubmoduleState, TrackingInfo, TsugiError, WorkingTreeSummary, WorktreeInfo,
};

/// Maximum number of recent commits to include in the fact sheet.
const MAX_RECENT_COMMITS: usize = 10;

/// Maximum number of reflog entries to include.
const MAX_REFLOG_ENTRIES: usize = 10;

/// Probe the Git repository at the given path and produce a complete [`RepoState`].
///
/// This function is the sole entry point for Layer 1. It:
/// 1. Discovers and opens the repository (walking up from `path` if needed).
/// 2. Determines HEAD state (attached/detached/unborn).
/// 3. Detects any ongoing multi-step operation (rebase, merge, cherry-pick, etc.).
/// 4. Enumerates conflict entries.
/// 5. Computes a working tree summary (staged/unstaged/untracked counts).
/// 6. Collects recent commits and reflog entries.
/// 7. Determines branch tracking info and stash count.
///
/// All operations are **read-only** — this function never modifies the repository.
pub fn probe_repo_state(path: &Path) -> Result<RepoState, TsugiError> {
    let repo = Repository::discover(path).map_err(|_| TsugiError::RepoNotFound {
        path: path.to_path_buf(),
    })?;

    let repo_root = repo
        .workdir()
        .unwrap_or_else(|| repo.path())
        .to_path_buf();

    debug!("Probing repository at: {}", repo_root.display());

    let head = probe_head(&repo)?;
    let ongoing_operation = probe_ongoing_operation(&repo);
    let conflicts = probe_conflicts(&repo);
    let working_tree = probe_working_tree(&repo);
    let recent_commits = probe_recent_commits(&repo);
    let recent_reflog = probe_reflog(&repo);
    let tracking = probe_tracking(&repo);
    let incoming_commits = probe_incoming_commits(&repo, &tracking, &ongoing_operation);
    let stash_count = probe_stash_count(&repo);
    let has_submodules = probe_submodules(&repo);
    let safe_rollback_ref = compute_safe_rollback(&repo, &ongoing_operation);
    let worktrees = probe_worktrees(&repo);
    let remotes = probe_remotes(&repo);
    let submodules = probe_submodule_details(&repo);
    let is_shallow = probe_is_shallow(&repo);
    let large_files = probe_large_files_in_index(&repo, 5_000_000);
    let installed_hooks = probe_installed_hooks(&repo);
    let graph_metrics = probe_graph_metrics(&repo, &recent_commits);

    Ok(RepoState {
        repo_root,
        head,
        ongoing_operation,
        conflicts,
        working_tree,
        recent_commits,
        recent_reflog,
        tracking,
        stash_count,
        has_submodules,
        safe_rollback_ref,
        worktrees,
        remotes,
        submodules,
        is_shallow,
        large_files,
        installed_hooks,
        graph_metrics,
        incoming_commits,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// HEAD State
// ─────────────────────────────────────────────────────────────────────────────

fn probe_head(repo: &Repository) -> Result<HeadState, TsugiError> {
    if repo.head_detached().unwrap_or(false) {
        let head = repo.head()?;
        let oid = head.target().map(|o| o.to_string()).unwrap_or_default();
        return Ok(HeadState::Detached { commit_sha: oid });
    }

    match repo.head() {
        Ok(head) => {
            let branch = head
                .shorthand()
                .unwrap_or("unknown")
                .to_string();
            let commit_sha = head.target().map(|o| o.to_string()).unwrap_or_default();
            Ok(HeadState::Attached { branch, commit_sha })
        }
        Err(e) if e.code() == git2::ErrorCode::UnbornBranch => {
            // HEAD points to a branch that doesn't exist yet (empty repo).
            let head_ref = repo.find_reference("HEAD").ok();
            let branch = head_ref
                .as_ref()
                .and_then(|r| r.symbolic_target().ok().flatten())
                .unwrap_or("main")
                .replace("refs/heads/", "");
            Ok(HeadState::Unborn { branch })
        }
        Err(e) => Err(TsugiError::Git(e)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ongoing Operation Detection
// ─────────────────────────────────────────────────────────────────────────────

fn probe_ongoing_operation(repo: &Repository) -> Option<OngoingOperation> {
    let state = repo.state();

    match state {
        RepositoryState::Merge => {
            let merge_head = read_git_file(repo, "MERGE_HEAD");
            Some(OngoingOperation::Merge {
                merge_head: merge_head.unwrap_or_else(|| "unknown".to_string()),
            })
        }

        RepositoryState::Rebase
        | RepositoryState::RebaseInteractive
        | RepositoryState::RebaseMerge => {
            let interactive = state == RepositoryState::RebaseInteractive
                || state == RepositoryState::RebaseMerge;

            let (current_step, total_steps) = read_rebase_progress(repo);
            let onto_branch = read_git_file(repo, "rebase-merge/onto")
                .or_else(|| read_git_file(repo, "rebase-apply/onto"));

            Some(OngoingOperation::Rebase {
                interactive,
                current_step,
                total_steps,
                onto_branch,
            })
        }

        RepositoryState::CherryPick | RepositoryState::CherryPickSequence => {
            let cherry_pick_head = read_git_file(repo, "CHERRY_PICK_HEAD");
            Some(OngoingOperation::CherryPick {
                cherry_pick_head: cherry_pick_head.unwrap_or_else(|| "unknown".to_string()),
            })
        }

        RepositoryState::Revert | RepositoryState::RevertSequence => {
            let revert_head = read_git_file(repo, "REVERT_HEAD");
            Some(OngoingOperation::Revert {
                revert_head: revert_head.unwrap_or_else(|| "unknown".to_string()),
            })
        }

        RepositoryState::Bisect => {
            let good = read_git_file(repo, "BISECT_EXPECTED_REV");
            let bad = read_git_file(repo, "BISECT_START");
            Some(OngoingOperation::Bisect { good, bad })
        }

        RepositoryState::ApplyMailbox | RepositoryState::ApplyMailboxOrRebase => {
            Some(OngoingOperation::ApplyMailbox)
        }

        RepositoryState::Clean => None,
    }
}

/// Read a file from the `.git/` directory and return its trimmed content.
fn read_git_file(repo: &Repository, relative: &str) -> Option<String> {
    let path = repo.path().join(relative);
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Read rebase progress from `rebase-merge/` or `rebase-apply/` directories.
fn read_rebase_progress(repo: &Repository) -> (Option<usize>, Option<usize>) {
    // Try rebase-merge first (interactive rebase)
    let msgnum = read_git_file(repo, "rebase-merge/msgnum")
        .or_else(|| read_git_file(repo, "rebase-apply/next"));
    let end = read_git_file(repo, "rebase-merge/end")
        .or_else(|| read_git_file(repo, "rebase-apply/last"));

    let current = msgnum.and_then(|s| s.parse::<usize>().ok());
    let total = end.and_then(|s| s.parse::<usize>().ok());

    (current, total)
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflicts
// ─────────────────────────────────────────────────────────────────────────────

fn probe_conflicts(repo: &Repository) -> Vec<ConflictEntry> {
    let index = match repo.index() {
        Ok(idx) => idx,
        Err(e) => {
            warn!("Failed to read index for conflict detection: {e}");
            return Vec::new();
        }
    };

    if !index.has_conflicts() {
        return Vec::new();
    }

    let conflicts = match index.conflicts() {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to iterate conflicts: {e}");
            return Vec::new();
        }
    };

    conflicts
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry
                .our
                .as_ref()
                .or(entry.their.as_ref())
                .or(entry.ancestor.as_ref())?
                .path
                .clone();

            let path_str = String::from_utf8_lossy(&path).to_string();

            let conflict_type = match (&entry.our, &entry.their) {
                (Some(_), Some(_)) => ConflictType::BothModified,
                (None, Some(_)) => ConflictType::DeletedByUs,
                (Some(_), None) => ConflictType::DeletedByThem,
                (None, None) => ConflictType::Unknown,
            };

            let primary_hunk = extract_conflict_hunk(repo.workdir(), &path_str);

            Some(ConflictEntry {
                path: path_str,
                conflict_type,
                primary_hunk,
            })
        })
        .collect()
}

/// Helper to parse the first conflict block (max 20 lines) from a conflicting file.
fn extract_conflict_hunk(workdir: Option<&Path>, rel_path: &str) -> Option<ConflictHunk> {
    let base = workdir?;
    let full_path = base.join(rel_path);
    let content = std::fs::read_to_string(&full_path).ok()?;

    let mut in_ours = false;
    let mut in_theirs = false;
    let mut ours_branch = "HEAD".to_string();
    let mut theirs_branch = "incoming".to_string();
    let mut ours_content = Vec::new();
    let mut theirs_content = Vec::new();
    let mut total_lines = 0;

    for line in content.lines() {
        if line.starts_with("<<<<<<< ") {
            in_ours = true;
            ours_branch = line.trim_start_matches("<<<<<<< ").trim().to_string();
            continue;
        }
        if in_ours && line.starts_with("=======") {
            in_ours = false;
            in_theirs = true;
            continue;
        }
        if in_theirs && line.starts_with(">>>>>>> ") {
            theirs_branch = line.trim_start_matches(">>>>>>> ").trim().to_string();
            break;
        }
        if in_ours {
            if ours_content.len() < 20 {
                ours_content.push(line.to_string());
            }
            total_lines += 1;
        } else if in_theirs {
            if theirs_content.len() < 20 {
                theirs_content.push(line.to_string());
            }
            total_lines += 1;
        }
    }

    if !ours_content.is_empty() || !theirs_content.is_empty() {
        Some(ConflictHunk {
            file_path: rel_path.to_string(),
            ours_branch,
            theirs_branch,
            ours_content,
            theirs_content,
            total_lines,
        })
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Working Tree Status
// ─────────────────────────────────────────────────────────────────────────────

fn probe_working_tree(repo: &Repository) -> WorkingTreeSummary {
    let mut opts = StatusOptions::new();
    opts.show(StatusShow::IndexAndWorkdir)
        .include_untracked(true)
        .recurse_untracked_dirs(false) // Keep it fast — don't recurse into large dirs
        .include_ignored(false);

    let statuses = match repo.statuses(Some(&mut opts)) {
        Ok(s) => s,
        Err(e) => {
            warn!("Failed to read working tree status: {e}");
            return WorkingTreeSummary::default();
        }
    };

    let mut summary = WorkingTreeSummary::default();

    for entry in statuses.iter() {
        let status = entry.status();
        let path = entry.path().unwrap_or("");

        // Staged (index) changes
        if status.is_index_new() {
            summary.staged_new += 1;
            if summary.staged_files.len() < 15 && !path.is_empty() {
                summary.staged_files.push(format!("A {path}"));
            }
        }
        if status.is_index_modified() {
            summary.staged_modified += 1;
            if summary.staged_files.len() < 15 && !path.is_empty() {
                summary.staged_files.push(format!("M {path}"));
            }
        }
        if status.is_index_deleted() {
            summary.staged_deleted += 1;
            if summary.staged_files.len() < 15 && !path.is_empty() {
                summary.staged_files.push(format!("D {path}"));
            }
        }
        if status.is_index_renamed() {
            summary.staged_renamed += 1;
            if summary.staged_files.len() < 15 && !path.is_empty() {
                summary.staged_files.push(format!("R {path}"));
            }
        }

        // Unstaged (workdir) changes
        if status.is_wt_modified() {
            summary.unstaged_modified += 1;
            if summary.unstaged_files.len() < 15 && !path.is_empty() {
                summary.unstaged_files.push(format!("M {path}"));
            }
        }
        if status.is_wt_deleted() {
            summary.unstaged_deleted += 1;
            if summary.unstaged_files.len() < 15 && !path.is_empty() {
                summary.unstaged_files.push(format!("D {path}"));
            }
        }

        // Untracked
        if status.is_wt_new() {
            summary.untracked += 1;
            if summary.untracked_files.len() < 15 && !path.is_empty() {
                summary.untracked_files.push(path.to_string());
            }
        }
    }

    summary
}

impl Default for WorkingTreeSummary {
    fn default() -> Self {
        Self {
            staged_new: 0,
            staged_modified: 0,
            staged_deleted: 0,
            staged_renamed: 0,
            unstaged_modified: 0,
            unstaged_deleted: 0,
            untracked: 0,
            ignored_but_tracked: 0,
            staged_files: Vec::new(),
            unstaged_files: Vec::new(),
            untracked_files: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recent Commits & Incoming Commits
// ─────────────────────────────────────────────────────────────────────────────

fn walk_commits(
    repo: &Repository,
    target_oid: git2::Oid,
    hide_oid: Option<git2::Oid>,
    max_count: usize,
) -> Vec<CommitEntry> {
    let mut revwalk = match repo.revwalk() {
        Ok(rw) => rw,
        Err(_) => return Vec::new(),
    };

    if revwalk.push(target_oid).is_err() {
        return Vec::new();
    }

    if let Some(hide) = hide_oid {
        let _ = revwalk.hide(hide);
    }

    revwalk
        .take(max_count)
        .filter_map(|oid_result| {
            let oid = oid_result.ok()?;
            let commit = repo.find_commit(oid).ok()?;
            let sha_short = oid.to_string()[..7.min(oid.to_string().len())].to_string();
            let message_oneline = commit
                .summary()
                .ok()
                .flatten()
                .unwrap_or("")
                .to_string();
            let author = commit.author().name().unwrap_or("unknown").to_string();
            let timestamp = {
                let time = commit.time();
                let secs = time.seconds();
                chrono::DateTime::from_timestamp(secs, 0)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                    .unwrap_or_else(|| secs.to_string())
            };

            let decorations = collect_decorations(repo, oid);

            Some(CommitEntry {
                sha_short,
                message_oneline,
                author,
                timestamp,
                decorations,
            })
        })
        .collect()
}

fn probe_recent_commits(repo: &Repository) -> Vec<CommitEntry> {
    let head_oid = match repo.head().ok().and_then(|h| h.target()) {
        Some(oid) => oid,
        None => return Vec::new(),
    };
    walk_commits(repo, head_oid, None, MAX_RECENT_COMMITS)
}

fn probe_incoming_commits(
    repo: &Repository,
    tracking: &Option<TrackingInfo>,
    ongoing_operation: &Option<OngoingOperation>,
) -> Vec<CommitEntry> {
    let head_oid = repo.head().ok().and_then(|h| h.target());

    // Case 1: Active branch tracking upstream with behind > 0
    if let Some(tr) = tracking {
        if tr.behind > 0 {
            if let Ok(head) = repo.head() {
                if let Ok(branch_name) = head.shorthand() {
                    if let Ok(local_branch) = repo.find_branch(branch_name, git2::BranchType::Local) {
                        if let Ok(upstream) = local_branch.upstream() {
                            if let Some(upstream_oid) = upstream.get().target() {
                                let commits = walk_commits(repo, upstream_oid, head_oid, 5);
                                if !commits.is_empty() {
                                    return commits;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Case 2: Ongoing Merge or Rebase operation
    if let Some(op) = ongoing_operation {
        match op {
            OngoingOperation::Merge { merge_head } => {
                if let Ok(merge_oid) = git2::Oid::from_str(merge_head) {
                    let commits = walk_commits(repo, merge_oid, head_oid, 5);
                    if !commits.is_empty() {
                        return commits;
                    }
                }
            }
            OngoingOperation::Rebase { onto_branch, .. } => {
                let onto_oid = read_git_file(repo, "rebase-merge/onto")
                    .or_else(|| read_git_file(repo, "rebase-apply/onto"))
                    .and_then(|s| git2::Oid::from_str(&s).ok())
                    .or_else(|| {
                        onto_branch.as_ref().and_then(|b| {
                            repo.resolve_reference_from_short_name(b).ok().and_then(|r| r.target())
                        })
                    });
                if let Some(oid) = onto_oid {
                    let commits = walk_commits(repo, oid, head_oid, 5);
                    if !commits.is_empty() {
                        return commits;
                    }
                }
            }
            _ => {}
        }
    }

    Vec::new()
}

/// Collect branch/tag names that point to a specific commit OID.
fn collect_decorations(repo: &Repository, oid: git2::Oid) -> Vec<String> {
    let mut decorations = Vec::new();

    if let Ok(refs) = repo.references() {
        for reference in refs.flatten() {
            if let Some(target) = reference.target() {
                if target == oid {
                    if let Ok(shorthand) = reference.shorthand() {
                        decorations.push(shorthand.to_string());
                    }
                }
            }
        }
    }

    decorations
}

// ─────────────────────────────────────────────────────────────────────────────
// Reflog
// ─────────────────────────────────────────────────────────────────────────────

fn probe_reflog(repo: &Repository) -> Vec<ReflogEntry> {
    let reflog = match repo.reflog("HEAD") {
        Ok(rl) => rl,
        Err(_) => return Vec::new(),
    };

    (0..reflog.len().min(MAX_REFLOG_ENTRIES))
        .filter_map(|i| {
            let entry = reflog.get(i)?;
            let sha_short = entry.id_new().to_string()[..7].to_string();
            let message = entry.message().ok().flatten().unwrap_or("").to_string();

            // Extract the action type from the reflog message (e.g., "commit", "rebase", "checkout")
            let action = message
                .split(':')
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_string();

            Some(ReflogEntry {
                index: i,
                sha_short,
                action,
                message,
            })
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tracking Info
// ─────────────────────────────────────────────────────────────────────────────

fn probe_tracking(repo: &Repository) -> Option<TrackingInfo> {
    let head = repo.head().ok()?;
    let branch_name = head.shorthand().ok()?;

    let local_branch = repo.find_branch(branch_name, git2::BranchType::Local).ok()?;
    let upstream = local_branch.upstream().ok()?;

    let upstream_name = upstream.name().ok()??.to_string();

    let local_oid = head.target()?;
    let upstream_oid = upstream.get().target()?;

    let (ahead, behind) = repo.graph_ahead_behind(local_oid, upstream_oid).ok()?;

    Some(TrackingInfo {
        upstream: upstream_name,
        ahead,
        behind,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Stash Count
// ─────────────────────────────────────────────────────────────────────────────

fn probe_stash_count(repo: &Repository) -> usize {
    // We need a mutable reference to iterate stashes, but we don't modify anything.
    // libgit2's stash_foreach requires &mut Repository due to internal locking.
    // We work around this by reading the reflog for refs/stash directly.
    let reflog = match repo.reflog("refs/stash") {
        Ok(rl) => rl,
        Err(_) => return 0,
    };
    reflog.len()
}

// ─────────────────────────────────────────────────────────────────────────────
// Submodules
// ─────────────────────────────────────────────────────────────────────────────

fn probe_submodules(repo: &Repository) -> bool {
    repo.submodules().map(|s| !s.is_empty()).unwrap_or(false)
}

fn probe_worktrees(repo: &Repository) -> Vec<WorktreeInfo> {
    let mut result = Vec::new();
    if let Ok(worktrees) = repo.worktrees() {
        for i in 0..worktrees.len() {
            if let Ok(Some(name)) = worktrees.get(i) {
                if let Ok(wt) = repo.find_worktree(name) {
                    let is_locked = wt.is_locked().is_ok();
                    let is_prunable = wt.is_prunable(None).unwrap_or(false);
                    result.push(WorktreeInfo {
                        name: name.to_string(),
                        path: wt.path().to_path_buf(),
                        is_locked,
                        is_prunable,
                        head_branch: None,
                    });
                }
            }
        }
    }
    result
}

fn probe_remotes(repo: &Repository) -> Vec<RemoteInfo> {
    let mut result = Vec::new();
    if let Ok(remotes) = repo.remotes() {
        for i in 0..remotes.len() {
            if let Ok(Some(name)) = remotes.get(i) {
                if let Ok(remote) = repo.find_remote(name) {
                    let fetch_url = remote.url().ok().map(|s| s.to_string());
                    let push_url = remote.pushurl().ok().flatten().map(|s| s.to_string()).or_else(|| fetch_url.clone());
                    result.push(RemoteInfo {
                        name: name.to_string(),
                        fetch_url,
                        push_url,
                    });
                }
            }
        }
    }
    result
}

fn probe_submodule_details(repo: &Repository) -> Vec<SubmoduleState> {
    let mut result = Vec::new();
    if let Ok(submodules) = repo.submodules() {
        for subm in submodules {
            let name = subm.name().ok().unwrap_or_default().to_string();
            let path = subm.path().to_string_lossy().to_string();
            let url = subm.url().ok().flatten().map(|u| u.to_string());
            let head_commit = subm.head_id().map(|id| id.to_string());
            let is_initialized = subm.open().is_ok();
            let is_dirty = repo
                .submodule_status(&name, git2::SubmoduleIgnore::None)
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            result.push(SubmoduleState {
                name,
                path,
                url,
                head_commit,
                is_initialized,
                is_dirty,
            });
        }
    }
    result
}

fn probe_is_shallow(repo: &Repository) -> bool {
    repo.is_shallow()
}

fn probe_large_files_in_index(repo: &Repository, max_bytes: usize) -> Vec<LargeFileEntry> {
    let mut result = Vec::new();
    if let Ok(index) = repo.index() {
        for entry in index.iter() {
            if (entry.file_size as usize) > max_bytes {
                let path = String::from_utf8_lossy(&entry.path).to_string();
                result.push(LargeFileEntry {
                    path,
                    size_bytes: entry.file_size as usize,
                    is_lfs_tracked: false,
                });
            }
        }
    }
    result
}

fn probe_installed_hooks(repo: &Repository) -> Vec<HookInfo> {
    let mut result = Vec::new();
    let hooks_dir = repo.path().join("hooks");
    let hook_names = ["pre-commit", "commit-msg", "pre-push", "post-merge", "pre-rebase"];
    for name in hook_names {
        let hook_path = hooks_dir.join(name);
        if hook_path.exists() && hook_path.is_file() {
            result.push(HookInfo {
                name: name.to_string(),
                is_active: true,
            });
        }
    }
    result
}

fn probe_graph_metrics(repo: &Repository, recent_commits: &[CommitEntry]) -> Option<CommitGraphMetrics> {
    let local_branch_count = repo.branches(Some(git2::BranchType::Local)).map(|b| b.count()).unwrap_or(0);
    let remote_branch_count = repo.branches(Some(git2::BranchType::Remote)).map(|b| b.count()).unwrap_or(0);
    let tag_count = repo.tag_names(None).map(|t| t.len()).unwrap_or(0);

    let mut merge_count = 0;
    let mut has_wip = false;
    for c in recent_commits {
        let lower = c.message_oneline.to_lowercase();
        if lower.starts_with("merge ") {
            merge_count += 1;
        }
        if lower.starts_with("wip") || lower.starts_with("fixup!") || lower.starts_with("squash!") || lower.contains("temp commit") {
            has_wip = true;
        }
    }

    let merge_commit_ratio = if !recent_commits.is_empty() {
        merge_count as f32 / recent_commits.len() as f32
    } else {
        0.0
    };

    Some(CommitGraphMetrics {
        local_branch_count,
        remote_branch_count,
        tag_count,
        merge_commit_ratio,
        has_wip_commits: has_wip,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe Rollback Target
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the safest rollback target based on the current repository state.
///
/// Strategy:
/// 1. If there's an ongoing rebase, the ORIG_HEAD is the pre-rebase state.
/// 2. If there's an ongoing merge, ORIG_HEAD points to pre-merge.
/// 3. Otherwise, check reflog for the most recent clean state.
fn compute_safe_rollback(
    repo: &Repository,
    ongoing_op: &Option<OngoingOperation>,
) -> Option<String> {
    // ORIG_HEAD is set by merge, rebase, and reset operations.
    if ongoing_op.is_some() {
        if let Some(orig_head) = read_git_file(repo, "ORIG_HEAD") {
            return Some(format!("ORIG_HEAD ({sha})", sha = &orig_head[..7.min(orig_head.len())]));
        }
    }

    // Fall back to the reflog: find the most recent entry that represents a clean state.
    let reflog = repo.reflog("HEAD").ok()?;
    for i in 0..reflog.len().min(20) {
        if let Some(entry) = reflog.get(i) {
            let msg = entry.message().ok().flatten().unwrap_or("");
            // Look for checkout, commit, or pull — these typically leave the tree clean.
            if msg.starts_with("commit")
                || msg.starts_with("checkout")
                || msg.starts_with("pull")
            {
                let sha = entry.id_new().to_string();
                return Some(format!(
                    "HEAD@{{{i}}} ({sha_short}): {msg}",
                    sha_short = &sha[..7.min(sha.len())]
                ));
            }
        }
    }

    None
}

/// Enumerate all changed, untracked, or deleted files in the repository with delta metrics and preview.
pub fn probe_changed_files(path: &Path) -> Result<Vec<ChangedFile>, TsugiError> {
    let repo = Repository::discover(path).map_err(|_| TsugiError::RepoNotFound {
        path: path.to_path_buf(),
    })?;

    let repo_root = repo
        .workdir()
        .unwrap_or_else(|| repo.path())
        .to_path_buf();

    let mut opts = StatusOptions::new();
    opts.show(StatusShow::IndexAndWorkdir)
        .include_untracked(true)
        .recurse_untracked_dirs(true)
        .include_ignored(false);

    let statuses = repo.statuses(Some(&mut opts)).map_err(|e| {
        TsugiError::SnapshotFailed(format!("Failed to query repository status: {e}"))
    })?;

    let mut results = Vec::new();

    for entry in statuses.iter() {
        let status = entry.status();
        let path_str = match entry.path() {
            Ok(p) => p.to_string(),
            Err(_) => continue,
        };

        let file_status = if status.is_wt_new() {
            FileDeltaStatus::Untracked
        } else if status.is_index_new() {
            FileDeltaStatus::Added
        } else if status.is_index_deleted() || status.is_wt_deleted() {
            FileDeltaStatus::Deleted
        } else if status.is_index_renamed() {
            FileDeltaStatus::Renamed
        } else {
            FileDeltaStatus::Modified
        };

        let (additions, deletions, diff_preview) =
            extract_file_diff_preview(&repo_root, &path_str, file_status);

        results.push(ChangedFile {
            path: path_str,
            status: file_status,
            additions,
            deletions,
            diff_preview,
        });
    }

    Ok(results)
}

/// Helper to compute lines added/deleted and a concise preview snippet for a file.
fn extract_file_diff_preview(
    repo_root: &Path,
    rel_path: &str,
    status: FileDeltaStatus,
) -> (usize, usize, String) {
    let full_path = repo_root.join(rel_path);

    if status == FileDeltaStatus::Untracked || status == FileDeltaStatus::Added {
        if let Ok(content) = std::fs::read_to_string(&full_path) {
            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len();
            let preview = lines
                .iter()
                .take(8)
                .map(|l| format!("+{l}"))
                .collect::<Vec<_>>()
                .join("\n");
            return (total, 0, preview);
        }
        return (0, 0, String::new());
    }

    // For modified or deleted files, query git diff against HEAD
    let output = std::process::Command::new("git")
        .args(["diff", "HEAD", "-U1", "--", rel_path])
        .current_dir(repo_root)
        .output();

    if let Ok(out) = output {
        let diff_str = String::from_utf8_lossy(&out.stdout);
        let mut adds = 0;
        let mut dels = 0;
        let mut preview_lines = Vec::new();

        for line in diff_str.lines() {
            if line.starts_with("+++") || line.starts_with("---") || line.starts_with("diff ") || line.starts_with("index ") {
                continue;
            }
            if line.starts_with('+') {
                adds += 1;
                if preview_lines.len() < 8 {
                    preview_lines.push(line.to_string());
                }
            } else if line.starts_with('-') {
                dels += 1;
                if preview_lines.len() < 8 {
                    preview_lines.push(line.to_string());
                }
            }
        }

        return (adds, dels, preview_lines.join("\n"));
    }

    (0, 0, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn init_test_repo() -> (TempDir, Repository) {
        let tmp = TempDir::new().expect("Failed to create temp dir");
        let repo = Repository::init(tmp.path()).expect("Failed to init repo");
        (tmp, repo)
    }

    #[test]
    fn test_probe_unborn_repo() {
        let (tmp, _repo) = init_test_repo();
        let state = probe_repo_state(tmp.path()).expect("probe should succeed");

        assert!(matches!(state.head, HeadState::Unborn { .. }));
        assert!(state.ongoing_operation.is_none());
        assert!(state.conflicts.is_empty());
        assert_eq!(state.stash_count, 0);
    }

    #[test]
    fn test_probe_with_initial_commit() {
        let (tmp, repo) = init_test_repo();

        // Create a file and commit it
        let file_path = tmp.path().join("README.md");
        std::fs::write(&file_path, "# Test").expect("write file");

        let mut index = repo.index().expect("get index");
        index
            .add_path(Path::new("README.md"))
            .expect("add to index");
        index.write().expect("write index");
        let tree_id = index.write_tree().expect("write tree");
        let tree = repo.find_tree(tree_id).expect("find tree");
        let sig = git2::Signature::now("Test", "test@test.com").expect("sig");

        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
            .expect("commit");

        let state = probe_repo_state(tmp.path()).expect("probe should succeed");

        assert!(matches!(state.head, HeadState::Attached { .. }));
        assert_eq!(state.recent_commits.len(), 1);
        assert_eq!(state.recent_commits[0].message_oneline, "Initial commit");
    }

    #[test]
    fn test_fact_sheet_output() {
        let (tmp, repo) = init_test_repo();

        // Create a file and commit it
        let file_path = tmp.path().join("test.txt");
        std::fs::write(&file_path, "hello").expect("write file");

        let mut index = repo.index().expect("get index");
        index.add_path(Path::new("test.txt")).expect("add to index");
        index.write().expect("write index");
        let tree_id = index.write_tree().expect("write tree");
        let tree = repo.find_tree(tree_id).expect("find tree");
        let sig = git2::Signature::now("Test", "test@test.com").expect("sig");

        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
            .expect("commit");

        let state = probe_repo_state(tmp.path()).expect("probe should succeed");
        let fact_sheet = state.to_fact_sheet();

        assert!(fact_sheet.starts_with("[REPO_STATE]"));
        assert!(fact_sheet.contains("HEAD: ATTACHED"));
        assert!(fact_sheet.contains("ONGOING_OP: NONE"));
        assert!(fact_sheet.contains("CONFLICTS: NONE"));
    }

    #[test]
    fn test_probe_changed_files() {
        let (tmp, repo) = init_test_repo();

        // 1. Initial commit
        let f1 = tmp.path().join("base.txt");
        std::fs::write(&f1, "base content\n").expect("write f1");

        let mut index = repo.index().expect("index");
        index.add_path(Path::new("base.txt")).expect("add base");
        index.write().expect("write index");
        let tree_id = index.write_tree().expect("tree");
        let tree = repo.find_tree(tree_id).expect("find tree");
        let sig = git2::Signature::now("Test", "test@test.com").expect("sig");
        repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[]).expect("commit");

        // 2. Modify base.txt
        std::fs::write(&f1, "base content\nadded line\n").expect("modify f1");

        // 3. Add an untracked file
        let f2 = tmp.path().join("docs.md");
        std::fs::write(&f2, "# Documentation\nMore docs\n").expect("write f2");

        let changed = probe_changed_files(tmp.path()).expect("probe changed");
        assert_eq!(changed.len(), 2);

        let base_entry = changed.iter().find(|c| c.path == "base.txt").expect("found base");
        assert_eq!(base_entry.status, FileDeltaStatus::Modified);

        let docs_entry = changed.iter().find(|c| c.path == "docs.md").expect("found docs");
        assert_eq!(docs_entry.status, FileDeltaStatus::Untracked);
        assert_eq!(docs_entry.additions, 2);
    }

    #[test]
    fn test_file_lists_in_probe_and_fact_sheet() {
        let (tmp, repo) = init_test_repo();

        // 1. Initial commit
        let f1 = tmp.path().join("base.txt");
        std::fs::write(&f1, "initial\n").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("base.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@test.com").unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[]).unwrap();

        // 2. Stage a new file
        let f2 = tmp.path().join("staged_new.txt");
        std::fs::write(&f2, "staged content\n").unwrap();
        index.add_path(Path::new("staged_new.txt")).unwrap();
        index.write().unwrap();

        // 3. Unstaged modification
        std::fs::write(&f1, "initial\nmodified unstaged\n").unwrap();

        // 4. Untracked file
        let f3 = tmp.path().join("untracked_file.rs");
        std::fs::write(&f3, "fn main() {}\n").unwrap();

        let state = probe_repo_state(tmp.path()).expect("probe should succeed");
        assert_eq!(state.working_tree.staged_files, vec!["A staged_new.txt".to_string()]);
        assert_eq!(state.working_tree.unstaged_files, vec!["M base.txt".to_string()]);
        assert_eq!(state.working_tree.untracked_files, vec!["untracked_file.rs".to_string()]);

        let fact_sheet = state.to_fact_sheet();
        assert!(fact_sheet.contains("STAGED_FILES: [\"A staged_new.txt\"]"));
        assert!(fact_sheet.contains("UNSTAGED_FILES: [\"M base.txt\"]"));
        assert!(fact_sheet.contains("UNTRACKED_FILES: [\"untracked_file.rs\"]"));
    }

    #[test]
    fn test_conflict_previews_and_incoming_commits_in_fact_sheet() {
        let (tmp, _repo) = init_test_repo();
        let mut state = probe_repo_state(tmp.path()).expect("probe should succeed");

        state.incoming_commits = vec![CommitEntry {
            sha_short: "abc1234".to_string(),
            message_oneline: "feat: upstream change".to_string(),
            author: "Alice".to_string(),
            timestamp: "2026-10-31 12:00".to_string(),
            decorations: vec!["origin/main".to_string()],
        }];

        state.conflicts = vec![ConflictEntry {
            path: "src/auth.rs".to_string(),
            conflict_type: ConflictType::BothModified,
            primary_hunk: Some(ConflictHunk {
                file_path: "src/auth.rs".to_string(),
                ours_branch: "HEAD".to_string(),
                theirs_branch: "origin/main".to_string(),
                ours_content: vec!["let token = get_local_token();".to_string()],
                theirs_content: vec!["let token = get_remote_token();".to_string()],
                total_lines: 2,
            }),
        }];

        let fact_sheet = state.to_fact_sheet();
        assert!(fact_sheet.contains("INCOMING_COMMITS:"));
        assert!(fact_sheet.contains("abc1234 feat: upstream change (origin/main)"));
        assert!(fact_sheet.contains("CONFLICT_PREVIEW for \"src/auth.rs\" (ours<HEAD> vs theirs<origin/main>):"));
        assert!(fact_sheet.contains("<<<<<<< HEAD"));
        assert!(fact_sheet.contains("let token = get_local_token();"));
        assert!(fact_sheet.contains("======="));
        assert!(fact_sheet.contains("let token = get_remote_token();"));
        assert!(fact_sheet.contains(">>>>>>> origin/main"));
    }
}

