//! Core type definitions shared across all layers.
//!
//! These types form the contract between the probe (state collector), the resolution
//! engines (tier0 + LLM), the guardrail (policy enforcement), and the executor.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────────────────────
// Repository State (output of Layer 1: Probe)
// ─────────────────────────────────────────────────────────────────────────────

/// Complete snapshot of the Git repository's current state, computed deterministically
/// via libgit2 in <10ms. This is the *only* input the LLM receives — it never sees
/// raw terminal output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoState {
    /// Absolute path to the repository root.
    pub repo_root: PathBuf,

    /// Current HEAD state.
    pub head: HeadState,

    /// Any in-progress Git operation (rebase, merge, cherry-pick, etc.).
    pub ongoing_operation: Option<OngoingOperation>,

    /// Files with merge/rebase conflicts.
    pub conflicts: Vec<ConflictEntry>,

    /// Summary of the working tree status.
    pub working_tree: WorkingTreeSummary,

    /// Recent commit log (last 10 entries, one-line format).
    pub recent_commits: Vec<CommitEntry>,

    /// Recent reflog entries (last 10).
    pub recent_reflog: Vec<ReflogEntry>,

    /// Current branch tracking info (upstream, ahead/behind counts).
    pub tracking: Option<TrackingInfo>,

    /// Stash count.
    pub stash_count: usize,

    /// Whether the repo has submodules.
    pub has_submodules: bool,

    /// Safe rollback target (if identifiable).
    pub safe_rollback_ref: Option<String>,

    /// Active git worktrees.
    #[serde(default)]
    pub worktrees: Vec<WorktreeInfo>,

    /// Configured remotes.
    #[serde(default)]
    pub remotes: Vec<RemoteInfo>,

    /// Submodule states.
    #[serde(default)]
    pub submodules: Vec<SubmoduleState>,

    /// Whether the repo is a shallow clone.
    #[serde(default)]
    pub is_shallow: bool,

    /// Large files staged in the index (e.g. >5MB without LFS).
    #[serde(default)]
    pub large_files: Vec<LargeFileEntry>,

    /// Installed hooks.
    #[serde(default)]
    pub installed_hooks: Vec<HookInfo>,

    /// Graph metrics (branches, tags, merge commits).
    #[serde(default)]
    pub graph_metrics: Option<CommitGraphMetrics>,

    /// Incoming commits on upstream or merge target not yet integrated into HEAD.
    #[serde(default)]
    pub incoming_commits: Vec<CommitEntry>,
}

/// Information about a linked git worktree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorktreeInfo {
    pub name: String,
    pub path: PathBuf,
    pub is_locked: bool,
    pub is_prunable: bool,
    pub head_branch: Option<String>,
}

/// Remote tracking configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteInfo {
    pub name: String,
    pub fetch_url: Option<String>,
    pub push_url: Option<String>,
}

/// Information about a git submodule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubmoduleState {
    pub name: String,
    pub path: String,
    pub url: Option<String>,
    pub head_commit: Option<String>,
    pub is_initialized: bool,
    pub is_dirty: bool,
}

/// Large file entry in index (>5MB or matching known large file patterns).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LargeFileEntry {
    pub path: String,
    pub size_bytes: usize,
    pub is_lfs_tracked: bool,
}

/// Git hook information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HookInfo {
    pub name: String,
    pub is_active: bool,
}

/// Graph metrics about the repository commit history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommitGraphMetrics {
    pub local_branch_count: usize,
    pub remote_branch_count: usize,
    pub tag_count: usize,
    pub merge_commit_ratio: f32,
    pub has_wip_commits: bool,
}

/// A conversation turn in a REPL session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionTurn {
    pub user_prompt: String,
    pub assistant_response: String,
    pub executed: bool,
}

/// HEAD can be attached to a branch, detached at a commit, or unborn (empty repo).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeadState {
    Attached {
        branch: String,
        commit_sha: String,
    },
    Detached {
        commit_sha: String,
    },
    Unborn {
        branch: String,
    },
}

/// An in-progress multi-step Git operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OngoingOperation {
    Merge {
        /// SHA of the commit being merged.
        merge_head: String,
    },
    Rebase {
        /// Interactive or non-interactive.
        interactive: bool,
        /// Current step (1-indexed).
        current_step: Option<usize>,
        /// Total steps.
        total_steps: Option<usize>,
        /// The branch being rebased onto.
        onto_branch: Option<String>,
    },
    CherryPick {
        cherry_pick_head: String,
    },
    Revert {
        revert_head: String,
    },
    Bisect {
        good: Option<String>,
        bad: Option<String>,
    },
    ApplyMailbox,
}

/// Extracted conflict hunk lines for inspection and resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictHunk {
    pub file_path: String,
    pub ours_branch: String,
    pub theirs_branch: String,
    pub ours_content: Vec<String>,
    pub theirs_content: Vec<String>,
    pub total_lines: usize,
}

/// A file with an active conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictEntry {
    pub path: String,
    pub conflict_type: ConflictType,
    #[serde(default)]
    pub primary_hunk: Option<ConflictHunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    BothModified,
    DeletedByUs,
    DeletedByThem,
    BothAdded,
    RenamedModified,
    Unknown,
}

/// Compact summary of the working tree status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingTreeSummary {
    pub staged_new: usize,
    pub staged_modified: usize,
    pub staged_deleted: usize,
    pub staged_renamed: usize,
    pub unstaged_modified: usize,
    pub unstaged_deleted: usize,
    pub untracked: usize,
    /// Files matching patterns in `.gitignore` that are tracked (potential secrets).
    pub ignored_but_tracked: usize,
    /// Actual staged file paths with status indicators (e.g. "M src/auth.rs", "A src/config.rs"), capped at 15.
    #[serde(default)]
    pub staged_files: Vec<String>,
    /// Actual unstaged modified/deleted file paths (e.g. "M src/api.rs", "D README.md"), capped at 15.
    #[serde(default)]
    pub unstaged_files: Vec<String>,
    /// Actual untracked file paths (e.g. "notes.txt"), capped at 15.
    #[serde(default)]
    pub untracked_files: Vec<String>,
}

impl WorkingTreeSummary {
    /// Returns true if there are any staged, unstaged, or untracked changes.
    pub fn has_changes(&self) -> bool {
        self.staged_new > 0
            || self.staged_modified > 0
            || self.staged_deleted > 0
            || self.staged_renamed > 0
            || self.unstaged_modified > 0
            || self.unstaged_deleted > 0
            || self.untracked > 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Commit Splitting Types
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a changed file in the working tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileDeltaStatus {
    Added,
    Modified,
    Deleted,
    Renamed,
    Untracked,
}

impl std::fmt::Display for FileDeltaStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Added => write!(f, "added"),
            Self::Modified => write!(f, "modified"),
            Self::Deleted => write!(f, "deleted"),
            Self::Renamed => write!(f, "renamed"),
            Self::Untracked => write!(f, "untracked"),
        }
    }
}

/// Metadata for a changed file in the working tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangedFile {
    pub path: String,
    pub status: FileDeltaStatus,
    pub additions: usize,
    pub deletions: usize,
    /// Up to 10 lines of leading diff summary for semantic categorization
    pub diff_preview: String,
}

/// A single proposed commit within a split plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposedCommit {
    pub commit_type: String,       // feat, fix, docs, test, refactor, chore, perf, style, ci
    pub scope: Option<String>,     // e.g. core, cli, safety, deps
    pub message: String,           // Conventional Commit message: "feat(core): add safety vault"
    pub explanation: String,       // Why these files are grouped together
    pub files: Vec<String>,        // File paths belonging to this commit
}

/// Full execution plan for splitting working tree changes into atomic commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitCommitPlan {
    pub summary: String,
    pub commits: Vec<ProposedCommit>,
    #[serde(default)]
    pub unassigned_files: Vec<String>,
}

/// A single commit entry for the recent log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitEntry {
    pub sha_short: String,
    pub message_oneline: String,
    pub author: String,
    pub timestamp: String,
    pub decorations: Vec<String>,
}

/// A single reflog entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflogEntry {
    pub index: usize,
    pub sha_short: String,
    pub action: String,
    pub message: String,
}

/// Branch tracking information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingInfo {
    pub upstream: String,
    pub ahead: usize,
    pub behind: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolution Plan (output of Layer 2/3: Tier0 or LLM Engine)
// ─────────────────────────────────────────────────────────────────────────────

/// A structured resolution plan — the universal output format for both the Tier-0
/// fast-path engine and the LLM inference engine. Enforced via GBNF grammar when
/// generated by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionPlan {
    /// Human-readable diagnosis of the root cause.
    pub root_cause: String,

    /// Risk classification for the planned operation.
    pub risk_tier: RiskTier,

    /// Steps to execute, in order.
    pub execution_steps: Vec<ExecutionStep>,

    /// Command to fully reverse the operation if anything goes wrong.
    pub rollback_cmd: String,

    /// Which engine produced this plan.
    pub source: PlanSource,

    /// Confidence score (0.0 – 1.0). Tier-0 plans always have 1.0.
    pub confidence: f64,
}

/// Risk classification that determines the level of user confirmation required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskTier {
    /// Read-only or trivially reversible (e.g., `git stash`, `git checkout -- file`).
    ReadOnly,

    /// Mutating but safely reversible (e.g., `git checkout -b`, `git rebase --continue`).
    MutatingSafe,

    /// Potentially destructive — requires explicit user confirmation
    /// (e.g., `git reset --hard`, `git push --force-with-lease`).
    MutatingDangerous,

    /// History-rewriting operation that affects remote collaborators
    /// (e.g., `git push --force` to shared branches).
    HistoryRewrite,
}

/// A single step in the resolution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    /// The exact Git command to execute.
    pub cmd: String,

    /// Human-readable explanation of what this step does.
    pub explanation: String,

    /// Whether this step requires user confirmation before execution.
    pub requires_confirmation: bool,
}

/// Which engine produced the resolution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanSource {
    /// Deterministic Tier-0 fast-path pattern match.
    Tier0,
    /// LLM inference (Qwen2.5-Coder-1.5B primary).
    LlmPrimary,
    /// LLM inference (Qwen3-0.6B low-memory fallback).
    LlmFallback,
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot & Ledger Types (Layer 4: Guardrail)
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot record stored in the local SQLite audit ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRecord {
    /// Unique ID for this snapshot.
    pub id: String,

    /// Timestamp when the snapshot was taken.
    pub timestamp: String,

    /// Absolute path to the repository root.
    #[serde(default)]
    pub repo_path: String,

    /// HEAD SHA at snapshot time.
    pub head_sha: String,

    /// Stash object SHA (created via `git stash create`) if dirty working tree.
    pub stash_sha: Option<String>,

    /// The resolution plan that triggered this snapshot.
    pub plan_summary: String,

    /// Whether this snapshot has been used for rollback.
    pub rolled_back: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Model tier selection based on available system memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTier {
    /// Qwen2.5-Coder-1.5B-Instruct (IQ4_XS) — ~840MB RAM. Default when >= 1.2GB free.
    Primary,
    /// Qwen3-0.6B (Q5_K_M) — ~420MB RAM. Fallback when < 1.2GB free.
    Fallback,
}

/// Runtime configuration for the tsugi engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsugiConfig {
    /// Path to the primary GGUF model file.
    pub primary_model_path: Option<PathBuf>,

    /// Path to the fallback GGUF model file.
    pub fallback_model_path: Option<PathBuf>,

    /// Maximum context length (in tokens) for inference.
    pub max_context_tokens: usize,

    /// Maximum generation tokens.
    pub max_generation_tokens: usize,

    /// Memory threshold (in bytes) below which the engine falls back to the smaller model.
    pub fallback_memory_threshold_bytes: u64,

    /// Whether to require explicit confirmation for dangerous operations.
    pub require_confirmation_for_dangerous: bool,

    /// Whether to auto-create snapshots before mutations.
    pub auto_snapshot: bool,

    /// Path to the GBNF grammar file for constrained generation.
    pub grammar_path: Option<PathBuf>,

    /// Number of threads for llama.cpp inference.
    pub n_threads: usize,

    /// Enable prompt lookup decoding for CPU speedup.
    pub prompt_lookup_decoding: bool,

    /// Optional HTTP endpoint for local llama-server (e.g. "http://127.0.0.1:8080").
    #[serde(default)]
    pub llama_server_url: Option<String>,

    /// Explicit path to llama-cli executable.
    #[serde(default)]
    pub llama_cli_path: Option<PathBuf>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in the tsugi-core library.
#[derive(Debug, thiserror::Error)]
pub enum TsugiError {
    #[error("Git repository error: {0}")]
    Git(#[from] git2::Error),

    #[error("Repository not found at: {path}")]
    RepoNotFound { path: PathBuf },

    #[error("No ongoing Git operation detected — nothing to fix")]
    NoIssueDetected,

    #[error("Model file not found at: {path}")]
    ModelNotFound { path: PathBuf },

    #[error("Inference engine error: {0}")]
    InferenceError(String),

    #[error("Guardrail policy violation: {0}")]
    PolicyViolation(String),

    #[error("Snapshot creation failed: {0}")]
    SnapshotFailed(String),

    #[error("Ledger database error: {0}")]
    LedgerError(String),

    #[error("Command execution failed: {cmd} — {reason}")]
    ExecutionFailed { cmd: String, reason: String },

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Insufficient memory: available {available_mb}MB, required {required_mb}MB")]
    InsufficientMemory { available_mb: u64, required_mb: u64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Display implementations for terminal rendering
// ─────────────────────────────────────────────────────────────────────────────

impl fmt::Display for HeadState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeadState::Attached { branch, commit_sha } => {
                write!(f, "ATTACHED to {branch} ({sha})", sha = &commit_sha[..7.min(commit_sha.len())])
            }
            HeadState::Detached { commit_sha } => {
                write!(f, "DETACHED at {sha}", sha = &commit_sha[..7.min(commit_sha.len())])
            }
            HeadState::Unborn { branch } => {
                write!(f, "UNBORN ({branch}, no commits yet)")
            }
        }
    }
}

impl fmt::Display for OngoingOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OngoingOperation::Merge { merge_head } => {
                write!(f, "MERGE in progress (merging {sha})", sha = &merge_head[..7.min(merge_head.len())])
            }
            OngoingOperation::Rebase { interactive, current_step, total_steps, onto_branch } => {
                let mode = if *interactive { "INTERACTIVE_REBASE" } else { "REBASE" };
                match (current_step, total_steps) {
                    (Some(cur), Some(total)) => write!(f, "{mode} in progress (step {cur}/{total})"),
                    _ => write!(f, "{mode} in progress"),
                }?;
                if let Some(onto) = onto_branch {
                    write!(f, " onto {onto}")?;
                }
                Ok(())
            }
            OngoingOperation::CherryPick { cherry_pick_head } => {
                write!(f, "CHERRY-PICK in progress ({sha})",
                    sha = &cherry_pick_head[..7.min(cherry_pick_head.len())])
            }
            OngoingOperation::Revert { revert_head } => {
                write!(f, "REVERT in progress ({sha})",
                    sha = &revert_head[..7.min(revert_head.len())])
            }
            OngoingOperation::Bisect { .. } => write!(f, "BISECT in progress"),
            OngoingOperation::ApplyMailbox => write!(f, "APPLY-MAILBOX in progress"),
        }
    }
}

impl fmt::Display for RiskTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskTier::ReadOnly => write!(f, "READ_ONLY"),
            RiskTier::MutatingSafe => write!(f, "MUTATING_SAFE"),
            RiskTier::MutatingDangerous => write!(f, "MUTATING_DANGEROUS"),
            RiskTier::HistoryRewrite => write!(f, "HISTORY_REWRITE"),
        }
    }
}

impl fmt::Display for PlanSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanSource::Tier0 => write!(f, "Tier-0 (deterministic)"),
            PlanSource::LlmPrimary => write!(f, "Qwen2.5-Coder-1.5B"),
            PlanSource::LlmFallback => write!(f, "Qwen3-0.6B (low-memory)"),
        }
    }
}

impl fmt::Display for ConflictType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConflictType::BothModified => write!(f, "both_modified"),
            ConflictType::DeletedByUs => write!(f, "deleted_by_us"),
            ConflictType::DeletedByThem => write!(f, "deleted_by_them"),
            ConflictType::BothAdded => write!(f, "both_added"),
            ConflictType::RenamedModified => write!(f, "renamed_modified"),
            ConflictType::Unknown => write!(f, "unknown"),
        }
    }
}

impl RepoState {
    /// Render this state as the compact Symbolic State Fact Sheet (<= 250 tokens)
    /// that gets fed to the LLM.
    pub fn to_fact_sheet(&self) -> String {
        let mut lines = Vec::with_capacity(16);

        lines.push(format!("[REPO_STATE]"));
        lines.push(format!("HEAD: {}", self.head));

        if let Some(ref op) = self.ongoing_operation {
            lines.push(format!("ONGOING_OP: {op}"));
        } else {
            lines.push("ONGOING_OP: NONE".to_string());
        }

        if self.conflicts.is_empty() {
            lines.push("CONFLICTS: NONE".to_string());
        } else {
            let conflict_list: Vec<String> = self.conflicts.iter()
                .map(|c| format!("\"{}\" ({})", c.path, c.conflict_type))
                .collect();
            lines.push(format!("CONFLICTS: [{}]", conflict_list.join(", ")));

            // Extract compact 10-line preview of the first conflict block for up to 3 conflicted files
            let mut preview_count = 0;
            for c in &self.conflicts {
                if preview_count >= 3 {
                    break;
                }
                if let Some(ref hunk) = c.primary_hunk {
                    lines.push(format!(
                        "CONFLICT_PREVIEW for \"{}\" (ours<{}> vs theirs<{}>):",
                        hunk.file_path, hunk.ours_branch, hunk.theirs_branch
                    ));
                    lines.push(format!("<<<<<<< {}", hunk.ours_branch));
                    for line in hunk.ours_content.iter().take(4) {
                        lines.push(format!("  {}", line));
                    }
                    if hunk.ours_content.len() > 4 {
                        lines.push("  ...".to_string());
                    }
                    lines.push("=======".to_string());
                    for line in hunk.theirs_content.iter().take(4) {
                        lines.push(format!("  {}", line));
                    }
                    if hunk.theirs_content.len() > 4 {
                        lines.push("  ...".to_string());
                    }
                    lines.push(format!(">>>>>>> {}", hunk.theirs_branch));
                    preview_count += 1;
                }
            }
        }

        let wt = &self.working_tree;
        lines.push(format!(
            "STAGED: +{} ~{} -{} R{} | UNSTAGED: ~{} -{} | UNTRACKED: {}",
            wt.staged_new, wt.staged_modified, wt.staged_deleted, wt.staged_renamed,
            wt.unstaged_modified, wt.unstaged_deleted, wt.untracked
        ));

        if !wt.staged_files.is_empty() {
            let file_list: Vec<String> = wt.staged_files.iter().take(15).map(|f| format!("\"{}\"", f)).collect();
            lines.push(format!("STAGED_FILES: [{}]", file_list.join(", ")));
        }

        if !wt.unstaged_files.is_empty() {
            let file_list: Vec<String> = wt.unstaged_files.iter().take(15).map(|f| format!("\"{}\"", f)).collect();
            lines.push(format!("UNSTAGED_FILES: [{}]", file_list.join(", ")));
        }

        if !wt.untracked_files.is_empty() {
            let file_list: Vec<String> = wt.untracked_files.iter().take(15).map(|f| format!("\"{}\"", f)).collect();
            lines.push(format!("UNTRACKED_FILES: [{}]", file_list.join(", ")));
        }

        if let Some(ref tracking) = self.tracking {
            lines.push(format!(
                "TRACKING: {} (ahead {}, behind {})",
                tracking.upstream, tracking.ahead, tracking.behind
            ));
        }

        if self.stash_count > 0 {
            lines.push(format!("STASHES: {}", self.stash_count));
        }

        if let Some(ref rollback) = self.safe_rollback_ref {
            lines.push(format!("SAFE_ROLLBACK_TARGET: {rollback}"));
        }

        if self.is_shallow {
            lines.push("SHALLOW_CLONE: true (history is truncated)".to_string());
        }

        if !self.remotes.is_empty() {
            let remote_strs: Vec<String> = self.remotes.iter().map(|r| {
                format!("{}:{}", r.name, r.fetch_url.as_deref().unwrap_or("none"))
            }).collect();
            lines.push(format!("REMOTES: [{}]", remote_strs.join(", ")));
        }

        if !self.submodules.is_empty() {
            let uninit = self.submodules.iter().filter(|s| !s.is_initialized).count();
            let dirty = self.submodules.iter().filter(|s| s.is_dirty).count();
            lines.push(format!("SUBMODULES: {} total ({} uninitialized, {} dirty)", self.submodules.len(), uninit, dirty));
        }

        if self.worktrees.len() > 1 {
            lines.push(format!("WORKTREES: {} active linked worktrees", self.worktrees.len()));
        }

        if !self.large_files.is_empty() {
            let file_names: Vec<String> = self.large_files.iter().map(|f| format!("{} ({}MB)", f.path, f.size_bytes / 1_000_000)).collect();
            lines.push(format!("LARGE_FILES_STAGED: [{}]", file_names.join(", ")));
        }

        if let Some(ref gm) = self.graph_metrics {
            if gm.has_wip_commits {
                lines.push("GRAPH_SIGNAL: WIP/fixup commits detected in recent history".to_string());
            }
        }

        // Append incoming commits from upstream or ongoing rebase/merge
        if !self.incoming_commits.is_empty() {
            lines.push("INCOMING_COMMITS:".to_string());
            for c in self.incoming_commits.iter().take(5) {
                let decorations = if c.decorations.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", c.decorations.join(", "))
                };
                lines.push(format!("  {} {}{}", c.sha_short, c.message_oneline, decorations));
            }
        }

        // Append recent commits (compact)
        if !self.recent_commits.is_empty() {
            lines.push("RECENT_COMMITS:".to_string());
            for c in self.recent_commits.iter().take(5) {
                let decorations = if c.decorations.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", c.decorations.join(", "))
                };
                lines.push(format!("  {} {}{}", c.sha_short, c.message_oneline, decorations));
            }
        }

        // Append reflog (compact, only if relevant)
        if self.ongoing_operation.is_some() && !self.recent_reflog.is_empty() {
            lines.push("REFLOG:".to_string());
            for r in self.recent_reflog.iter().take(5) {
                lines.push(format!("  {}: {} {}", r.index, r.sha_short, r.message));
            }
        }

        lines.join("\n")
    }
}

impl Default for TsugiConfig {
    fn default() -> Self {
        Self {
            primary_model_path: None,
            fallback_model_path: None,
            max_context_tokens: 1024,
            max_generation_tokens: 256,
            // 1.2 GB in bytes
            fallback_memory_threshold_bytes: 1_200_000_000,
            require_confirmation_for_dangerous: true,
            auto_snapshot: true,
            grammar_path: None,
            n_threads: num_cpus(),
            prompt_lookup_decoding: true,
            llama_server_url: None,
            llama_cli_path: None,
        }
    }
}

/// Detect the number of physical CPU cores (capped at 4 for efficiency on low-end machines).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get().min(4))
        .unwrap_or(2)
}
