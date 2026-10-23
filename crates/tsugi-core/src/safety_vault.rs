//! # Safety Vault: Unified Snapshot & Audit Subsystem
//!
//! Encapsulates both zero-cost shadow Git snapshots and the SQLite audit ledger
//! into a single deep module. Callers never handle the database connection or
//! Git ref plumbing directly.
//!
//! ## How It Works
//!
//! 1. **Zero-Side-Effect Checkpoints**: Before executing mutating commands,
//!    `checkpoint` captures the HEAD SHA and runs `git stash create` to generate
//!    a dangling commit for unstaged/untracked changes.
//! 2. **Ref-Pinned Storage**: Stash objects are pinned under `refs/tsugi/snapshots/<timestamp>_<id>`
//!    so Git garbage collection never deletes them.
//! 3. **Audit Ledger Sync**: Metadata, plan details, executed commands, and rollback
//!    status are recorded atomically in the local SQLite ledger.
//! 4. **Instant Rollback**: `rollback_latest` or `rollback_snapshot` restores HEAD and
//!    re-applies dirty working tree state while marking the audit record as rolled back.

use std::path::Path;
use std::process::{Command, Stdio};

use chrono::Utc;
use git2::Repository;
use tracing::{debug, info, warn};

use crate::ledger::{Ledger, OperationRecord};
use crate::types::{ResolutionPlan, SnapshotRecord, TsugiError};

/// Unified safety vault managing shadow git snapshots and operation audit trail.
pub struct SafetyVault {
    ledger: Ledger,
}

/// Normalize a repository path for cross-platform, slash-consistent database recording and querying.
pub fn normalize_repo_path(path: &Path) -> String {
    if let Ok(repo) = Repository::discover(path) {
        if let Some(workdir) = repo.workdir() {
            let s = workdir.to_string_lossy().replace('\\', "/");
            return s.trim_end_matches('/').to_string();
        }
        let s = repo.path().to_string_lossy().replace('\\', "/");
        return s.trim_end_matches('/').to_string();
    }
    path.to_string_lossy().replace('\\', "/").trim_end_matches('/').to_string()
}

impl SafetyVault {
    /// Open the safety vault with a ledger at the default path (`~/.tsugi/ledger.sqlite`).
    pub fn open_default() -> Result<Self, TsugiError> {
        let ledger = Ledger::open_default()?;
        Ok(Self { ledger })
    }

    /// Open the safety vault with a ledger at a specified path.
    pub fn open(db_path: &Path) -> Result<Self, TsugiError> {
        let ledger = Ledger::open(db_path)?;
        Ok(Self { ledger })
    }

    /// Construct a safety vault from an existing ledger instance.
    pub fn from_ledger(ledger: Ledger) -> Self {
        Self { ledger }
    }

    /// Create a zero-cost shadow snapshot before executing a plan.
    ///
    /// Records HEAD commit SHA, creates a stash object for dirty working tree state,
    /// stores a protective git ref, and logs the snapshot in the audit ledger.
    pub fn checkpoint(
        &self,
        repo_path: &Path,
        plan: &ResolutionPlan,
    ) -> Result<SnapshotRecord, TsugiError> {
        let repo = Repository::discover(repo_path).map_err(|e| {
            TsugiError::SnapshotFailed(format!("Failed to open repository: {e}"))
        })?;
        let repo_root = repo.workdir().unwrap_or_else(|| repo.path()).to_path_buf();
        let norm_path = normalize_repo_path(repo_path);

        // 1. Capture HEAD SHA
        let head_sha = repo
            .head()
            .ok()
            .and_then(|h| h.target())
            .map(|oid| oid.to_string())
            .unwrap_or_else(|| "unborn".to_string());

        debug!("SafetyVault checkpoint: HEAD = {}", head_sha);

        // 2. Create stash commit object for working tree without modifying index/refs
        let stash_sha = create_stash_object(&repo_root, &head_sha)?;

        // 3. Store the stash object and head SHA under protective refs so GC does not collect them
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let snapshot_id = uuid::Uuid::new_v4().to_string();

        if let Some(ref sha) = stash_sha {
            let ref_name = format!("refs/tsugi/snapshots/{timestamp}_{id}", id = &snapshot_id[..8]);
            let oid = git2::Oid::from_str(sha).map_err(|e| {
                TsugiError::SnapshotFailed(format!("Invalid stash SHA: {e}"))
            })?;
            repo.reference(&ref_name, oid, true, "tsugi snapshot")
                .map_err(|e| {
                    TsugiError::SnapshotFailed(format!("Failed to create snapshot ref: {e}"))
                })?;
            info!("SafetyVault: pinned stash ref {} -> {}", ref_name, sha);
        }

        // Pin HEAD ref even on a clean tree so git gc never collects pre-reset commit
        if head_sha != "unborn" {
            if let Ok(head_oid) = git2::Oid::from_str(&head_sha) {
                let head_ref_name = format!("refs/tsugi/snapshots/{timestamp}_{id}_head", id = &snapshot_id[..8]);
                let _ = repo.reference(&head_ref_name, head_oid, true, "tsugi head snapshot");
                info!("SafetyVault: pinned head ref {} -> {}", head_ref_name, head_sha);
            }
        }

        // 4. Record snapshot in audit ledger with normalized repository path
        let record = SnapshotRecord {
            id: snapshot_id,
            timestamp: Utc::now().to_rfc3339(),
            repo_path: norm_path,
            head_sha,
            stash_sha,
            plan_summary: plan.root_cause.clone(),
            rolled_back: false,
        };

        self.ledger.insert_snapshot(&record)?;
        info!("SafetyVault: checkpoint created with ID {}", record.id);

        Ok(record)
    }

    /// Roll back the most recent un-reverted snapshot for the given repository.
    ///
    /// Returns `Some(SnapshotRecord)` if rolled back, or `None` if no snapshot was pending.
    pub fn rollback_latest(&self, repo_path: &Path) -> Result<Option<SnapshotRecord>, TsugiError> {
        let norm_path = normalize_repo_path(repo_path);
        let latest = self.ledger.get_latest_snapshot(Some(&norm_path))?;
        if let Some(snapshot) = latest {
            let repo = Repository::discover(repo_path).map_err(|e| {
                TsugiError::SnapshotFailed(format!("Failed to locate repository: {e}"))
            })?;
            let repo_root = repo.workdir().unwrap_or_else(|| repo.path()).to_path_buf();
            self.rollback_snapshot(&repo_root, &snapshot)?;
            Ok(Some(snapshot))
        } else {
            Ok(None)
        }
    }

    /// Roll back a specific snapshot record by resetting HEAD, reapplying stash, and marking rolled back.
    pub fn rollback_snapshot(
        &self,
        repo_path: &Path,
        snapshot: &SnapshotRecord,
    ) -> Result<(), TsugiError> {
        info!("SafetyVault: rolling back snapshot {}", snapshot.id);

        // 1. Reset HEAD to recorded SHA
        if snapshot.head_sha != "unborn" {
            let output = Command::new("git")
                .args(["reset", "--hard", &snapshot.head_sha])
                .current_dir(repo_path)
                .output()
                .map_err(|e| TsugiError::ExecutionFailed {
                    cmd: "git reset --hard".to_string(),
                    reason: e.to_string(),
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(TsugiError::ExecutionFailed {
                    cmd: "git reset --hard".to_string(),
                    reason: stderr.to_string(),
                });
            }
        }

        // 2. Re-apply stash commit object if one was created
        if let Some(ref stash_sha) = snapshot.stash_sha {
            let output = Command::new("git")
                .args(["stash", "apply", stash_sha])
                .current_dir(repo_path)
                .output()
                .map_err(|e| TsugiError::ExecutionFailed {
                    cmd: "git stash apply".to_string(),
                    reason: e.to_string(),
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                warn!("SafetyVault: stash apply had issues (expected if tree was clean): {stderr}");
            }
        }

        // 3. Mark rolled back in ledger
        self.ledger.mark_rolled_back(&snapshot.id)?;
        info!("SafetyVault: snapshot {} restored successfully", snapshot.id);

        Ok(())
    }

    /// Record a resolution operation into the audit ledger.
    pub fn record_execution(
        &self,
        repo_path: &str,
        user_intent: &str,
        plan: &ResolutionPlan,
        executed_commands: &[String],
        success: bool,
        snapshot_id: Option<&str>,
    ) -> Result<(), TsugiError> {
        let norm_path = normalize_repo_path(Path::new(repo_path));
        self.ledger.record_operation(
            &norm_path,
            user_intent,
            &plan.source.to_string(),
            &plan.risk_tier.to_string(),
            executed_commands,
            success,
            snapshot_id,
        )
    }

    /// Query recent operations for audit history or DPO training.
    pub fn recent_operations(
        &self,
        repo_path: Option<&str>,
        limit: usize,
    ) -> Result<Vec<OperationRecord>, TsugiError> {
        let norm_path = repo_path.map(|p| normalize_repo_path(Path::new(p)));
        self.ledger.recent_operations(norm_path.as_deref(), limit)
    }

    /// Retrieve the latest pending snapshot if available (optionally filtered by repo).
    pub fn get_latest_snapshot(&self, repo_path: Option<&str>) -> Result<Option<SnapshotRecord>, TsugiError> {
        self.ledger.get_latest_snapshot(repo_path)
    }

    /// Retrieve the latest pending snapshot discovering repo root from Path.
    pub fn get_latest_snapshot_for_repo(&self, repo_path: &Path) -> Result<Option<SnapshotRecord>, TsugiError> {
        let norm_path = normalize_repo_path(repo_path);
        self.ledger.get_latest_snapshot(Some(&norm_path))
    }

    /// Prune snapshot refs older than `max_age_days`.
    pub fn cleanup(&self, repo_path: &Path, max_age_days: u32) -> Result<usize, TsugiError> {
        let repo = Repository::open(repo_path).map_err(|e| {
            TsugiError::SnapshotFailed(format!("Failed to open repository: {e}"))
        })?;

        let cutoff = Utc::now() - chrono::Duration::days(max_age_days as i64);
        let old_snapshots = self.ledger.get_snapshots_before(&cutoff.to_rfc3339())?;

        let mut cleaned = 0;
        for snapshot in &old_snapshots {
            if let Ok(refs) = repo.references_glob("refs/tsugi/snapshots/*") {
                for mut reference in refs.flatten() {
                    let matches_id = reference
                        .name()
                        .map(|n| n.contains(&snapshot.id[..8]))
                        .unwrap_or(false);
                    if matches_id {
                        let ref_name = reference.name().unwrap_or("unknown").to_string();
                        if let Err(e) = reference.delete() {
                            warn!("SafetyVault: failed to delete snapshot ref {ref_name}: {e}");
                        } else {
                            cleaned += 1;
                        }
                    }
                }
            }

            self.ledger.mark_rolled_back(&snapshot.id)?;
        }

        if cleaned > 0 {
            info!("SafetyVault: cleaned up {cleaned} old snapshot refs");
        }

        Ok(cleaned)
    }

    /// Access the underlying ledger for tests or low-level migrations.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }
}

/// Helper to create a stash commit object using `git stash create` and untracked file capture.
fn create_stash_object(repo_path: &Path, head_sha: &str) -> Result<Option<String>, TsugiError> {
    if head_sha == "unborn" {
        debug!("SafetyVault: unborn repository, no stash possible");
        return Ok(None);
    }

    // 1. Run git stash create to capture staged and unstaged tracked changes
    let output = Command::new("git")
        .args(["stash", "create"])
        .current_dir(repo_path)
        .output()
        .map_err(|e| {
            TsugiError::SnapshotFailed(format!("Failed to spawn git stash create: {e}"))
        })?;

    // Fail-closed on error status (e.g. unmerged index entries / active conflicts)
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(TsugiError::SnapshotFailed(format!(
            "git stash create failed with exit status {}: {}",
            output.status.code().unwrap_or(-1),
            if stderr.is_empty() {
                "unmerged index entries or working tree conflict".to_string()
            } else {
                stderr
            }
        )));
    }

    let raw_stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let tracked_stash_sha = if !raw_stdout.is_empty()
        && raw_stdout.len() == 40
        && raw_stdout.chars().all(|c| c.is_ascii_hexdigit())
    {
        Some(raw_stdout)
    } else {
        None
    };

    // 2. Query untracked files
    let untracked_output = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard"])
        .current_dir(repo_path)
        .output()
        .map_err(|e| {
            TsugiError::SnapshotFailed(format!("Failed to list untracked files: {e}"))
        })?;

    if !untracked_output.status.success() {
        let stderr = String::from_utf8_lossy(&untracked_output.stderr).trim().to_string();
        return Err(TsugiError::SnapshotFailed(format!(
            "git ls-files failed: {stderr}"
        )));
    }

    let untracked_files: Vec<String> = String::from_utf8_lossy(&untracked_output.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    // If there are no untracked files, return the tracked stash object (or None if working tree clean)
    if untracked_files.is_empty() {
        if let Some(ref sha) = tracked_stash_sha {
            debug!("SafetyVault: created stash object for tracked changes: {sha}");
            return Ok(Some(sha.clone()));
        } else {
            debug!("SafetyVault: working tree clean, no stash needed");
            return Ok(None);
        }
    }

    // 3. Untracked files exist: capture them in an isolated temporary index without touching repo state
    let repo = Repository::discover(repo_path).map_err(|e| {
        TsugiError::SnapshotFailed(format!("Failed to discover repository for untracked snapshot: {e}"))
    })?;
    let git_dir = repo.path();
    let tmp_index_path = git_dir.join(format!("tsugi_untracked_{}.idx", uuid::Uuid::new_v4().simple()));

    let mut update_cmd = Command::new("git");
    update_cmd
        .arg("update-index")
        .arg("--add")
        .arg("--stdin")
        .env("GIT_INDEX_FILE", &tmp_index_path)
        .current_dir(repo_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = update_cmd.spawn().map_err(|e| {
        let _ = std::fs::remove_file(&tmp_index_path);
        TsugiError::SnapshotFailed(format!("Failed to spawn git update-index for untracked files: {e}"))
    })?;

    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        for file in &untracked_files {
            let _ = writeln!(stdin, "{}", file);
        }
    }

    let update_res = child.wait_with_output().map_err(|e| {
        let _ = std::fs::remove_file(&tmp_index_path);
        TsugiError::SnapshotFailed(format!("Failed to wait for git update-index: {e}"))
    })?;

    if !update_res.status.success() {
        let _ = std::fs::remove_file(&tmp_index_path);
        let stderr = String::from_utf8_lossy(&update_res.stderr);
        return Err(TsugiError::SnapshotFailed(format!(
            "Failed to index untracked files: {stderr}"
        )));
    }

    // Write tree for untracked files
    let write_tree_output = Command::new("git")
        .arg("write-tree")
        .env("GIT_INDEX_FILE", &tmp_index_path)
        .current_dir(repo_path)
        .output()
        .map_err(|e| {
            let _ = std::fs::remove_file(&tmp_index_path);
            TsugiError::SnapshotFailed(format!("Failed to write tree for untracked files: {e}"))
        })?;

    let _ = std::fs::remove_file(&tmp_index_path);

    if !write_tree_output.status.success() {
        let stderr = String::from_utf8_lossy(&write_tree_output.stderr);
        return Err(TsugiError::SnapshotFailed(format!(
            "git write-tree for untracked files failed: {stderr}"
        )));
    }

    let u_tree = String::from_utf8_lossy(&write_tree_output.stdout).trim().to_string();

    // Commit tree for untracked files
    let u_commit_output = Command::new("git")
        .args(["commit-tree", "-m", "untracked files on tsugi snapshot", &u_tree])
        .current_dir(repo_path)
        .output()
        .map_err(|e| {
            TsugiError::SnapshotFailed(format!("Failed to commit tree for untracked files: {e}"))
        })?;

    if !u_commit_output.status.success() {
        let stderr = String::from_utf8_lossy(&u_commit_output.stderr);
        return Err(TsugiError::SnapshotFailed(format!(
            "git commit-tree for untracked files failed: {stderr}"
        )));
    }

    let u_commit = String::from_utf8_lossy(&u_commit_output.stdout).trim().to_string();

    // 4. Assemble standard 3-parent stash commit object:
    // Parent 1: HEAD
    // Parent 2: Index commit
    // Parent 3: Untracked commit
    let (tree_sha, p1, p2) = if let Some(ref t_sha) = tracked_stash_sha {
        let tree = rev_parse(repo_path, &format!("{t_sha}^{{tree}}"))?;
        let p1 = rev_parse(repo_path, &format!("{t_sha}^1"))?;
        let p2 = rev_parse(repo_path, &format!("{t_sha}^2"))?;
        (tree, p1, p2)
    } else {
        let head_commit = rev_parse(repo_path, "HEAD")?;
        let head_tree = rev_parse(repo_path, "HEAD^{tree}")?;
        let i_commit_output = Command::new("git")
            .args(["commit-tree", "-m", "index on tsugi snapshot", "-p", &head_commit, &head_tree])
            .current_dir(repo_path)
            .output()
            .map_err(|e| TsugiError::SnapshotFailed(format!("Failed to create index commit: {e}")))?;
        let i_commit = String::from_utf8_lossy(&i_commit_output.stdout).trim().to_string();
        (head_tree, head_commit, i_commit)
    };

    let stash_commit_output = Command::new("git")
        .args([
            "commit-tree",
            "-m",
            "WIP on tsugi snapshot (with untracked files)",
            "-p",
            &p1,
            "-p",
            &p2,
            "-p",
            &u_commit,
            &tree_sha,
        ])
        .current_dir(repo_path)
        .output()
        .map_err(|e| TsugiError::SnapshotFailed(format!("Failed to create 3-parent stash commit: {e}")))?;

    if !stash_commit_output.status.success() {
        let stderr = String::from_utf8_lossy(&stash_commit_output.stderr);
        return Err(TsugiError::SnapshotFailed(format!(
            "git commit-tree for stash failed: {stderr}"
        )));
    }

    let final_stash_sha = String::from_utf8_lossy(&stash_commit_output.stdout).trim().to_string();
    debug!("SafetyVault: created 3-parent stash object with untracked files: {final_stash_sha}");
    Ok(Some(final_stash_sha))
}

fn rev_parse(repo_path: &Path, rev: &str) -> Result<String, TsugiError> {
    let output = Command::new("git")
        .args(["rev-parse", rev])
        .current_dir(repo_path)
        .output()
        .map_err(|e| TsugiError::SnapshotFailed(format!("Failed to rev-parse {rev}: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(TsugiError::SnapshotFailed(format!("git rev-parse {rev} failed: {stderr}")));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExecutionStep, PlanSource, RiskTier};
    use tempfile::TempDir;

    fn test_plan() -> ResolutionPlan {
        ResolutionPlan {
            root_cause: "Test fix".to_string(),
            risk_tier: RiskTier::MutatingSafe,
            execution_steps: vec![ExecutionStep {
                cmd: "git status".to_string(),
                explanation: "Check status".to_string(),
                requires_confirmation: false,
            }],
            rollback_cmd: "git status".to_string(),
            source: PlanSource::Tier0,
            confidence: 1.0,
        }
    }

    #[test]
    fn test_safety_vault_checkpoint_and_rollback() {
        let tmp = TempDir::new().unwrap();
        let repo = Repository::init(tmp.path()).unwrap();

        // Create an initial commit
        let test_file = tmp.path().join("file.txt");
        std::fs::write(&test_file, "initial").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("file.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
            .unwrap();

        // Initialize vault with a temp sqlite db
        let db_path = tmp.path().join("vault.sqlite");
        let vault = SafetyVault::open(&db_path).unwrap();

        // Mutate working tree
        std::fs::write(&test_file, "dirty state").unwrap();

        // Create checkpoint
        let plan = test_plan();
        let snapshot = vault.checkpoint(tmp.path(), &plan).unwrap();
        assert_eq!(snapshot.plan_summary, "Test fix");
        assert!(!snapshot.rolled_back);
        assert!(snapshot.stash_sha.is_some());

        // Make another commit
        std::fs::write(&test_file, "committed change").unwrap();
        index.add_path(Path::new("file.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let parent = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Second commit", &tree, &[&parent])
            .unwrap();

        // Verify latest snapshot is found
        let latest = vault.get_latest_snapshot(None).unwrap();
        assert!(latest.is_some());

        // Roll back latest
        let restored = vault.rollback_latest(tmp.path()).unwrap();
        assert!(restored.is_some());

        // Verify snapshot is now marked rolled back
        let after_rollback = vault.get_latest_snapshot(None).unwrap();
        assert!(after_rollback.is_none());
    }

    #[test]
    fn test_safety_vault_pins_head_ref_on_clean_tree() {
        let tmp = TempDir::new().unwrap();
        let repo = Repository::init(tmp.path()).unwrap();

        // Create an initial commit
        let test_file = tmp.path().join("file.txt");
        std::fs::write(&test_file, "initial").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("file.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        let head_commit = repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
            .unwrap();

        // Working tree is completely clean
        let db_tmp = TempDir::new().unwrap();
        let db_path = db_tmp.path().join("vault.sqlite");
        let vault = SafetyVault::open(&db_path).unwrap();
        let plan = test_plan();
        let snapshot = vault.checkpoint(tmp.path(), &plan).unwrap();
        assert!(snapshot.stash_sha.is_none()); // clean tree

        // Verify head ref was pinned under refs/tsugi/snapshots/*_head
        let head_refs: Vec<String> = repo.references_glob("refs/tsugi/snapshots/*_head")
            .unwrap()
            .flatten()
            .map(|r| r.name().unwrap().to_string())
            .collect();
        assert_eq!(head_refs.len(), 1);

        let pinned_target = repo.find_reference(&head_refs[0]).unwrap().target().unwrap();
        assert_eq!(pinned_target, head_commit);
    }

    #[test]
    fn test_safety_vault_cross_repo_rollback_isolation() {
        let tmp = TempDir::new().unwrap();
        let repo_a_path = tmp.path().join("repo_a");
        let repo_b_path = tmp.path().join("repo_b");
        std::fs::create_dir_all(&repo_a_path).unwrap();
        std::fs::create_dir_all(&repo_b_path).unwrap();

        let repo_a = Repository::init(&repo_a_path).unwrap();
        let _repo_b = Repository::init(&repo_b_path).unwrap();

        // Create commit in repo_a
        let file_a = repo_a_path.join("a.txt");
        std::fs::write(&file_a, "initial a").unwrap();
        let mut idx_a = repo_a.index().unwrap();
        idx_a.add_path(Path::new("a.txt")).unwrap();
        idx_a.write().unwrap();
        let tree_id_a = idx_a.write_tree().unwrap();
        let tree_a = repo_a.find_tree(tree_id_a).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        repo_a.commit(Some("HEAD"), &sig, &sig, "A commit", &tree_a, &[]).unwrap();

        let db_path = tmp.path().join("shared_vault.sqlite");
        let vault = SafetyVault::open(&db_path).unwrap();

        // Checkpoint in repo A
        let plan = test_plan();
        let snapshot_a = vault.checkpoint(&repo_a_path, &plan).unwrap();
        assert_eq!(snapshot_a.repo_path, normalize_repo_path(&repo_a_path));

        // Attempting to rollback repo B should return None (not touch repo A's snapshot)
        let rollback_b = vault.rollback_latest(&repo_b_path).unwrap();
        assert!(rollback_b.is_none());

        // Rollback repo A succeeds
        let rollback_a = vault.rollback_latest(&repo_a_path).unwrap();
        assert!(rollback_a.is_some());
        assert_eq!(rollback_a.unwrap().id, snapshot_a.id);
    }

    #[test]
    fn test_safety_vault_record_and_recent_operations() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("vault.sqlite");
        let vault = SafetyVault::open(&db_path).unwrap();
        let plan = test_plan();

        vault
            .record_execution(
                "/tmp/test_repo",
                "fix merge",
                &plan,
                &["git merge --continue".to_string()],
                true,
                None,
            )
            .unwrap();

        let ops = vault.recent_operations(Some("/tmp/test_repo"), 10).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].user_intent, "fix merge");
        assert!(ops[0].success);
    }

    #[test]
    fn test_safety_vault_captures_untracked_files_and_restores() {
        let tmp = TempDir::new().unwrap();
        let repo = Repository::init(tmp.path()).unwrap();

        // Initial commit so we have a HEAD
        let test_file = tmp.path().join("tracked.txt");
        std::fs::write(&test_file, "initial").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("tracked.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[]).unwrap();

        // Create an untracked file
        let untracked_file = tmp.path().join("untracked.txt");
        std::fs::write(&untracked_file, "untracked secret data").unwrap();

        let db_path = tmp.path().join("vault.sqlite");
        let vault = SafetyVault::open(&db_path).unwrap();
        let plan = test_plan();

        // Checkpoint captures untracked files into 3-parent stash commit
        let snapshot = vault.checkpoint(tmp.path(), &plan).unwrap();
        assert!(snapshot.stash_sha.is_some());

        // Simulate destructive operation: delete the untracked file and modify tracked file
        std::fs::remove_file(&untracked_file).unwrap();
        std::fs::write(&test_file, "corrupted tracked data").unwrap();
        assert!(!untracked_file.exists());

        // Roll back
        let rolled_back = vault.rollback_latest(tmp.path()).unwrap();
        assert!(rolled_back.is_some());

        // Verify untracked file was restored
        assert!(untracked_file.exists());
        let content = std::fs::read_to_string(&untracked_file).unwrap();
        assert_eq!(content, "untracked secret data");
    }

    #[test]
    fn test_safety_vault_fails_closed_on_unmerged_index_entries() {
        let tmp = TempDir::new().unwrap();
        let repo = Repository::init(tmp.path()).unwrap();
        let file_path = tmp.path().join("conflict.txt");
        std::fs::write(&file_path, "base\n").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("conflict.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        let base_commit = repo.commit(Some("HEAD"), &sig, &sig, "Base commit", &tree, &[]).unwrap();

        // Create branch b1
        let base_obj = repo.find_object(base_commit, None).unwrap();
        let _ = repo.branch("b1", base_obj.as_commit().unwrap(), false).unwrap();

        // Commit on master
        std::fs::write(&file_path, "master change\n").unwrap();
        index.add_path(Path::new("conflict.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let master_parent = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Master change", &tree, &[&master_parent]).unwrap();

        // Switch to b1 and commit conflicting change
        repo.set_head("refs/heads/b1").unwrap();
        repo.checkout_head(Some(git2::build::CheckoutBuilder::new().force())).unwrap();
        std::fs::write(&file_path, "branch change\n").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("conflict.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let b1_parent = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "B1 change", &tree, &[&b1_parent]).unwrap();

        // Switch back to master
        repo.set_head("refs/heads/master").unwrap();
        repo.checkout_head(Some(git2::build::CheckoutBuilder::new().force())).unwrap();

        // Merge b1 into master to create conflict
        let b1_annotated = repo.find_annotated_commit(repo.resolve_reference_from_short_name("b1").unwrap().target().unwrap()).unwrap();
        repo.merge(&[&b1_annotated], None, None).unwrap();
        assert!(repo.index().unwrap().has_conflicts());

        // Checkpoint must fail-closed because index has unmerged entries
        let db_path = tmp.path().join("vault.sqlite");
        let vault = SafetyVault::open(&db_path).unwrap();
        let plan = test_plan();
        let res = vault.checkpoint(tmp.path(), &plan);
        assert!(res.is_err(), "Checkpoint should fail-closed on unmerged conflict state");
    }
}
