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
use std::process::Command;

use chrono::Utc;
use git2::Repository;
use tracing::{debug, info, warn};

use crate::ledger::{Ledger, OperationRecord};
use crate::types::{ResolutionPlan, SnapshotRecord, TsugiError};

/// Unified safety vault managing shadow git snapshots and operation audit trail.
pub struct SafetyVault {
    ledger: Ledger,
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
        let repo = Repository::open(repo_path).map_err(|e| {
            TsugiError::SnapshotFailed(format!("Failed to open repository: {e}"))
        })?;

        // 1. Capture HEAD SHA
        let head_sha = repo
            .head()
            .ok()
            .and_then(|h| h.target())
            .map(|oid| oid.to_string())
            .unwrap_or_else(|| "unborn".to_string());

        debug!("SafetyVault checkpoint: HEAD = {}", head_sha);

        // 2. Create stash commit object for working tree without modifying index/refs
        let stash_sha = create_stash_object(repo_path)?;

        // 3. Store the stash object under a protective ref so GC does not collect it
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let snapshot_id = uuid::Uuid::new_v4().to_string();
        let ref_name = format!("refs/tsugi/snapshots/{timestamp}_{id}", id = &snapshot_id[..8]);

        if let Some(ref sha) = stash_sha {
            let oid = git2::Oid::from_str(sha).map_err(|e| {
                TsugiError::SnapshotFailed(format!("Invalid stash SHA: {e}"))
            })?;
            repo.reference(&ref_name, oid, true, "tsugi snapshot")
                .map_err(|e| {
                    TsugiError::SnapshotFailed(format!("Failed to create snapshot ref: {e}"))
                })?;
            info!("SafetyVault: pinned ref {} -> {}", ref_name, sha);
        }

        // 4. Record snapshot in audit ledger
        let record = SnapshotRecord {
            id: snapshot_id,
            timestamp: Utc::now().to_rfc3339(),
            head_sha,
            stash_sha,
            plan_summary: plan.root_cause.clone(),
            rolled_back: false,
        };

        self.ledger.insert_snapshot(&record)?;
        info!("SafetyVault: checkpoint created with ID {}", record.id);

        Ok(record)
    }

    /// Roll back the most recent un-reverted snapshot.
    ///
    /// Returns `Some(SnapshotRecord)` if rolled back, or `None` if no snapshot was pending.
    pub fn rollback_latest(&self, repo_path: &Path) -> Result<Option<SnapshotRecord>, TsugiError> {
        let latest = self.ledger.get_latest_snapshot()?;
        if let Some(snapshot) = latest {
            self.rollback_snapshot(repo_path, &snapshot)?;
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
        self.ledger.record_operation(
            repo_path,
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
        self.ledger.recent_operations(repo_path, limit)
    }

    /// Retrieve the latest pending snapshot if available.
    pub fn get_latest_snapshot(&self) -> Result<Option<SnapshotRecord>, TsugiError> {
        self.ledger.get_latest_snapshot()
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

/// Helper to create a stash commit object using `git stash create`.
fn create_stash_object(repo_path: &Path) -> Result<Option<String>, TsugiError> {
    let output = Command::new("git")
        .args(["stash", "create"])
        .current_dir(repo_path)
        .output()
        .map_err(|e| {
            TsugiError::SnapshotFailed(format!("Failed to create stash object: {e}"))
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();

    if stdout.is_empty() {
        debug!("SafetyVault: working tree clean, no stash needed");
        Ok(None)
    } else {
        debug!("SafetyVault: created stash object: {stdout}");
        Ok(Some(stdout))
    }
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
        repo.commit(Some("HEAD"), &sig, &sig, "Second commit", &tree, &[])
            .unwrap();

        // Verify latest snapshot is found
        let latest = vault.get_latest_snapshot().unwrap();
        assert!(latest.is_some());

        // Roll back latest
        let restored = vault.rollback_latest(tmp.path()).unwrap();
        assert!(restored.is_some());

        // Verify snapshot is now marked rolled back
        let after_rollback = vault.get_latest_snapshot().unwrap();
        assert!(after_rollback.is_none());
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
}
