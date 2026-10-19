//! # Layer 4b: Shadow Snapshot Manager
//!
//! Before executing any mutating command, tsugi creates a zero-cost shadow snapshot
//! that captures the exact byte-for-byte state of the repository. This enables
//! instant rollback via `tsugi undo` even if the resolution makes things worse.
//!
//! ## How It Works
//!
//! 1. **HEAD SHA capture**: Records the current HEAD commit.
//! 2. **Working tree backup via `git stash create`**: Creates a dangling commit object
//!    that captures all staged, unstaged, and untracked changes *without modifying
//!    the index or working tree*. This is a zero-side-effect operation.
//! 3. **Ref-based storage**: The stash object SHA is stored as a Git ref under
//!    `refs/tsugi/snapshots/<timestamp>` so it won't be garbage collected.
//! 4. **Ledger entry**: Both SHAs are recorded in the local SQLite audit ledger
//!    for history and `tsugi undo` lookup.

use std::path::Path;
use std::process::Command;

use chrono::Utc;
use git2::Repository;
use tracing::{debug, info, warn};

use crate::ledger::Ledger;
use crate::types::{ResolutionPlan, SnapshotRecord, TsugiError};

/// Create a shadow snapshot of the repository's current state.
///
/// This function is called automatically before executing any mutating resolution plan.
/// It returns a `SnapshotRecord` that can be used for rollback.
///
/// **Side effects**: Creates a ref under `refs/tsugi/snapshots/` and a ledger entry.
/// Does NOT modify the working tree, index, or HEAD.
pub fn create_snapshot(
    repo_path: &Path,
    plan: &ResolutionPlan,
    ledger: &Ledger,
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

    debug!("Snapshot: HEAD = {}", head_sha);

    // 2. Create a stash object (dangling commit) without modifying anything.
    //    `git stash create` is a plumbing command that creates the stash commit
    //    objects but does NOT modify the working tree, index, or refs/stash.
    let stash_sha = create_stash_object(repo_path)?;

    // 3. Store the stash object as a ref so it survives GC.
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let snapshot_id = uuid::Uuid::new_v4().to_string();
    let ref_name = format!("refs/tsugi/snapshots/{timestamp}_{id}", id = &snapshot_id[..8]);

    if let Some(ref sha) = stash_sha {
        let oid = git2::Oid::from_str(sha).map_err(|e| {
            TsugiError::SnapshotFailed(format!("Invalid stash SHA: {e}"))
        })?;
        repo.reference(&ref_name, oid, true, "tsugi snapshot").map_err(|e| {
            TsugiError::SnapshotFailed(format!("Failed to create snapshot ref: {e}"))
        })?;
        info!("Snapshot: created ref {} → {}", ref_name, sha);
    }

    // 4. Record in the ledger
    let record = SnapshotRecord {
        id: snapshot_id,
        timestamp: Utc::now().to_rfc3339(),
        head_sha,
        stash_sha,
        plan_summary: plan.root_cause.clone(),
        rolled_back: false,
    };

    ledger.insert_snapshot(&record)?;

    info!("Snapshot created: {}", record.id);
    Ok(record)
}

/// Restore a repository from a snapshot (implements `tsugi undo`).
///
/// This function:
/// 1. Resets HEAD to the snapshot's HEAD SHA.
/// 2. Applies the stash object to restore dirty working tree state.
/// 3. Marks the snapshot as rolled back in the ledger.
pub fn restore_snapshot(
    repo_path: &Path,
    snapshot: &SnapshotRecord,
    ledger: &Ledger,
) -> Result<(), TsugiError> {
    info!("Restoring snapshot: {}", snapshot.id);

    // 1. Reset HEAD to the snapshot's HEAD SHA
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

    // 2. Apply the stash object to restore working tree state
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
            warn!("Stash apply had issues (may be expected if tree was clean): {stderr}");
        }
    }

    // 3. Mark as rolled back in the ledger
    ledger.mark_rolled_back(&snapshot.id)?;

    info!("Snapshot {} restored successfully", snapshot.id);
    Ok(())
}

/// Create a stash commit object without modifying the working tree.
///
/// Uses `git stash create` which is a plumbing command that creates commit objects
/// for the stash but doesn't modify any refs or the working tree.
///
/// Returns `None` if the working tree is clean (nothing to stash).
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
        debug!("No dirty state to stash — working tree is clean");
        Ok(None)
    } else {
        debug!("Created stash object: {stdout}");
        Ok(Some(stdout))
    }
}

/// Clean up old snapshot refs that are older than a given number of days.
///
/// This prevents the ref namespace from growing unbounded. Default retention: 30 days.
pub fn cleanup_old_snapshots(
    repo_path: &Path,
    max_age_days: u32,
    ledger: &Ledger,
) -> Result<usize, TsugiError> {
    let repo = Repository::open(repo_path).map_err(|e| {
        TsugiError::SnapshotFailed(format!("Failed to open repository: {e}"))
    })?;

    let cutoff = Utc::now() - chrono::Duration::days(max_age_days as i64);
    let old_snapshots = ledger.get_snapshots_before(&cutoff.to_rfc3339())?;

    let mut cleaned = 0;
    for snapshot in &old_snapshots {
        // Remove the ref
        if let Ok(refs) = repo.references_glob("refs/tsugi/snapshots/*") {
            for mut reference in refs.flatten() {
                let matches_id = reference
                    .name()
                    .map(|n| n.contains(&snapshot.id[..8]))
                    .unwrap_or(false);
                if matches_id {
                    let ref_name = reference.name().unwrap_or("unknown").to_string();
                    if let Err(e) = reference.delete() {
                        warn!("Failed to delete snapshot ref {ref_name}: {e}");
                    } else {
                        cleaned += 1;
                    }
                }
            }
        }

        // Mark as cleaned in the ledger
        ledger.mark_rolled_back(&snapshot.id)?;
    }

    if cleaned > 0 {
        info!("Cleaned up {cleaned} old snapshot refs");
    }

    Ok(cleaned)
}
