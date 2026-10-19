//! # Local SQLite Audit Ledger
//!
//! Every tsugi operation is recorded in a local SQLite database at
//! `~/.tsugi/ledger.sqlite`. This provides:
//!
//! - **Snapshot history**: Every pre-mutation snapshot is logged for `tsugi undo`.
//! - **Operation audit trail**: Full history of what tsugi has done in each repo.
//! - **DPO flywheel data**: Failed resolutions (detected by `tsugi undo` within
//!   60 seconds) become hard-negative training pairs for the next model version.

use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};
use tracing::debug;

use crate::types::{SnapshotRecord, TsugiError};

/// The local audit ledger backed by SQLite.
pub struct Ledger {
    conn: Connection,
}

impl Ledger {
    /// Open or create the ledger database at the given path.
    pub fn open(path: &Path) -> Result<Self, TsugiError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                TsugiError::LedgerError(format!("Failed to create ledger directory: {e}"))
            })?;
        }

        let conn = Connection::open(path).map_err(|e| {
            TsugiError::LedgerError(format!("Failed to open ledger database: {e}"))
        })?;

        // Enable WAL mode for better concurrent read performance
        conn.execute_batch("PRAGMA journal_mode=WAL;").ok();

        // Create tables if they don't exist
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                stash_sha TEXT,
                plan_summary TEXT NOT NULL,
                rolled_back INTEGER NOT NULL DEFAULT 0,
                repo_path TEXT
            );

            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                user_intent TEXT NOT NULL,
                plan_source TEXT NOT NULL,
                risk_tier TEXT NOT NULL,
                commands_executed TEXT NOT NULL,
                success INTEGER NOT NULL,
                undo_triggered INTEGER NOT NULL DEFAULT 0,
                undo_timestamp TEXT,
                snapshot_id TEXT,
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON operations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_operations_repo ON operations(repo_path);",
        )
        .map_err(|e| {
            TsugiError::LedgerError(format!("Failed to create ledger tables: {e}"))
        })?;

        debug!("Ledger opened at: {}", path.display());
        Ok(Self { conn })
    }

    /// Open the ledger at the default location (`~/.tsugi/ledger.sqlite`).
    pub fn open_default() -> Result<Self, TsugiError> {
        let path = default_ledger_path();
        Self::open(&path)
    }

    /// Insert a snapshot record.
    pub fn insert_snapshot(&self, snapshot: &SnapshotRecord) -> Result<(), TsugiError> {
        self.conn
            .execute(
                "INSERT INTO snapshots (id, timestamp, head_sha, stash_sha, plan_summary, rolled_back)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    snapshot.id,
                    snapshot.timestamp,
                    snapshot.head_sha,
                    snapshot.stash_sha,
                    snapshot.plan_summary,
                    snapshot.rolled_back as i32,
                ],
            )
            .map_err(|e| TsugiError::LedgerError(format!("Failed to insert snapshot: {e}")))?;

        Ok(())
    }

    /// Mark a snapshot as rolled back.
    pub fn mark_rolled_back(&self, snapshot_id: &str) -> Result<(), TsugiError> {
        self.conn
            .execute(
                "UPDATE snapshots SET rolled_back = 1 WHERE id = ?1",
                params![snapshot_id],
            )
            .map_err(|e| TsugiError::LedgerError(format!("Failed to mark snapshot: {e}")))?;

        Ok(())
    }

    /// Get the most recent snapshot (for `tsugi undo`).
    pub fn get_latest_snapshot(&self) -> Result<Option<SnapshotRecord>, TsugiError> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, timestamp, head_sha, stash_sha, plan_summary, rolled_back
                 FROM snapshots
                 WHERE rolled_back = 0
                 ORDER BY timestamp DESC
                 LIMIT 1",
            )
            .map_err(|e| TsugiError::LedgerError(format!("Failed to query snapshots: {e}")))?;

        let result = stmt
            .query_row([], |row| {
                Ok(SnapshotRecord {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    head_sha: row.get(2)?,
                    stash_sha: row.get(3)?,
                    plan_summary: row.get(4)?,
                    rolled_back: row.get::<_, i32>(5)? != 0,
                })
            })
            .ok();

        Ok(result)
    }

    /// Get all snapshots before a given timestamp (for cleanup).
    pub fn get_snapshots_before(&self, before: &str) -> Result<Vec<SnapshotRecord>, TsugiError> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, timestamp, head_sha, stash_sha, plan_summary, rolled_back
                 FROM snapshots
                 WHERE timestamp < ?1
                 ORDER BY timestamp ASC",
            )
            .map_err(|e| TsugiError::LedgerError(format!("Failed to query snapshots: {e}")))?;

        let results = stmt
            .query_map(params![before], |row| {
                Ok(SnapshotRecord {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    head_sha: row.get(2)?,
                    stash_sha: row.get(3)?,
                    plan_summary: row.get(4)?,
                    rolled_back: row.get::<_, i32>(5)? != 0,
                })
            })
            .map_err(|e| TsugiError::LedgerError(format!("Failed to iterate snapshots: {e}")))?;

        let mut snapshots = Vec::new();
        for result in results {
            snapshots.push(
                result.map_err(|e| TsugiError::LedgerError(format!("Row read error: {e}")))?,
            );
        }

        Ok(snapshots)
    }

    /// Record a completed operation.
    pub fn record_operation(
        &self,
        repo_path: &str,
        user_intent: &str,
        plan_source: &str,
        risk_tier: &str,
        commands: &[String],
        success: bool,
        snapshot_id: Option<&str>,
    ) -> Result<(), TsugiError> {
        let commands_json =
            serde_json::to_string(commands).unwrap_or_else(|_| "[]".to_string());

        self.conn
            .execute(
                "INSERT INTO operations (timestamp, repo_path, user_intent, plan_source, risk_tier, commands_executed, success, snapshot_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    chrono::Utc::now().to_rfc3339(),
                    repo_path,
                    user_intent,
                    plan_source,
                    risk_tier,
                    commands_json,
                    success as i32,
                    snapshot_id,
                ],
            )
            .map_err(|e| TsugiError::LedgerError(format!("Failed to record operation: {e}")))?;

        Ok(())
    }

    /// Get the count of operations for a given repository.
    pub fn operation_count(&self, repo_path: &str) -> Result<usize, TsugiError> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM operations WHERE repo_path = ?1",
                params![repo_path],
                |row| row.get(0),
            )
            .map_err(|e| TsugiError::LedgerError(format!("Failed to count operations: {e}")))?;

        Ok(count as usize)
    }

    /// Get recent operations for display in `tsugi history`.
    pub fn recent_operations(
        &self,
        repo_path: Option<&str>,
        limit: usize,
    ) -> Result<Vec<OperationRecord>, TsugiError> {
        let (query, param): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(path) = repo_path {
            (
                "SELECT timestamp, repo_path, user_intent, plan_source, risk_tier, commands_executed, success, undo_triggered
                 FROM operations
                 WHERE repo_path = ?1
                 ORDER BY timestamp DESC
                 LIMIT ?2",
                vec![Box::new(path.to_string()), Box::new(limit as i64)],
            )
        } else {
            (
                "SELECT timestamp, repo_path, user_intent, plan_source, risk_tier, commands_executed, success, undo_triggered
                 FROM operations
                 ORDER BY timestamp DESC
                 LIMIT ?1",
                vec![Box::new(limit as i64)],
            )
        };

        let mut stmt = self
            .conn
            .prepare(query)
            .map_err(|e| TsugiError::LedgerError(format!("Failed to prepare query: {e}")))?;

        let param_refs: Vec<&dyn rusqlite::types::ToSql> = param.iter().map(|p| p.as_ref()).collect();

        let results = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok(OperationRecord {
                    timestamp: row.get(0)?,
                    repo_path: row.get(1)?,
                    user_intent: row.get(2)?,
                    plan_source: row.get(3)?,
                    risk_tier: row.get(4)?,
                    commands_executed: row.get(5)?,
                    success: row.get::<_, i32>(6)? != 0,
                    undo_triggered: row.get::<_, i32>(7)? != 0,
                })
            })
            .map_err(|e| TsugiError::LedgerError(format!("Failed to query operations: {e}")))?;

        let mut ops = Vec::new();
        for result in results {
            ops.push(
                result.map_err(|e| TsugiError::LedgerError(format!("Row read error: {e}")))?,
            );
        }

        Ok(ops)
    }
}

/// A record of a completed tsugi operation.
#[derive(Debug, Clone)]
pub struct OperationRecord {
    pub timestamp: String,
    pub repo_path: String,
    pub user_intent: String,
    pub plan_source: String,
    pub risk_tier: String,
    pub commands_executed: String,
    pub success: bool,
    pub undo_triggered: bool,
}

/// Returns the default ledger database path.
pub fn default_ledger_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".tsugi")
        .join("ledger.sqlite")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_ledger_open_and_insert() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("test_ledger.sqlite");
        let ledger = Ledger::open(&db_path).unwrap();

        let snapshot = SnapshotRecord {
            id: "test-id-123".to_string(),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            head_sha: "abc1234".to_string(),
            stash_sha: Some("def5678".to_string()),
            plan_summary: "Test snapshot".to_string(),
            rolled_back: false,
        };

        ledger.insert_snapshot(&snapshot).unwrap();

        let latest = ledger.get_latest_snapshot().unwrap();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().id, "test-id-123");
    }

    #[test]
    fn test_ledger_mark_rolled_back() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("test_ledger.sqlite");
        let ledger = Ledger::open(&db_path).unwrap();

        let snapshot = SnapshotRecord {
            id: "rollback-test".to_string(),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            head_sha: "abc1234".to_string(),
            stash_sha: None,
            plan_summary: "Test".to_string(),
            rolled_back: false,
        };

        ledger.insert_snapshot(&snapshot).unwrap();
        ledger.mark_rolled_back("rollback-test").unwrap();

        // After rollback, get_latest_snapshot should return None (filtered out)
        let latest = ledger.get_latest_snapshot().unwrap();
        assert!(latest.is_none());
    }

    #[test]
    fn test_record_operation() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("test_ledger.sqlite");
        let ledger = Ledger::open(&db_path).unwrap();

        ledger
            .record_operation(
                "/tmp/test-repo",
                "fix rebase",
                "Tier0",
                "MUTATING_SAFE",
                &["git rebase --continue".to_string()],
                true,
                None,
            )
            .unwrap();

        let count = ledger.operation_count("/tmp/test-repo").unwrap();
        assert_eq!(count, 1);
    }
}
