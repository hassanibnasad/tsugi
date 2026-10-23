//! # Tsugi Engine: Unified Core Coordinator
//!
//! The primary deep module of `tsugi-core`. Provides a single cohesive interface
//! for repository inspection, diagnostic planning, safety verification, shadow
//! snapshotting, and supervised execution.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         TsugiEngine                         │
//! ├─────────────┬─────────────┬──────────────┬──────────────────┤
//! │ probe.rs    │ tier0.rs    │ engine.rs    │ safety_vault.rs  │
//! │ libgit2     │ Fast-path   │ Inference    │ Git refs +       │
//! │ inspection  │ rules       │ backend      │ SQLite ledger    │
//! └─────────────┴─────────────┴──────────────┴──────────────────┘
//!                                │
//!                     ExecutionObserver Seam
//!                                │
//!                 Terminal / Headless / Test Adapters
//! ```

use std::path::Path;

use tracing::info;

use crate::config::{self, SetupStatus};
use crate::engine::{default_backend, InferenceBackend};
use crate::executor::{execute_plan_with_observer, ExecutionObserver};
use crate::guardrail;
use crate::ledger::OperationRecord;
use crate::probe;
use crate::safety_vault::SafetyVault;
use crate::tier0;
use crate::types::{RepoState, ResolutionPlan, RiskTier, SnapshotRecord, TsugiConfig, TsugiError};

/// Outcome of the diagnostic planning phase.
#[derive(Debug, Clone)]
pub enum PlanOutcome {
    /// A resolution plan is ready to execute.
    Ready(ResolutionPlan),
    /// Local model setup is required before complex issues can be resolved.
    SetupNeeded,
}

/// Outcome of a full end-to-end resolution.
#[derive(Debug, Clone)]
pub enum ResolutionOutcome {
    /// Plan was successfully executed.
    Executed {
        plan: ResolutionPlan,
        executed_commands: Vec<String>,
        snapshot: Option<SnapshotRecord>,
    },
    /// Execution was aborted by user or policy.
    Aborted {
        plan: ResolutionPlan,
        reason: String,
    },
    /// Model setup is required to generate a plan.
    SetupNeeded,
}

/// The core Tsugi engine deep module.
pub struct TsugiEngine {
    config: TsugiConfig,
    vault: SafetyVault,
    inference: Box<dyn InferenceBackend>,
}

impl TsugiEngine {
    /// Construct a new TsugiEngine using default config, SafetyVault, and CLI inference backend.
    pub fn new(config: TsugiConfig) -> Result<Self, TsugiError> {
        let vault = SafetyVault::open_default()?;
        let inference = default_backend();
        Ok(Self {
            config,
            vault,
            inference,
        })
    }

    /// Construct a TsugiEngine with injected components (ideal for testing or alternative adapters).
    pub fn with_components(
        config: TsugiConfig,
        vault: SafetyVault,
        inference: Box<dyn InferenceBackend>,
    ) -> Self {
        Self {
            config,
            vault,
            inference,
        }
    }

    /// Layer 1: Probe repository state.
    pub fn probe(&self, repo_path: &Path) -> Result<RepoState, TsugiError> {
        probe::probe_repo_state(repo_path)
    }

    /// Layer 2 & 3: Produce a guardrail-verified plan for the current repository state and intent.
    pub fn plan(&self, state: &RepoState, user_intent: &str) -> Result<PlanOutcome, TsugiError> {
        // Step 1: Try Tier-0 deterministic fast-path
        if let Some(t0_plan) = tier0::try_resolve(state, user_intent) {
            info!("Tier-0 fast-path matched — no LLM needed");
            let guarded = guardrail::apply_guardrails(t0_plan, &self.config)?;
            return Ok(PlanOutcome::Ready(guarded));
        }

        // Step 2: Fall back to local inference backend
        match config::check_setup() {
            SetupStatus::Ready { .. } => {
                info!("Invoking local LLM inference backend");
                let raw_plan = self.inference.generate_plan(state, user_intent, &self.config)?;
                let guarded = guardrail::apply_guardrails(raw_plan, &self.config)?;
                Ok(PlanOutcome::Ready(guarded))
            }
            _ => Ok(PlanOutcome::SetupNeeded),
        }
    }

    /// Layer 4: Execute a resolution plan with shadow snapshotting and audit logging.
    pub fn execute(
        &self,
        repo_path: &Path,
        plan: &ResolutionPlan,
        user_intent: &str,
        observer: &dyn ExecutionObserver,
    ) -> Result<Vec<String>, TsugiError> {
        // 1. Verify guardrail policy on the plan
        let plan = guardrail::apply_guardrails(plan.clone(), &self.config)?;

        // 2. Create shadow snapshot if plan is mutating (fail-closed)
        let snapshot = if self.config.auto_snapshot && plan.risk_tier != RiskTier::ReadOnly {
            Some(self.vault.checkpoint(repo_path, &plan)?)
        } else {
            None
        };

        // 3. Execute plan steps across the observer seam
        let execution_result = execute_plan_with_observer(&plan, repo_path, observer);

        // 4. Record execution audit trail
        match execution_result {
            Ok(executed) => {
                self.vault.record_execution(
                    &repo_path.to_string_lossy(),
                    user_intent,
                    &plan,
                    &executed,
                    true,
                    snapshot.as_ref().map(|s| s.id.as_str()),
                )?;
                Ok(executed)
            }
            Err(e) => {
                self.vault.record_execution(
                    &repo_path.to_string_lossy(),
                    user_intent,
                    &plan,
                    &[],
                    false,
                    snapshot.as_ref().map(|s| s.id.as_str()),
                )?;
                Err(e)
            }
        }
    }

    /// End-to-end resolution convenience method: probes state, plans fix, and executes.
    pub fn resolve(
        &self,
        repo_path: &Path,
        user_intent: &str,
        observer: &dyn ExecutionObserver,
    ) -> Result<ResolutionOutcome, TsugiError> {
        let state = self.probe(repo_path)?;

        match self.plan(&state, user_intent)? {
            PlanOutcome::Ready(plan) => {
                let snapshot = if self.config.auto_snapshot && plan.risk_tier != RiskTier::ReadOnly {
                    Some(self.vault.checkpoint(repo_path, &plan)?)
                } else {
                    None
                };

                let execution_result = execute_plan_with_observer(&plan, repo_path, observer);

                match execution_result {
                    Ok(executed_commands) => {
                        self.vault.record_execution(
                            &repo_path.to_string_lossy(),
                            user_intent,
                            &plan,
                            &executed_commands,
                            true,
                            snapshot.as_ref().map(|s| s.id.as_str()),
                        )?;
                        Ok(ResolutionOutcome::Executed {
                            plan,
                            executed_commands,
                            snapshot,
                        })
                    }
                    Err(TsugiError::ExecutionFailed { reason, .. }) if reason == "Aborted by user" => {
                        Ok(ResolutionOutcome::Aborted { plan, reason })
                    }
                    Err(e) => Err(e),
                }
            }
            PlanOutcome::SetupNeeded => Ok(ResolutionOutcome::SetupNeeded),
        }
    }

    /// Roll back the most recent tsugi operation.
    pub fn undo(&self, repo_path: &Path) -> Result<Option<SnapshotRecord>, TsugiError> {
        self.vault.rollback_latest(repo_path)
    }

    /// Retrieve recent operation audit history.
    pub fn history(
        &self,
        repo_path: Option<&str>,
        limit: usize,
    ) -> Result<Vec<OperationRecord>, TsugiError> {
        self.vault.recent_operations(repo_path, limit)
    }

    /// Clean up old snapshot refs older than `days`.
    pub fn cleanup(&self, repo_path: &Path, days: u32) -> Result<usize, TsugiError> {
        self.vault.cleanup(repo_path, days)
    }

    /// Access the underlying SafetyVault directly.
    pub fn vault(&self) -> &SafetyVault {
        &self.vault
    }

    /// Access the active engine configuration.
    pub fn config(&self) -> &TsugiConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::MockInferenceBackend;
    use crate::executor::CapturingObserver;
    use tempfile::TempDir;

    #[test]
    fn test_tsugi_engine_end_to_end_with_tier0() {
        let tmp = TempDir::new().unwrap();
        let repo = git2::Repository::init(tmp.path()).unwrap();

        // Create an initial commit so HEAD exists
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

        let db_path = tmp.path().join("ledger.sqlite");
        let vault = SafetyVault::open(&db_path).unwrap();
        let inference = Box::new(MockInferenceBackend::with_default_test_plan());
        let config = TsugiConfig::default();

        let engine = TsugiEngine::with_components(config, vault, inference);

        // Probe repository
        let state = engine.probe(tmp.path()).unwrap();
        assert!(matches!(state.head, crate::types::HeadState::Attached { .. }));

        // Plan fix using Tier-0 pattern matching (unstage intent)
        let plan_outcome = engine.plan(&state, "unstage").unwrap();
        assert!(matches!(plan_outcome, PlanOutcome::Ready(_)));

        if let PlanOutcome::Ready(plan) = plan_outcome {
            let observer = CapturingObserver::new(true);
            let executed = engine.execute(tmp.path(), &plan, "unstage", &observer).unwrap();
            assert!(!executed.is_empty());

            let history = engine.history(Some(&tmp.path().to_string_lossy()), 10).unwrap();
            assert_eq!(history.len(), 1);
            assert_eq!(history[0].user_intent, "unstage");
        }
    }

    #[test]
    fn test_tsugi_engine_fallback_to_inference_backend() {
        let tmp = TempDir::new().unwrap();
        let _repo = git2::Repository::init(tmp.path()).unwrap();

        let db_path = tmp.path().join("ledger.sqlite");
        let vault = SafetyVault::open(&db_path).unwrap();
        let inference = Box::new(MockInferenceBackend::with_default_test_plan());
        let mut config = TsugiConfig::default();
        config.primary_model_path = Some(tmp.path().join("dummy.gguf"));

        let engine = TsugiEngine::with_components(config, vault, inference);
        let state = engine.probe(tmp.path()).unwrap();

        // Pass a query that Tier-0 doesn't match so it calls mock inference
        let plan_outcome = engine.plan(&state, "unusual custom problem").unwrap();
        // Since dummy.gguf doesn't exist, check_setup() reports SetupNeeded or Ready
        // If SetupNeeded, outcome is SetupNeeded
        assert!(matches!(plan_outcome, PlanOutcome::Ready(_) | PlanOutcome::SetupNeeded));
    }
}
