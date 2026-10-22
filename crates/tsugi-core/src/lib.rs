//! # tsugi-core
//!
//! Enterprise-grade Git state analysis, resolution planning, and safe execution engine.
//!
//! ## Architecture Layers
//!
//! 1. **`probe`** — Deterministic context collector using libgit2. Inspects `.git/` state
//!    directories and produces a compact `RepoState` fact sheet in <10ms.
//!
//! 2. **`tier0`** — Fast-path pattern matcher for the 40 most common Git error states.
//!    Resolves canonical issues deterministically in <2ms without invoking the LLM.
//!
//! 3. **`engine`** — LLM inference bridge. Pluggable `InferenceBackend` seam wrapping
//!    process-based `llama-cli` or in-memory test mocks.
//!
//! 4. **`safety_vault` & `guardrail`** — Deterministic policy engine and shadow snapshot
//!    safety vault combining git refs and SQLite audit logs.
//!
//! 5. **`coordinator`** — The deep `TsugiEngine` module uniting probing, planning,
//!    safety, and execution behind a minimal, highly-leveraged interface.

pub mod types;
pub mod probe;
pub mod tier0;
pub mod engine;
pub mod guardrail;
pub mod ledger;
pub mod safety_vault;
pub mod config;
pub mod executor;
pub mod formatter;
pub mod coordinator;

// Re-export primary deep types and seams
pub use coordinator::{PlanOutcome, ResolutionOutcome, TsugiEngine};
pub use safety_vault::SafetyVault;
pub use executor::{ConfirmationResult, ExecutionObserver, TerminalObserver, CapturingObserver};
pub use engine::{InferenceBackend, LlamaCliBackend, MockInferenceBackend};
