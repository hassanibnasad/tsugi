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
//! 3. **`engine`** — LLM inference bridge. Wraps llama.cpp via FFI for grammar-constrained
//!    generation using the fine-tuned Qwen2.5-Coder-1.5B / Qwen3-0.6B GGUF models.
//!
//! 4. **`guardrail`** — Deterministic policy engine and shadow snapshot manager.
//!    Intercepts destructive commands, creates automatic backups, and provides instant rollback.

pub mod types;
pub mod probe;
pub mod tier0;
pub mod engine;
pub mod guardrail;
pub mod snapshot;
pub mod ledger;
pub mod config;
pub mod executor;
pub mod formatter;
