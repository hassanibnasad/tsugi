# Domain Context: Tsugi (次)

This document defines the ubiquitous language and domain model for the Tsugi Git assistant.

## Core Domain Concepts

### `RepoState`
Deterministic fact sheet capturing the repository's topology (HEAD state, branch tracking divergence, in-progress operations such as rebase/merge, conflict entries, stash count, and working tree summaries). Extracted via `libgit2` in <10ms.

### `ResolutionPlan`
The structured decision output produced either by deterministic pattern-matching (Tier-0) or grammar-constrained LLM inference. Contains a root cause diagnosis, a risk tier (`ReadOnly`, `MutatingSafe`, `MutatingDangerous`, `HistoryRewrite`), an ordered sequence of `ExecutionStep` commands, and an inverse rollback command.

### `TsugiEngine`
The primary deep module of `tsugi-core`. It presents a cohesive interface to callers (`diagnose`, `resolve`, `undo`, `history`) and encapsulates the internal pipeline: probing repo state, consulting Tier-0 or LLM planners, running guardrail policy verification, creating pre-mutation snapshots in `SafetyVault`, and executing verified commands through an `ExecutionObserver`.

### `SafetyVault`
The unified safety and audit manager. Encapsulates two underlying persistence mechanisms:
1. **Git Shadow Refs**: Plumbing snapshot commits created via `git stash create` and pinned under `refs/tsugi/snapshots/` to prevent garbage collection without altering user worktrees or refs.
2. **Audit Ledger**: A local SQLite database recording every plan execution, executed command list, timestamp, and rollback status.
Callers interact solely with `SafetyVault` (`checkpoint`, `rollback_latest`, `recent_operations`) and never touch the raw database connection or ref plumbing.

### `ExecutionObserver`
The interaction seam separating command execution from presentation and user I/O. Defines lifecycle callbacks:
- `on_step_start(step_num, total, step)`
- `on_step_output(line)`
- `request_confirmation(step) -> bool`
- `on_step_complete(step_num, step, success)`

Adapters:
- `TerminalObserver`: Interactive CLI adapter with ANSI formatting and stdin confirmation prompts.
- `CapturingObserver`: In-memory test and headless adapter that records steps and auto-confirms or aborts deterministically.

### `InferenceBackend`
The model execution seam separating resolution reasoning from the physical execution vehicle.
Adapters:
- `LlamaCliAdapter`: Invokes the local `llama-cli` process with GBNF grammar constraints.
- `MockInferenceAdapter`: Returns predetermined plans for offline, hermetic unit/integration tests without requiring model weights.

### `ReplSession`
The conversational co-pilot loop maintaining an in-memory rolling session context window (<2MB RAM) across iterative user turns, enabling interactive clarification and multi-step repair workflows.

### `PrCheck`
Deterministic pre-PR audit engine evaluating branch readiness: Conventional Commits compliance, WIP commit detection, secret leakage scanning, and linear history validation against upstream target bases.

### `RecoveryArchaeology`
Dangling object and reflog scanner locating unreferenced, reset, or orphaned commits without relying on cloud backups or external tooling.
