# Tsugi (次) — Enterprise-Grade Terminal Git Assistant

> A local-first, AI-powered Git diagnostic and resolution engine that runs entirely on your machine.
> No API keys. No cloud. No data leaves your laptop.

## Architecture

Tsugi treats Git as a **deterministic state machine** (a Directed Acyclic Graph) and uses a specialized local model (1.5B parameters) strictly as a reasoning engine over pre-fetched repository state, constrained by grammar-enforced schemas and an automatic rollback layer.

```
┌──────────────────────────────────────────────────────────────────┐
│                        tsugi CLI Binary                          │
├──────────┬──────────┬────────────┬──────────────────────────────┤
│ Layer 1  │ Layer 2  │  Layer 3   │          Layer 4             │
│ Probe    │ Tier-0   │  LLM       │  Guardrail + Snapshot        │
│ (git2)   │ Fast-path│  Engine    │  + Ledger + Executor         │
│ <10ms    │ <2ms     │  ~0.4s     │  Policy + Rollback + Audit   │
└──────────┴──────────┴────────────┴──────────────────────────────┘
```

### Layer 1: Deterministic Context Collector (`probe.rs`)
Uses libgit2 to inspect `.git/` state directories and produce a compact `RepoState` fact sheet in <10ms. Probes HEAD topology, ongoing operations (rebase/merge/cherry-pick/revert/bisect/mailbox), conflicts, tracking divergence, linked worktrees, submodule states, shallow clone status, and large staged binaries. The LLM **never sees raw terminal output** — it only receives the structured fact sheet.

### Layer 2: Tier-0 Fast-Path Engine (`tier0.rs`)
Pattern-matches 80+ canonical Git scenarios deterministically in <2ms **without invoking the LLM**. Handles ~80-90% of common developer issues (detached HEADs, diverged remotes, unpushed tracking, submodules, fast-forwards, stashes, and cleanup) with zero memory overhead.

### Layer 3: LLM Inference Engine (`engine.rs`)
Invokes the local GGUF model via `llama-cli` or local `llama-server` with GBNF grammar constraints for structured JSON output. Supports multi-turn session history in conversational mode and automatically selects model tier based on available system memory.

### Layer 4: Guardrail + Snapshot + Ledger (`guardrail.rs`, `safety_vault.rs`, `ledger.rs`)
- **Guardrail**: Intercepts destructive commands, rewrites `--force` to `--force-with-lease`, prevents deletion of protected branches, and flags risky commands for confirmation.
- **Snapshot**: Creates zero-cost 3-parent shadow backups before every mutation for instant `tsugi undo`.
- **Ledger**: SQLite audit trail of every operation for history, rollback safety, and DPO training data.

## Usage

```bash
# Interactive conversational Git co-pilot (conversational REPL session)
tsugi                                     # Bare invocation launches interactive REPL
tsugi chat                                # Explicit REPL session launch

# Explain any complex Git command or strategy in plain English
tsugi explain "git rebase --onto main feat-a feat-b"
tsugi explain "git push --force-with-lease"

# Pre-PR Readiness Audit (conventional commits, WIP commits, secret leaks, linear history)
tsugi pr-ready                            # Compare against main/master
tsugi pr-ready --base develop             # Compare against specific base branch

# Reflog & Dangling Object Archaeology (resurrect accidentally lost commits)
tsugi recover                             # Find lost commits & print recovery branch commands
tsugi recover --limit 20

# Commit Graph Intelligence Analysis
tsugi log --analyze                       # Audit conventional commits % & WIP detection

# Diagnose and fix the current Git issue (or interactive conflict resolution)
tsugi fix "I messed up my rebase"
tsugi fix                                 # Auto-detect from repo state

# Diagnose without executing (dry run)
tsugi diagnose

# Undo the last tsugi operation (restores pre-fix state)
tsugi undo

# Autonomously split messy working tree changes into atomic, logical commits
tsugi split --dry-run                     # Preview proposed commit clusters
tsugi split                               # Interactive review & progressive commit

# Automated zero-touch setup (downloads llama-cli & verified GGUF into ~/.tsugi/)
tsugi setup

# Check system readiness and model status
tsugi status

# Install fail-safe Git guard hooks (automatic <15ms snapshots before rebase/merge/checkout)
tsugi guard install
tsugi guard status

# Shell failure integration (prints hint when any git command exits non-zero)
tsugi init powershell | Out-String | Invoke-Expression    # PowerShell
eval "$(tsugi init bash)"                                 # Bash
eval "$(tsugi init zsh)"                                  # Zsh

# View operation history
tsugi history
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | Intel Core i3 | Intel Core i5+ |
| Disk | 1 GB free | 2 GB free |
| GPU | Not required | Not required |

The primary model (Qwen2.5-Coder-1.5B, IQ4_XS) uses ~840 MB of RAM during inference and returns 100% of it immediately after. The fallback model (Qwen3-0.6B) uses only ~420 MB.

## Models

| Model | Quantization | RAM | Speed (i3/i5) | Role |
|-------|-------------|-----|---------------|------|
| Qwen2.5-Coder-1.5B-Instruct | IQ4_XS / Q4_K_M | ~840-920 MB | 24-35 tok/s | Primary |
| Qwen3-0.6B | Q5_K_M | ~420 MB | 50-65 tok/s | Low-memory fallback |

## Building

```bash
# Prerequisites: Rust toolchain (rustup.rs)
cargo build --release

# The binary is at target/release/tsugi (or tsugi.exe on Windows)
# Copy to your PATH:
cp target/release/tsugi ~/.local/bin/
```

## Project Structure

```
tsugi/
├── Cargo.toml                    # Workspace root
├── MODEL_CARD.md                 # Model specifications, safety & benchmarks
├── grammars/
│   └── tsugi_plan.gbnf           # GBNF grammar for constrained LLM output
├── crates/
│   ├── tsugi-core/               # Core library (all 4 layers)
│   │   └── src/
│   │       ├── lib.rs            # Module declarations
│   │       ├── types.rs          # Shared type definitions (RepoState, ConflictHunk, SessionTurn)
│   │       ├── probe.rs          # Layer 1: Advanced Git state probe (worktrees, submodules, remotes)
│   │       ├── tier0.rs          # Layer 2: Deterministic fast-path (80+ canonical patterns)
│   │       ├── engine.rs         # Layer 3: LLM inference bridge with conversational context
│   │       ├── guardrail.rs      # Layer 4a: Safety policy engine
│   │       ├── safety_vault.rs   # Layer 4b: Unified 3-parent snapshot & audit ledger
│   │       ├── coordinator.rs    # Deep engine facade
│   │       ├── config.rs         # Configuration & platform asset resolution
│   │       ├── executor.rs       # Command executor & observers
│   │       ├── formatter.rs      # ANSI terminal & conflict diff formatter
│   │       ├── explain.rs        # Static & contextual Git command explainer
│   │       └── log_analysis.rs   # Commit graph intelligence analysis
│   └── tsugi-cli/                # CLI binary
│       └── src/
│           ├── lib.rs            # Clap CLI parser & command routing
│           ├── repl.rs           # Conversational REPL Co-Pilot session
│           ├── pr_check.rs       # Pre-PR readiness auditor
│           ├── recover.rs        # Reflog & fsck dangling object archaeology
│           ├── setup.rs          # Zero-touch automated installer
│           ├── downloader.rs     # Streaming downloader with SHA-256 verification
│           ├── guard.rs          # Git guard hooks (<15ms auto-snapshots)
│           ├── conflict_ui.rs    # Interactive 3-choice conflict resolver
│           ├── split_ui.rs       # Progressive commit review & UI
│           ├── shell.rs          # Shell failure integration hooks
│           └── bin/              # tsugi & git-tsugi entry points
└── README.md
```

## License

Apache-2.0
