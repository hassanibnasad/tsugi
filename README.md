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
Uses libgit2 to inspect `.git/` state directories and produce a compact `RepoState` fact sheet in <10ms. The LLM **never sees raw terminal output** — it only receives the structured fact sheet.

### Layer 2: Tier-0 Fast-Path Engine (`tier0.rs`)
Pattern-matches the top 40+ canonical Git error states deterministically in <2ms **without invoking the LLM**. Handles ~80% of common issues.

### Layer 3: LLM Inference Engine (`engine.rs`)
Invokes the local GGUF model via `llama-cli` with GBNF grammar constraints for structured JSON output. Automatically selects model tier based on available system memory.

### Layer 4: Guardrail + Snapshot + Ledger (`guardrail.rs`, `snapshot.rs`, `ledger.rs`)
- **Guardrail**: Intercepts destructive commands, rewrites `--force` to `--force-with-lease`, blocks dangerous operations.
- **Snapshot**: Creates zero-cost shadow backups before every mutation for instant `tsugi undo`.
- **Ledger**: SQLite audit trail of every operation for history and DPO training data.

## Usage

```bash
# Diagnose and fix the current Git issue
tsugi fix "I messed up my rebase"
tsugi fix                          # Auto-detect from repo state

# Diagnose without executing (dry run)
tsugi diagnose

# Undo the last tsugi operation
tsugi undo

# View operation history
tsugi history

# Check setup status
tsugi status

# Initial setup wizard
tsugi setup
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
├── grammars/
│   └── tsugi_plan.gbnf           # GBNF grammar for constrained LLM output
├── crates/
│   ├── tsugi-core/               # Core library (all 4 layers)
│   │   └── src/
│   │       ├── lib.rs            # Module declarations
│   │       ├── types.rs          # Shared type definitions
│   │       ├── probe.rs          # Layer 1: Git state probe (libgit2)
│   │       ├── tier0.rs          # Layer 2: Deterministic fast-path
│   │       ├── engine.rs         # Layer 3: LLM inference bridge
│   │       ├── guardrail.rs      # Layer 4a: Safety policy engine
│   │       ├── snapshot.rs       # Layer 4b: Shadow snapshot manager
│   │       ├── ledger.rs         # Layer 4c: SQLite audit ledger
│   │       ├── config.rs         # Configuration management
│   │       ├── executor.rs       # Command executor
│   │       └── formatter.rs      # Terminal output formatter
│   └── tsugi-cli/                # CLI binary
│       └── src/
│           └── main.rs           # Entry point + subcommand routing
└── README.md
```

## License

Apache-2.0
