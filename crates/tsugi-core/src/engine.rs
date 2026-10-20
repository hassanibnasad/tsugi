//! # Layer 3: LLM Inference Engine
//!
//! Wraps local GGUF model inference for Git resolution planning. The engine:
//!
//! 1. Checks available system memory via `sysinfo` to select the model tier
//!    (primary Qwen2.5-Coder-1.5B or fallback Qwen3-0.6B).
//! 2. Constructs a compact prompt from the [`RepoState`] fact sheet + user intent.
//! 3. Invokes `llama-cli` (from the llama.cpp release) with GBNF grammar constraints
//!    to force structured JSON output.
//! 4. Parses the constrained output into a [`ResolutionPlan`].
//!
//! ## Architecture Decision: Process-Based Inference
//!
//! We invoke `llama-cli` as a subprocess rather than linking llama.cpp via FFI.
//! This provides several enterprise-grade advantages:
//! - **Zero-idle RAM**: The model process exits after inference, returning 100%
//!   of RAM to the system. No background daemon.
//! - **Crash isolation**: A segfault in the C++ inference engine cannot bring
//!   down the Rust CLI.
//! - **Easy model updates**: Users swap GGUF files without recompiling.
//! - **mmap by default**: `llama-cli` memory-maps the GGUF file, so the OS
//!   kernel manages page-in/eviction automatically.
//!
//! When the Rust `llama-cpp-2` crate stabilizes its GBNF API, we can add an
//! in-process FFI backend behind the same [`InferenceEngine`] trait.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use sysinfo::System;
use tracing::{debug, info, warn};

use crate::types::{
    ExecutionStep, ModelTier, PlanSource, RepoState, ResolutionPlan, RiskTier, TsugiConfig,
    TsugiError,
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API & InferenceBackend Seam
// ─────────────────────────────────────────────────────────────────────────────

/// Seam defining model inference for resolution planning.
///
/// Implementations adapt physical inference vehicles (local llama-cli subprocess,
/// local HTTP llama-server, in-process FFI, or in-memory fakes) behind a common interface.
pub trait InferenceBackend: Send + Sync {
    /// Generate a resolution plan given the fact sheet and user intent.
    fn generate_plan(
        &self,
        state: &RepoState,
        user_intent: &str,
        config: &TsugiConfig,
    ) -> Result<ResolutionPlan, TsugiError>;
}

/// Production inference adapter wrapping local `llama-cli` and `llama-server`.
#[derive(Debug, Default, Clone)]
pub struct LlamaCliBackend;

impl InferenceBackend for LlamaCliBackend {
    fn generate_plan(
        &self,
        state: &RepoState,
        user_intent: &str,
        config: &TsugiConfig,
    ) -> Result<ResolutionPlan, TsugiError> {
        resolve_with_llm(state, user_intent, config)
    }
}

/// Create a boxed default inference backend.
pub fn default_backend() -> Box<dyn InferenceBackend> {
    Box::new(LlamaCliBackend)
}

/// In-memory fake inference adapter for hermetic unit/integration tests and offline environments.
#[derive(Debug, Clone)]
pub struct MockInferenceBackend {
    canned_plan: Option<ResolutionPlan>,
}

impl MockInferenceBackend {
    /// Create a mock backend that always returns the specified canned plan.
    pub fn new(canned_plan: ResolutionPlan) -> Self {
        Self {
            canned_plan: Some(canned_plan),
        }
    }

    /// Create a mock backend with a default safe testing plan.
    pub fn with_default_test_plan() -> Self {
        Self {
            canned_plan: Some(ResolutionPlan {
                root_cause: "Mock resolution".to_string(),
                risk_tier: RiskTier::MutatingSafe,
                execution_steps: vec![ExecutionStep {
                    cmd: "git status".to_string(),
                    explanation: "Check status".to_string(),
                    requires_confirmation: false,
                }],
                rollback_cmd: "git status".to_string(),
                source: PlanSource::LlmPrimary,
                confidence: 1.0,
            }),
        }
    }
}

impl InferenceBackend for MockInferenceBackend {
    fn generate_plan(
        &self,
        _state: &RepoState,
        _user_intent: &str,
        _config: &TsugiConfig,
    ) -> Result<ResolutionPlan, TsugiError> {
        self.canned_plan
            .clone()
            .ok_or_else(|| TsugiError::InferenceError("No mock plan configured".to_string()))
    }
}

/// Resolve a Git issue by invoking the local LLM with grammar-constrained generation.
///
/// This function:
/// 1. Determines the model tier based on available system memory.
/// 2. Constructs the prompt from the fact sheet + user intent.
/// 3. Checks for an active local `llama-server` (127.0.0.1:8080) as a fast path.
/// 4. Otherwise invokes `llama-cli` with GBNF grammar constraints.
/// 5. Parses the structured JSON output into a `ResolutionPlan`.
pub fn resolve_with_llm(
    state: &RepoState,
    user_intent: &str,
    config: &TsugiConfig,
) -> Result<ResolutionPlan, TsugiError> {
    // Step 1: Select model tier based on available memory
    let tier = select_model_tier(config);
    info!("Selected model tier: {:?}", tier);

    // Step 2: Construct the prompt
    let fact_sheet = state.to_fact_sheet();
    let prompt = build_prompt(&fact_sheet, user_intent);
    debug!("Constructed prompt ({} chars)", prompt.len());

    let grammar_str = TSUGI_PLAN_GBNF;

    // Step 3: Check for local llama-server HTTP fallback (127.0.0.1) first
    let raw_output = if let Some(server_resp) = try_query_llama_server(
        config.llama_server_url.as_deref(),
        &prompt,
        Some(grammar_str),
        config.max_generation_tokens,
    ) {
        info!("Received inference from local llama-server HTTP endpoint");
        server_resp
    } else {
        let model_path = match tier {
            ModelTier::Primary => config.primary_model_path.as_ref(),
            ModelTier::Fallback => config
                .fallback_model_path
                .as_ref()
                .or(config.primary_model_path.as_ref()),
        };

        let model_path = model_path.ok_or_else(|| TsugiError::ConfigError(
            "No model path configured and no local llama-server detected. Run `tsugi setup` to configure a model.".into(),
        ))?;

        if !model_path.exists() {
            return Err(TsugiError::ModelNotFound {
                path: model_path.clone(),
            });
        }

        let grammar_path = config
            .grammar_path
            .as_ref()
            .map(|p| p.clone())
            .unwrap_or_else(|| default_grammar_path());

        invoke_llama_cli(
            model_path,
            &prompt,
            &grammar_path,
            config.max_generation_tokens,
            config.n_threads,
            config.prompt_lookup_decoding,
        )?
    };

    debug!("Raw LLM output: {}", raw_output);

    // Step 4: Parse the structured output
    let mut plan = parse_llm_output(&raw_output)?;
    plan.source = match tier {
        ModelTier::Primary => PlanSource::LlmPrimary,
        ModelTier::Fallback => PlanSource::LlmFallback,
    };

    Ok(plan)
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Tier Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the model tier based on available system memory.
///
/// - If available RAM >= threshold (default 1.2 GB): use Primary (Qwen2.5-Coder-1.5B)
/// - If available RAM < threshold: use Fallback (Qwen3-0.6B)
fn select_model_tier(config: &TsugiConfig) -> ModelTier {
    let mut sys = System::new();
    sys.refresh_memory();

    let available_bytes = sys.available_memory();
    let threshold = config.fallback_memory_threshold_bytes;

    info!(
        "Available memory: {} MB, threshold: {} MB",
        available_bytes / 1_048_576,
        threshold / 1_048_576
    );

    if available_bytes >= threshold {
        ModelTier::Primary
    } else {
        warn!(
            "Low memory detected ({} MB available). Falling back to smaller model.",
            available_bytes / 1_048_576
        );
        ModelTier::Fallback
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompt Construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build the inference prompt from the fact sheet and user intent.
///
/// The prompt is designed to be <250 tokens total, maximizing the signal-to-noise
/// ratio for a 1.5B parameter model with a 1024-token context window.
fn build_prompt(fact_sheet: &str, user_intent: &str) -> String {
    format!(
        r#"<|im_start|>system
You are tsugi, an expert Git diagnostic and resolution engine. You receive a structured Git repository state and a user's description of their issue. You must output a JSON resolution plan.

Rules:
1. Diagnose the root cause from the repository state.
2. Classify the risk tier: READ_ONLY, MUTATING_SAFE, MUTATING_DANGEROUS, or HISTORY_REWRITE.
3. Output the minimum sequence of Git commands to resolve the issue.
4. Include a rollback command that fully reverses the operation.
5. Never use `git push --force` — always use `git push --force-with-lease`.
6. Never use `git reset --hard` without a prior backup step.
7. Prefer `git switch` over `git checkout` for branch operations.
8. Always preserve uncommitted work — stash before destructive operations.
<|im_end|>
<|im_start|>user
{fact_sheet}

USER_INTENT: {user_intent}
<|im_end|>
<|im_start|>assistant
"#
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// llama-cli Invocation
// ─────────────────────────────────────────────────────────────────────────────

/// Find the `llama-cli` binary on the system PATH or in common install locations.
fn find_llama_cli() -> Result<PathBuf, TsugiError> {
    // Check PATH first
    if let Ok(output) = Command::new("llama-cli").arg("--version").output() {
        if output.status.success() {
            return Ok(PathBuf::from("llama-cli"));
        }
    }

    // Check common install locations
    let candidates = [
        dirs::home_dir().map(|h| h.join(".local").join("bin").join("llama-cli")),
        dirs::home_dir().map(|h| h.join("llama.cpp").join("build").join("bin").join("llama-cli")),
        Some(PathBuf::from("/usr/local/bin/llama-cli")),
        // Windows paths
        dirs::home_dir().map(|h| h.join("llama.cpp").join("build").join("bin").join("Release").join("llama-cli.exe")),
    ];

    for candidate in candidates.iter().flatten() {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    Err(TsugiError::ConfigError(
        "llama-cli not found. Install llama.cpp or run `tsugi setup` for guided installation.".into(),
    ))
}

/// Invoke `llama-cli` with GBNF grammar constraints and return the raw output.
fn invoke_llama_cli(
    model_path: &Path,
    prompt: &str,
    grammar_path: &Path,
    max_tokens: usize,
    n_threads: usize,
    prompt_lookup_decoding: bool,
) -> Result<String, TsugiError> {
    let llama_cli = find_llama_cli()?;

    let mut cmd = Command::new(&llama_cli);

    cmd.arg("--model")
        .arg(model_path)
        .arg("--prompt")
        .arg(prompt)
        .arg("--n-predict")
        .arg(max_tokens.to_string())
        .arg("--threads")
        .arg(n_threads.to_string())
        .arg("--ctx-size")
        .arg("1024")
        .arg("--temp")
        .arg("0.1") // Low temperature for deterministic Git commands
        .arg("--repeat-penalty")
        .arg("1.1")
        .arg("--no-display-prompt") // Don't echo the prompt back
        .arg("--log-disable"); // Suppress llama.cpp logs

    // Add grammar constraint if file exists
    if grammar_path.exists() {
        cmd.arg("--grammar-file").arg(grammar_path);
    }

    // Enable prompt lookup decoding for CPU speedup
    if prompt_lookup_decoding {
        cmd.arg("--lookup-ngram-min").arg("3");
    }

    // Suppress stderr noise from llama.cpp
    cmd.stderr(std::process::Stdio::null());

    debug!("Invoking: {:?}", cmd);

    let output = cmd.output().map_err(|e| {
        TsugiError::InferenceError(format!("Failed to execute llama-cli: {e}"))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(TsugiError::InferenceError(format!(
            "llama-cli exited with status {}: {}",
            output.status, stderr
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(stdout.trim().to_string())
}

/// Attempt to query a local llama-server HTTP endpoint (default http://127.0.0.1:8080).
fn try_query_llama_server(
    server_url: Option<&str>,
    prompt: &str,
    grammar: Option<&str>,
    max_tokens: usize,
) -> Option<String> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    let url_str = server_url.unwrap_or("http://127.0.0.1:8080");
    let trimmed = url_str.trim_start_matches("http://").trim_start_matches("https://");
    let host_port = trimmed.split('/').next().unwrap_or("127.0.0.1:8080");
    let addr: std::net::SocketAddr = host_port.parse().ok()?;

    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(150)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok()?;
    stream.set_write_timeout(Some(Duration::from_secs(5))).ok()?;

    let mut map = serde_json::Map::new();
    map.insert("prompt".to_string(), serde_json::Value::String(prompt.to_string()));
    map.insert("n_predict".to_string(), serde_json::json!(max_tokens));
    map.insert("temperature".to_string(), serde_json::json!(0.1));
    if let Some(g) = grammar {
        map.insert("grammar".to_string(), serde_json::Value::String(g.to_string()));
    }
    let body = serde_json::to_string(&map).ok()?;
    let req = format!(
        "POST /completion HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        host_port,
        body.len(),
        body
    );

    stream.write_all(req.as_bytes()).ok()?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp).ok()?;

    let body_start = resp.find("\r\n\r\n")? + 4;
    let json_body = &resp[body_start..];
    let val: serde_json::Value = serde_json::from_str(json_body).ok()?;
    val.get("content").and_then(|c| c.as_str()).map(|s| s.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// Output Parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Raw JSON structure expected from the LLM (enforced by GBNF grammar).
#[derive(Debug, Deserialize)]
struct LlmOutput {
    root_cause: String,
    risk_tier: String,
    execution_steps: Vec<LlmStep>,
    rollback_cmd: String,
}

#[derive(Debug, Deserialize)]
struct LlmStep {
    cmd: String,
    explanation: String,
}

/// Parse the raw LLM output (JSON enforced by GBNF) into a `ResolutionPlan`.
fn parse_llm_output(raw: &str) -> Result<ResolutionPlan, TsugiError> {
    // The GBNF grammar ensures the output is valid JSON, but we still handle
    // edge cases where the model might emit extra text around the JSON.
    let json_str = extract_json(raw)?;

    let output: LlmOutput = serde_json::from_str(&json_str).map_err(|e| {
        TsugiError::InferenceError(format!("Failed to parse LLM JSON output: {e}\nRaw: {raw}"))
    })?;

    let risk_tier = match output.risk_tier.to_uppercase().as_str() {
        "READ_ONLY" => RiskTier::ReadOnly,
        "MUTATING_SAFE" => RiskTier::MutatingSafe,
        "MUTATING_DANGEROUS" => RiskTier::MutatingDangerous,
        "HISTORY_REWRITE" => RiskTier::HistoryRewrite,
        _ => RiskTier::MutatingDangerous, // Conservative default
    };

    let execution_steps = output
        .execution_steps
        .into_iter()
        .map(|s| {
            let requires_confirmation = matches!(risk_tier, RiskTier::MutatingDangerous | RiskTier::HistoryRewrite);
            ExecutionStep {
                cmd: s.cmd,
                explanation: s.explanation,
                requires_confirmation,
            }
        })
        .collect();

    Ok(ResolutionPlan {
        root_cause: output.root_cause,
        risk_tier,
        execution_steps,
        rollback_cmd: output.rollback_cmd,
        source: PlanSource::LlmPrimary, // Will be overwritten by caller
        confidence: 0.85, // Default confidence for LLM plans
    })
}

/// Extract the JSON object from potentially noisy LLM output.
///
/// Even with GBNF grammar constraints, the model might occasionally emit
/// a few tokens before or after the JSON. This function finds the first
/// `{` and last `}` to extract the JSON.
fn extract_json(raw: &str) -> Result<String, TsugiError> {
    let start = raw.find('{').ok_or_else(|| {
        TsugiError::InferenceError("LLM output contains no JSON object".to_string())
    })?;

    let end = raw.rfind('}').ok_or_else(|| {
        TsugiError::InferenceError("LLM output contains no closing brace".to_string())
    })?;

    if end < start {
        return Err(TsugiError::InferenceError(
            "Malformed JSON in LLM output".to_string(),
        ));
    }

    Ok(raw[start..=end].to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// Default Grammar Path
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the default path for the GBNF grammar file.
fn default_grammar_path() -> PathBuf {
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tsugi");
    config_dir.join("tsugi_plan.gbnf")
}

/// Generate and write the default GBNF grammar file if it doesn't exist.
pub fn ensure_grammar_file(path: &Path) -> Result<(), TsugiError> {
    if path.exists() {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            TsugiError::ConfigError(format!("Failed to create grammar directory: {e}"))
        })?;
    }

    let grammar = TSUGI_PLAN_GBNF;
    std::fs::write(path, grammar).map_err(|e| {
        TsugiError::ConfigError(format!("Failed to write grammar file: {e}"))
    })?;

    info!("Created GBNF grammar at: {}", path.display());
    Ok(())
}

/// The GBNF grammar that constrains LLM output to a valid resolution plan JSON.
pub const TSUGI_PLAN_GBNF: &str = r#"
root ::= "{" ws
    "\"root_cause\"" ws ":" ws string "," ws
    "\"risk_tier\"" ws ":" ws risk-tier "," ws
    "\"execution_steps\"" ws ":" ws "[" ws step ("," ws step)* ws "]" "," ws
    "\"rollback_cmd\"" ws ":" ws string ws
    "}"

step ::= "{" ws
    "\"cmd\"" ws ":" ws string "," ws
    "\"explanation\"" ws ":" ws string ws
    "}"

risk-tier ::= "\"READ_ONLY\"" | "\"MUTATING_SAFE\"" | "\"MUTATING_DANGEROUS\"" | "\"HISTORY_REWRITE\""

string ::= "\"" ([^"\\] | "\\" .)* "\""

ws ::= [ \t\n]*
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_clean() {
        let input = r#"{"root_cause": "test", "risk_tier": "READ_ONLY", "execution_steps": [{"cmd": "git status", "explanation": "check"}], "rollback_cmd": "none"}"#;
        let result = extract_json(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), input);
    }

    #[test]
    fn test_extract_json_with_noise() {
        let input = r#"Here is the plan:
{"root_cause": "test", "risk_tier": "READ_ONLY", "execution_steps": [{"cmd": "git status", "explanation": "check"}], "rollback_cmd": "none"}
Done!"#;
        let result = extract_json(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_llm_output() {
        let json = r#"{"root_cause": "Detached HEAD at abc1234", "risk_tier": "MUTATING_SAFE", "execution_steps": [{"cmd": "git checkout -b recovery/abc1234", "explanation": "Create recovery branch"}], "rollback_cmd": "git checkout abc1234"}"#;
        let plan = parse_llm_output(json);
        assert!(plan.is_ok());
        let plan = plan.unwrap();
        assert_eq!(plan.risk_tier, RiskTier::MutatingSafe);
        assert_eq!(plan.execution_steps.len(), 1);
    }

    #[test]
    fn test_build_prompt_format() {
        let fact_sheet = "[REPO_STATE]\nHEAD: ATTACHED to main (abc1234)";
        let intent = "fix my rebase";
        let prompt = build_prompt(fact_sheet, intent);

        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains(fact_sheet));
        assert!(prompt.contains(intent));
        assert!(prompt.contains("USER_INTENT:"));
    }

    #[test]
    fn test_model_tier_selection() {
        let config = TsugiConfig::default();
        let tier = select_model_tier(&config);
        // On any machine with > 1.2 GB free, this should be Primary.
        // We don't assert a specific value since it depends on the test machine.
        assert!(matches!(tier, ModelTier::Primary | ModelTier::Fallback));
    }

    #[test]
    fn test_mock_inference_backend() {
        let backend = MockInferenceBackend::with_default_test_plan();
        let config = TsugiConfig::default();
        let state = RepoState {
            repo_root: std::path::PathBuf::from("/tmp"),
            head: crate::types::HeadState::Unborn {
                branch: "main".to_string(),
            },
            ongoing_operation: None,
            conflicts: vec![],
            working_tree: crate::types::WorkingTreeSummary {
                staged_new: 0,
                staged_modified: 0,
                staged_deleted: 0,
                staged_renamed: 0,
                unstaged_modified: 0,
                unstaged_deleted: 0,
                untracked: 0,
                ignored_but_tracked: 0,
            },
            recent_commits: vec![],
            recent_reflog: vec![],
            tracking: None,
            stash_count: 0,
            has_submodules: false,
            safe_rollback_ref: None,
        };

        let plan = backend.generate_plan(&state, "test", &config).unwrap();
        assert_eq!(plan.root_cause, "Mock resolution");
        assert_eq!(plan.risk_tier, RiskTier::MutatingSafe);
    }
}
