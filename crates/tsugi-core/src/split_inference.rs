//! # SLM-Powered Semantic Commit Splitter
//!
//! Uses local GGUF model inference constrained by GBNF grammar (`tsugi_split.gbnf`)
//! to cluster complex, intermingled working tree changes into logical Conventional Commits.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{debug, info};

use crate::config;
use crate::engine::find_llama_cli;
use crate::types::{ChangedFile, SplitCommitPlan, TsugiConfig, TsugiError};

const TSUGI_SPLIT_GBNF: &str = include_str!("../../../grammars/tsugi_split.gbnf");

/// Default path for the split GBNF grammar file.
pub fn default_split_grammar_path() -> PathBuf {
    config::config_dir().join("tsugi_split.gbnf")
}

/// Ensure the split GBNF grammar file exists on disk.
pub fn ensure_split_grammar_file(path: &Path) -> Result<(), TsugiError> {
    if !path.exists() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, TSUGI_SPLIT_GBNF).map_err(|e| {
            TsugiError::InferenceError(format!(
                "Failed to write split grammar file at {}: {e}",
                path.display()
            ))
        })?;
    }
    Ok(())
}

/// Cluster changed files using the local SLM with GBNF grammar constraints.
pub fn split_with_llm(
    files: &[ChangedFile],
    config: &TsugiConfig,
) -> Result<SplitCommitPlan, TsugiError> {
    let model_path = config
        .primary_model_path
        .as_ref()
        .or(config.fallback_model_path.as_ref())
        .ok_or_else(|| TsugiError::ConfigError("No local model configured for inference".into()))?;

    if !model_path.exists() {
        return Err(TsugiError::ModelNotFound {
            path: model_path.clone(),
        });
    }

    let grammar_path = default_split_grammar_path();
    ensure_split_grammar_file(&grammar_path)?;

    let prompt = build_split_prompt(files);
    debug!("Split prompt:\n{}", prompt);

    let raw_output = invoke_llama_cli_split(
        model_path,
        &prompt,
        &grammar_path,
        config.max_generation_tokens.max(512),
        config.n_threads,
    )?;

    debug!("Raw split output: {}", raw_output);

    let mut plan = parse_split_json(&raw_output)?;

    // Validate that all original files are accounted for
    let all_original_paths: HashSet<String> = files.iter().map(|f| f.path.clone()).collect();
    let mut assigned_paths: HashSet<String> = HashSet::new();

    for commit in &mut plan.commits {
        commit.files.retain(|f| all_original_paths.contains(f));
        for f in &commit.files {
            assigned_paths.insert(f.clone());
        }
    }

    // Remove any empty commits
    plan.commits.retain(|c| !c.files.is_empty());

    // Record any unassigned files
    plan.unassigned_files = all_original_paths
        .difference(&assigned_paths)
        .cloned()
        .collect();

    info!(
        "Split inference complete: {} commits produced, {} unassigned files",
        plan.commits.len(),
        plan.unassigned_files.len()
    );

    Ok(plan)
}

/// Construct prompt for the LLM to cluster changed files.
pub fn build_split_prompt(files: &[ChangedFile]) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str("You are tsugi (次), an expert Git assistant. Your task is to inspect uncommitted changes and cluster them into atomic, cohesive Conventional Commits (feat, fix, docs, test, refactor, chore, style, perf, ci).\n");
    prompt.push_str("Output strictly conforms to the JSON schema with summary and commits array.\n");
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str("Cluster these changed files into logical, atomic commits:\n\n");

    for f in files {
        prompt.push_str(&format!(
            "- file: \"{}\" ({}, +{}, -{})\n",
            f.path, f.status, f.additions, f.deletions
        ));
        if !f.diff_preview.is_empty() {
            prompt.push_str("  diff preview:\n");
            for line in f.diff_preview.lines().take(4) {
                prompt.push_str(&format!("    {line}\n"));
            }
        }
    }

    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt
}

/// Parse raw JSON from LLM into a `SplitCommitPlan`.
pub fn parse_split_json(raw: &str) -> Result<SplitCommitPlan, TsugiError> {
    let trimmed = raw.trim();
    let json_str = if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        &trimmed[start..=end]
    } else {
        trimmed
    };

    serde_json::from_str(json_str).map_err(|e| {
        TsugiError::InferenceError(format!(
            "Failed to parse split plan JSON: {e}\nRaw output: {raw}"
        ))
    })
}

/// Invoke `llama-cli` with split GBNF grammar constraints.
fn invoke_llama_cli_split(
    model_path: &Path,
    prompt: &str,
    grammar_path: &Path,
    max_tokens: usize,
    n_threads: usize,
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
        .arg("2048")
        .arg("--temp")
        .arg("0.1")
        .arg("--repeat-penalty")
        .arg("1.1")
        .arg("--no-display-prompt")
        .arg("--log-disable")
        .arg("--grammar-file")
        .arg(grammar_path);

    cmd.stderr(std::process::Stdio::null());

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FileDeltaStatus;

    #[test]
    fn test_parse_split_json_clean() {
        let raw = r#"{
            "summary": "Split into 2 commits",
            "commits": [
                {
                    "commit_type": "feat",
                    "scope": "core",
                    "message": "feat(core): add snapshot support",
                    "explanation": "Core engine modifications",
                    "files": ["crates/tsugi-core/src/safety_vault.rs"]
                },
                {
                    "commit_type": "docs",
                    "scope": null,
                    "message": "docs: update readme",
                    "explanation": "Documentation updates",
                    "files": ["README.md"]
                }
            ]
        }"#;

        let plan = parse_split_json(raw).expect("should parse");
        assert_eq!(plan.commits.len(), 2);
        assert_eq!(plan.commits[0].commit_type, "feat");
        assert_eq!(plan.commits[0].scope.as_deref(), Some("core"));
        assert_eq!(plan.commits[1].commit_type, "docs");
    }

    #[test]
    fn test_build_split_prompt() {
        let files = vec![ChangedFile {
            path: "src/lib.rs".to_string(),
            status: FileDeltaStatus::Modified,
            additions: 5,
            deletions: 1,
            diff_preview: "+fn new()".to_string(),
        }];

        let prompt = build_split_prompt(&files);
        assert!(prompt.contains("src/lib.rs"));
        assert!(prompt.contains("+fn new()"));
    }
}
