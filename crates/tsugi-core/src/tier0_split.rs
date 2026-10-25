//! # Tier-0 Fast-Path Commit Splitter
//!
//! Provides deterministic, rule-based partitioning of uncommitted working tree changes
//! into logical commits in <1ms without requiring SLM inference.

use std::path::Path;

use crate::types::{ChangedFile, ProposedCommit, SplitCommitPlan};

/// Attempt to deterministically partition a list of changed files into atomic commits.
/// Returns `Some(SplitCommitPlan)` if the changes can be cleanly segmented, or `None`
/// if the changes are intermingled and require SLM semantic clustering.
pub fn try_partition_changes(files: &[ChangedFile]) -> Option<SplitCommitPlan> {
    if files.is_empty() {
        return None;
    }

    // Single file special-case: deterministic commit message
    if files.len() == 1 {
        let f = &files[0];
        let (commit_type, scope, msg) = categorize_single_file(&f.path);
        return Some(SplitCommitPlan {
            summary: format!("Single file change in {}", f.path),
            commits: vec![ProposedCommit {
                commit_type,
                scope,
                message: msg,
                explanation: format!("Isolated modification to {}", f.path),
                files: vec![f.path.clone()],
            }],
            unassigned_files: Vec::new(),
        });
    }

    let mut doc_files = Vec::new();
    let mut dep_files = Vec::new();
    let mut test_files = Vec::new();
    let mut core_files = Vec::new();
    let mut cli_files = Vec::new();
    let mut training_files = Vec::new();
    let mut grammar_files = Vec::new();
    let mut other_files = Vec::new();

    for f in files {
        let p = f.path.replace('\\', "/");
        if is_doc_file(&p) {
            doc_files.push(f.path.clone());
        } else if is_dep_or_build_file(&p) {
            dep_files.push(f.path.clone());
        } else if is_test_file(&p) {
            test_files.push(f.path.clone());
        } else if p.starts_with("crates/tsugi-core/") {
            core_files.push(f.path.clone());
        } else if p.starts_with("crates/tsugi-cli/") {
            cli_files.push(f.path.clone());
        } else if p.starts_with("training/") {
            training_files.push(f.path.clone());
        } else if p.starts_with("grammars/") {
            grammar_files.push(f.path.clone());
        } else {
            other_files.push(f.path.clone());
        }
    }

    // If there are unclassifiable "other" files, we cannot guarantee clean partitioning
    if !other_files.is_empty() {
        return None;
    }

    let mut commits = Vec::new();

    if !core_files.is_empty() {
        commits.push(ProposedCommit {
            commit_type: "feat".to_string(),
            scope: Some("core".to_string()),
            message: "feat(core): update core engine components".to_string(),
            explanation: format!("Core engine modifications across {} files", core_files.len()),
            files: core_files,
        });
    }

    if !cli_files.is_empty() {
        commits.push(ProposedCommit {
            commit_type: "feat".to_string(),
            scope: Some("cli".to_string()),
            message: "feat(cli): update CLI commands and user interfaces".to_string(),
            explanation: format!("CLI interface updates across {} files", cli_files.len()),
            files: cli_files,
        });
    }

    if !grammar_files.is_empty() {
        commits.push(ProposedCommit {
            commit_type: "feat".to_string(),
            scope: Some("grammar".to_string()),
            message: "feat(grammar): update inference GBNF grammars".to_string(),
            explanation: format!("Grammar updates for structured inference ({} files)", grammar_files.len()),
            files: grammar_files,
        });
    }

    if !training_files.is_empty() {
        commits.push(ProposedCommit {
            commit_type: "feat".to_string(),
            scope: Some("training".to_string()),
            message: "feat(training): update training pipeline and evaluation benchmarks".to_string(),
            explanation: format!("Training and evaluation assets ({} files)", training_files.len()),
            files: training_files,
        });
    }

    if !dep_files.is_empty() {
        commits.push(ProposedCommit {
            commit_type: "chore".to_string(),
            scope: Some("deps".to_string()),
            message: "chore(deps): update project dependencies and build configuration".to_string(),
            explanation: format!("Dependency and manifest changes across {} files", dep_files.len()),
            files: dep_files,
        });
    }

    if !test_files.is_empty() {
        commits.push(ProposedCommit {
            commit_type: "test".to_string(),
            scope: None,
            message: "test: update test suites and coverage verification".to_string(),
            explanation: format!("Test coverage changes across {} files", test_files.len()),
            files: test_files,
        });
    }

    if !doc_files.is_empty() {
        commits.push(ProposedCommit {
            commit_type: "docs".to_string(),
            scope: None,
            message: "docs: update documentation and specifications".to_string(),
            explanation: format!("Documentation updates across {} files", doc_files.len()),
            files: doc_files,
        });
    }

    // Only split if we produced 2 or more distinct commits
    if commits.len() >= 2 {
        Some(SplitCommitPlan {
            summary: format!("Deterministically partitioned into {} atomic commits", commits.len()),
            commits,
            unassigned_files: Vec::new(),
        })
    } else {
        None
    }
}

fn categorize_single_file(path_str: &str) -> (String, Option<String>, String) {
    let p = path_str.replace('\\', "/");
    let filename = Path::new(&p).file_name().and_then(|s| s.to_str()).unwrap_or(&p);

    if is_doc_file(&p) {
        ("docs".to_string(), None, format!("docs: update {filename}"))
    } else if is_dep_or_build_file(&p) {
        ("chore".to_string(), Some("deps".to_string()), format!("chore(deps): update {filename}"))
    } else if is_test_file(&p) {
        ("test".to_string(), None, format!("test: update {filename}"))
    } else if p.starts_with("crates/tsugi-core/") {
        ("feat".to_string(), Some("core".to_string()), format!("feat(core): update {filename}"))
    } else if p.starts_with("crates/tsugi-cli/") {
        ("feat".to_string(), Some("cli".to_string()), format!("feat(cli): update {filename}"))
    } else {
        ("feat".to_string(), None, format!("feat: update {filename}"))
    }
}

fn is_doc_file(p: &str) -> bool {
    p.ends_with(".md")
        || p.ends_with(".txt")
        || p.starts_with("docs/")
        || p.contains("/docs/")
        || p.eq_ignore_ascii_case("license")
        || p.eq_ignore_ascii_case("notice")
}

fn is_dep_or_build_file(p: &str) -> bool {
    p.ends_with("Cargo.toml")
        || p.ends_with("Cargo.lock")
        || p.ends_with("package.json")
        || p.ends_with("package-lock.json")
        || p.ends_with("pnpm-lock.yaml")
        || p.ends_with("yarn.lock")
        || p.ends_with("requirements.txt")
        || p.ends_with("pyproject.toml")
        || p.ends_with("go.mod")
        || p.ends_with("go.sum")
}

fn is_test_file(p: &str) -> bool {
    p.starts_with("tests/")
        || p.contains("/tests/")
        || p.ends_with("_test.rs")
        || p.ends_with("test.rs")
        || p.ends_with("_test.py")
        || p.ends_with("test_")
        || p.ends_with(".spec.ts")
        || p.ends_with(".test.ts")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FileDeltaStatus;

    fn mock_file(path: &str) -> ChangedFile {
        ChangedFile {
            path: path.to_string(),
            status: FileDeltaStatus::Modified,
            additions: 10,
            deletions: 2,
            diff_preview: "+test".to_string(),
        }
    }

    #[test]
    fn test_partition_docs_and_code() {
        let files = vec![
            mock_file("README.md"),
            mock_file("crates/tsugi-core/src/safety_vault.rs"),
        ];

        let plan = try_partition_changes(&files).expect("should partition");
        assert_eq!(plan.commits.len(), 2);
        assert_eq!(plan.commits[0].commit_type, "feat");
        assert_eq!(plan.commits[0].scope.as_deref(), Some("core"));
        assert_eq!(plan.commits[1].commit_type, "docs");
    }

    #[test]
    fn test_partition_multi_subsystem() {
        let files = vec![
            mock_file("Cargo.toml"),
            mock_file("crates/tsugi-core/src/probe.rs"),
            mock_file("crates/tsugi-cli/src/main.rs"),
            mock_file("docs/architecture.md"),
        ];

        let plan = try_partition_changes(&files).expect("should partition");
        assert_eq!(plan.commits.len(), 4);
    }

    #[test]
    fn test_single_file_partition() {
        let files = vec![mock_file("README.md")];
        let plan = try_partition_changes(&files).expect("single file plan");
        assert_eq!(plan.commits.len(), 1);
        assert_eq!(plan.commits[0].commit_type, "docs");
    }

    #[test]
    fn test_ambiguous_files_returns_none() {
        // Files in an unknown directory root
        let files = vec![
            mock_file("app/index.js"),
            mock_file("server/main.go"),
        ];
        assert!(try_partition_changes(&files).is_none());
    }
}
