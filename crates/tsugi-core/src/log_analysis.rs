//! # Commit Log & Graph Intelligence Analysis
//!
//! Provides deterministic commit history audits: Conventional Commits compliance,
//! WIP commit detection, merge vs rebase cleanliness, and author contributions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::CommitEntry;

/// Detailed analysis report of recent commit history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogAnalysisReport {
    pub total_analyzed: usize,
    pub conventional_commits_count: usize,
    pub conventional_compliance_pct: f32,
    pub wip_commits_count: usize,
    pub merge_commits_count: usize,
    pub author_distribution: Vec<(String, usize)>,
    pub recommendations: Vec<String>,
}

/// Analyze recent commit history and produce a health report.
pub fn analyze_commits(commits: &[CommitEntry]) -> LogAnalysisReport {
    if commits.is_empty() {
        return LogAnalysisReport {
            total_analyzed: 0,
            conventional_commits_count: 0,
            conventional_compliance_pct: 100.0,
            wip_commits_count: 0,
            merge_commits_count: 0,
            author_distribution: Vec::new(),
            recommendations: vec!["No recent commits to analyze.".to_string()],
        };
    }

    let mut conventional_count = 0;
    let mut wip_count = 0;
    let mut merge_count = 0;
    let mut authors: HashMap<String, usize> = HashMap::new();

    let conventional_prefixes = [
        "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert",
    ];

    for c in commits {
        *authors.entry(c.author.clone()).or_insert(0) += 1;

        let msg = c.message_oneline.trim();
        let lower = msg.to_lowercase();

        // Check merge
        if lower.starts_with("merge ") {
            merge_count += 1;
        }

        // Check WIP
        if lower.starts_with("wip")
            || lower.starts_with("fixup!")
            || lower.starts_with("squash!")
            || lower.contains("temp commit")
            || lower == "wip"
            || lower == "temp"
        {
            wip_count += 1;
        }

        // Check Conventional Commits: <type>[optional scope]: <description>
        if let Some(colon_idx) = msg.find(':') {
            let prefix_part = &msg[..colon_idx].trim();
            // strip optional (scope)
            let type_part = if let Some(paren_idx) = prefix_part.find('(') {
                &prefix_part[..paren_idx]
            } else {
                prefix_part
            };
            if conventional_prefixes.iter().any(|&p| p.eq_ignore_ascii_case(type_part)) {
                conventional_count += 1;
            }
        }
    }

    let total = commits.len();
    let conventional_compliance_pct = (conventional_count as f32 / total as f32) * 100.0;

    let mut author_distribution: Vec<(String, usize)> = authors.into_iter().collect();
    author_distribution.sort_by(|a, b| b.1.cmp(&a.1));

    let mut recommendations = Vec::new();

    if wip_count > 0 {
        recommendations.push(format!(
            "Found {} WIP/fixup commit(s). Consider running `tsugi split` or interactive rebase to squash them before PR.",
            wip_count
        ));
    }

    if conventional_compliance_pct < 70.0 {
        recommendations.push(format!(
            "Conventional Commits compliance is low ({:.1}%). Standardize commit messages (feat:, fix:, docs:, chore:).",
            conventional_compliance_pct
        ));
    } else {
        recommendations.push(format!(
            "Commit messages are well-structured ({:.1}% Conventional Commits compliance).",
            conventional_compliance_pct
        ));
    }

    if merge_count > 0 {
        recommendations.push(format!(
            "History contains {} merge commit(s). Prefer `git pull --rebase` to maintain a linear history.",
            merge_count
        ));
    }

    LogAnalysisReport {
        total_analyzed: total,
        conventional_commits_count: conventional_count,
        conventional_compliance_pct,
        wip_commits_count: wip_count,
        merge_commits_count: merge_count,
        author_distribution,
        recommendations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_commits_wip_and_conventional() {
        let commits = vec![
            CommitEntry {
                sha_short: "abc1234".to_string(),
                message_oneline: "feat(core): add safety vault".to_string(),
                author: "Alice".to_string(),
                timestamp: "2026-10-28".to_string(),
                decorations: vec![],
            },
            CommitEntry {
                sha_short: "def5678".to_string(),
                message_oneline: "WIP: partial work".to_string(),
                author: "Alice".to_string(),
                timestamp: "2026-10-28".to_string(),
                decorations: vec![],
            },
            CommitEntry {
                sha_short: "ghi9012".to_string(),
                message_oneline: "fix: resolve deadlock".to_string(),
                author: "Bob".to_string(),
                timestamp: "2026-10-28".to_string(),
                decorations: vec![],
            },
        ];

        let report = analyze_commits(&commits);
        assert_eq!(report.total_analyzed, 3);
        assert_eq!(report.wip_commits_count, 1);
        assert_eq!(report.conventional_commits_count, 2);
        assert_eq!(report.author_distribution[0], ("Alice".to_string(), 2));
    }
}
