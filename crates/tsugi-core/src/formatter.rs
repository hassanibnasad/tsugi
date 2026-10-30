//! # Terminal Output Formatter
//!
//! Rich, colored terminal rendering for tsugi's diagnostic output and resolution plans.
//! Designed for clarity under pressure — when a developer's Git state is broken,
//! the output must be scannable and unambiguous.

use colored::Colorize;

use crate::types::{RepoState, ResolutionPlan, RiskTier, HeadState, PlanSource};
use crate::ledger::OperationRecord;

/// Print the tsugi banner.
pub fn print_banner() {
    println!(
        "{}",
        r#"
  ╔═══════════════════════════════════════╗
  ║       次  tsugi  ·  Git Assistant     ║
  ╚═══════════════════════════════════════╝"#
            .cyan()
            .bold()
    );
}

/// Print a compact diagnostic summary of the repository state.
pub fn print_diagnosis(state: &RepoState) {
    println!("\n{}", "─── Diagnosis ──────────────────────────────".dimmed());

    // HEAD state
    let head_icon = match &state.head {
        HeadState::Attached { .. } => "🔗",
        HeadState::Detached { .. } => "⚡",
        HeadState::Unborn { .. } => "🌱",
    };
    println!("  {} HEAD: {}", head_icon, format!("{}", state.head).white().bold());

    // Ongoing operation
    if let Some(ref op) = state.ongoing_operation {
        println!(
            "  {} Operation: {}",
            "⚠".yellow(),
            format!("{op}").yellow().bold()
        );
    }

    // Conflicts
    if !state.conflicts.is_empty() {
        println!(
            "  {} Conflicts: {}",
            "❌".red(),
            format!("{} file(s)", state.conflicts.len()).red().bold()
        );
        for conflict in &state.conflicts {
            println!(
                "     {} {} ({})",
                "·".dimmed(),
                conflict.path.white(),
                format!("{}", conflict.conflict_type).dimmed()
            );

            if let Some(ref hunk) = conflict.primary_hunk {
                println!("       {} {}", "┌── Conflict block in".dimmed(), hunk.file_path.cyan());
                println!("       {} <<<<<<< {}", "│".dimmed(), hunk.ours_branch.yellow().bold());
                for line in &hunk.ours_content {
                    println!("       {} {}", "│".dimmed(), line.green());
                }
                println!("       {} =======", "│".dimmed());
                for line in &hunk.theirs_content {
                    println!("       {} {}", "│".dimmed(), line.red());
                }
                println!("       {} >>>>>>> {}", "│".dimmed(), hunk.theirs_branch.yellow().bold());
                println!("       {}", "└──".dimmed());
            }
        }
    }

    // Working tree summary
    let wt = &state.working_tree;
    if wt.has_changes() {
        let staged = wt.staged_new + wt.staged_modified + wt.staged_deleted + wt.staged_renamed;
        let unstaged = wt.unstaged_modified + wt.unstaged_deleted;
        let untracked = wt.untracked;

        let mut parts = Vec::new();
        if staged > 0 {
            parts.push(format!("{} staged", staged).green().to_string());
        }
        if unstaged > 0 {
            parts.push(format!("{} unstaged", unstaged).yellow().to_string());
        }
        if untracked > 0 {
            parts.push(format!("{} untracked", untracked).dimmed().to_string());
        }

        println!("  📁 Working tree: {}", parts.join(" · "));
    } else {
        println!("  📁 Working tree: {}", "clean".green());
    }

    // Tracking
    if let Some(ref tracking) = state.tracking {
        let tracking_str = if tracking.ahead > 0 && tracking.behind > 0 {
            format!(
                "{} (↑{} ↓{})",
                tracking.upstream, tracking.ahead, tracking.behind
            )
            .red()
            .to_string()
        } else if tracking.ahead > 0 {
            format!("{} (↑{})", tracking.upstream, tracking.ahead)
                .green()
                .to_string()
        } else if tracking.behind > 0 {
            format!("{} (↓{})", tracking.upstream, tracking.behind)
                .yellow()
                .to_string()
        } else {
            format!("{} (up to date)", tracking.upstream)
                .dimmed()
                .to_string()
        };
        println!("  🔀 Tracking: {tracking_str}");
    }

    // Stashes
    if state.stash_count > 0 {
        println!(
            "  📦 Stashes: {}",
            format!("{}", state.stash_count).white()
        );
    }

    // Submodules
    if !state.submodules.is_empty() {
        let uninit = state.submodules.iter().filter(|s| !s.is_initialized).count();
        let dirty = state.submodules.iter().filter(|s| s.is_dirty).count();
        if uninit > 0 || dirty > 0 {
            println!("  🧩 Submodules: {} ({} uninit, {} dirty)", state.submodules.len().to_string().yellow(), uninit, dirty);
        }
    }

    // Worktrees
    if state.worktrees.len() > 1 {
        println!("  🌲 Worktrees: {} active linked worktrees", state.worktrees.len().to_string().cyan());
    }

    // Shallow
    if state.is_shallow {
        println!("  🌱 Shallow: Truncated history clone");
    }

    // Large files
    if !state.large_files.is_empty() {
        println!("  ⚠️ Large Files: {} file(s) >5MB staged in index", state.large_files.len().to_string().yellow());
    }

    println!("{}", "────────────────────────────────────────────".dimmed());
}

/// Print a resolution plan in a clear, actionable format.
pub fn print_plan(plan: &ResolutionPlan) {
    println!("\n{}", "─── Resolution Plan ────────────────────────".dimmed());

    // Root cause
    println!(
        "  {} {}",
        "Root Cause:".white().bold(),
        plan.root_cause.white()
    );

    // Risk tier with color coding
    let risk_display = match plan.risk_tier {
        RiskTier::ReadOnly => "READ ONLY".green().bold(),
        RiskTier::MutatingSafe => "MUTATING (safe)".blue().bold(),
        RiskTier::MutatingDangerous => "⚠ MUTATING (dangerous)".yellow().bold(),
        RiskTier::HistoryRewrite => "🚨 HISTORY REWRITE".red().bold(),
    };
    println!("  {} {}", "Risk:".white().bold(), risk_display);

    // Source engine
    let source_display = match plan.source {
        PlanSource::Tier0 => "Deterministic (Tier-0)".green(),
        PlanSource::LlmPrimary => "Qwen2.5-Coder-1.5B".cyan(),
        PlanSource::LlmFallback => "Qwen3-0.6B (low-mem)".yellow(),
    };
    println!(
        "  {} {} (confidence: {:.0}%)",
        "Engine:".white().bold(),
        source_display,
        plan.confidence * 100.0
    );

    // Steps
    println!("\n  {}", "Steps:".white().bold());
    for (i, step) in plan.execution_steps.iter().enumerate() {
        let num = i + 1;
        let icon = if step.requires_confirmation {
            "⚠".yellow().to_string()
        } else if step.cmd.starts_with('#') {
            "📝".to_string()
        } else {
            "▶".green().to_string()
        };

        println!(
            "  {icon} {}: {}",
            format!("[{num}]").dimmed(),
            step.cmd.white().bold()
        );
        println!("      {}", step.explanation.dimmed());

        if step.requires_confirmation {
            println!("      {}", "↑ Requires confirmation before execution".yellow());
        }
    }

    // Rollback
    println!("\n  {} {}", "Rollback:".white().bold(), plan.rollback_cmd.dimmed());
    println!("{}", "────────────────────────────────────────────".dimmed());
}

/// Print the result of a `tsugi undo` operation.
pub fn print_undo_result(success: bool, snapshot_summary: &str) {
    if success {
        println!(
            "\n  {} {}",
            "✔ Undo successful!".green().bold(),
            format!("Restored to: {snapshot_summary}").dimmed()
        );
    } else {
        println!(
            "\n  {} {}",
            "✖ Undo failed".red().bold(),
            "Check `tsugi history` for manual recovery options".dimmed()
        );
    }
}

/// Print operation history.
pub fn print_history(operations: &[OperationRecord]) {
    if operations.is_empty() {
        println!("\n  {}", "No operations recorded yet.".dimmed());
        return;
    }

    println!("\n{}", "─── Operation History ──────────────────────".dimmed());

    for (_i, op) in operations.iter().enumerate() {
        let status_icon = if op.undo_triggered {
            "↩".yellow().to_string()
        } else if op.success {
            "✔".green().to_string()
        } else {
            "✖".red().to_string()
        };

        println!(
            "  {status_icon} {} {} [{}]",
            op.timestamp[..19].dimmed(), // Trim timezone for readability
            op.user_intent.white(),
            op.plan_source.dimmed()
        );
        println!(
            "    {} Risk: {} | {}",
            "·".dimmed(),
            op.risk_tier.dimmed(),
            op.repo_path.dimmed()
        );
    }

    println!("{}", "────────────────────────────────────────────".dimmed());
}

/// Print setup status.
pub fn print_setup_needed() {
    println!(
        "\n{}",
        "─── Setup Required ────────────────────────".yellow()
    );
    println!("  tsugi needs a local GGUF model to provide AI-powered resolutions.");
    println!();
    println!("  Run {} to configure:", "tsugi setup".white().bold());
    println!("    1. Download Qwen2.5-Coder-1.5B-Instruct (IQ4_XS) — ~840 MB");
    println!("    2. Install llama-cli (inference engine)");
    println!("    3. Generate the GBNF grammar file");
    println!();
    println!(
        "  {} tsugi works without AI for common issues (Tier-0 fast-path).",
        "Note:".cyan().bold()
    );
    println!(
        "  Only complex, ambiguous cases require the AI model.",
    );
    println!("{}", "────────────────────────────────────────────".dimmed());
}
