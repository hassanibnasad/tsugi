//! # tsugi CLI — Enterprise-Grade Terminal Git Assistant
//!
//! ## Usage
//!
//! ```bash
//! # Diagnose and fix the current Git issue
//! tsugi fix "I messed up my rebase"
//! tsugi fix                          # Auto-detect from repo state
//!
//! # Undo the last tsugi operation
//! tsugi undo
//!
//! # View operation history
//! tsugi history
//!
//! # Diagnose without executing (dry run)
//! tsugi diagnose
//!
//! # Setup and configuration
//! tsugi setup
//! tsugi status
//! ```

use std::path::PathBuf;
use std::process;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use tracing_subscriber::{fmt, EnvFilter};

mod conflict_ui;
mod downloader;
mod guard;
mod setup;
mod shell;

use tsugi_core::{
    config::{check_setup, load_config, SetupStatus},
    formatter,
    probe,
    tier0,
    types::RiskTier,
    PlanOutcome, TerminalObserver, TsugiEngine,
};

#[derive(Parser)]
#[command(
    name = "tsugi",
    version,
    about = "次 tsugi — Enterprise-grade terminal Git assistant powered by local AI",
    long_about = "tsugi diagnoses and resolves Git issues using a combination of deterministic \
                  pattern matching and a fine-tuned local language model. It runs entirely on \
                  your machine — no API keys, no cloud, no data leaves your laptop.",
    after_help = "EXAMPLES:\n  \
                  tsugi fix \"I messed up my rebase\"\n  \
                  tsugi fix\n  \
                  tsugi undo\n  \
                  tsugi diagnose\n  \
                  tsugi history"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Enable verbose debug logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Specify the repository path (defaults to current directory)
    #[arg(short = 'C', long, global = true)]
    repo: Option<PathBuf>,

    /// Skip all confirmation prompts (use with caution)
    #[arg(long, global = true)]
    yes: bool,

    /// Output in JSON format (for scripting/integration)
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Diagnose and fix the current Git issue
    Fix {
        /// Description of your Git issue (optional — tsugi auto-detects from repo state)
        #[arg(trailing_var_arg = true)]
        intent: Vec<String>,
    },

    /// Diagnose the current Git state without executing anything (dry run)
    Diagnose,

    /// Undo the last tsugi operation (restores pre-fix state)
    Undo,

    /// View tsugi operation history
    History {
        /// Maximum number of entries to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Check tsugi setup status
    Status,

    /// Automated zero-touch setup wizard
    Setup {
        /// Skip downloading model and llama-cli binaries (for offline / CI environments)
        #[arg(long)]
        skip_download: bool,

        /// Path to an existing local GGUF model file
        #[arg(long)]
        model_path: Option<PathBuf>,

        /// Custom Hugging Face download URL for a GGUF model
        #[arg(long)]
        model_url: Option<String>,
    },

    /// Git Guard safety net: automatic pre-operation snapshots
    Guard {
        #[command(subcommand)]
        action: GuardActions,
    },

    /// Silent fast checkpoint command invoked by Git guard hooks
    #[command(hide = true)]
    Checkpoint {
        #[arg(long)]
        quiet: bool,
    },

    /// Output shell integration script to hint tsugi on failed git commands
    Init {
        /// Shell type: powershell, bash, zsh, fish
        shell: String,
    },

    /// Clean up old snapshot refs (default: older than 30 days)
    Cleanup {
        /// Maximum age in days for snapshot retention
        #[arg(short, long, default_value = "30")]
        days: u32,
    },
}

#[derive(Subcommand)]
enum GuardActions {
    /// Install guard hooks into .git/hooks/
    Install {
        /// Install hooks globally via git core.hooksPath
        #[arg(long)]
        global: bool,
    },
    /// Uninstall guard hooks
    Uninstall {
        /// Uninstall global guard hooks
        #[arg(long)]
        global: bool,
    },
    /// Check guard hook installation status
    Status,
}

pub fn run() {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("warn")
    };

    fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();

    // Determine repository path
    let repo_path = cli
        .repo
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Execute the command (defaults to Fix with auto-diagnosis if invoked as bare `tsugi` or `git tsugi`)
    let result = match cli.command {
        Some(Commands::Fix { intent }) => cmd_fix(&repo_path, &intent.join(" "), cli.yes, cli.json),
        Some(Commands::Diagnose) => cmd_diagnose(&repo_path, cli.json),
        Some(Commands::Undo) => cmd_undo(&repo_path),
        Some(Commands::History { limit }) => cmd_history(&repo_path, limit),
        Some(Commands::Status) => cmd_status(),
        Some(Commands::Setup { skip_download, model_path, model_url }) => {
            setup::run_setup(setup::SetupOptions {
                skip_download,
                model_path,
                model_url,
            })
        }
        Some(Commands::Guard { action }) => match action {
            GuardActions::Install { global } => guard::install_hooks(&repo_path, global),
            GuardActions::Uninstall { global } => guard::uninstall_hooks(&repo_path, global),
            GuardActions::Status => guard::guard_status(&repo_path),
        },
        Some(Commands::Checkpoint { quiet }) => guard::cmd_checkpoint(&repo_path, quiet),
        Some(Commands::Init { shell }) => shell::print_shell_init(&shell),
        Some(Commands::Cleanup { days }) => cmd_cleanup(&repo_path, days),
        None => cmd_fix(&repo_path, "", cli.yes, cli.json),
    };

    if let Err(e) = result {
        eprintln!("\n  {} {}", "Error:".red().bold(), e);
        process::exit(1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Command Implementations
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_fix(repo_path: &PathBuf, user_intent: &str, auto_confirm: bool, json_output: bool) -> Result<()> {
    if !json_output {
        formatter::print_banner();
    }

    let config = load_config().context("Failed to load configuration")?;
    let engine = TsugiEngine::new(config)?;

    // Layer 1: Probe repository state
    let state = engine.probe(repo_path)
        .context("Failed to probe repository state")?;

    if !json_output {
        formatter::print_diagnosis(&state);
    }

    // Interactive quick-choice conflict resolution menu
    if !auto_confirm && !json_output && !state.conflicts.is_empty() {
        if let Some(conflict) = state.conflicts.iter().find(|c| c.primary_hunk.is_some()) {
            match conflict_ui::prompt_conflict_resolution(conflict)? {
                conflict_ui::ConflictChoice::KeepOurs => {
                    conflict_ui::resolve_with_checkout(repo_path, &conflict.path, true)?;
                    println!("  {} Run 'tsugi' to verify remaining status or continue operation.", "💡".cyan());
                    return Ok(());
                }
                conflict_ui::ConflictChoice::KeepTheirs => {
                    conflict_ui::resolve_with_checkout(repo_path, &conflict.path, false)?;
                    println!("  {} Run 'tsugi' to verify remaining status or continue operation.", "💡".cyan());
                    return Ok(());
                }
                conflict_ui::ConflictChoice::LetAiResolve => {
                    println!("  {} Delegating conflict resolution to AI...", "▶".cyan());
                }
                conflict_ui::ConflictChoice::Cancel => {
                    println!("  {} Conflict resolution cancelled.", "✖".red());
                    return Ok(());
                }
            }
        }
    }

    // Layer 2 & 3: Produce a guardrail-checked plan
    let plan = match engine.plan(&state, user_intent)? {
        PlanOutcome::Ready(p) => p,
        PlanOutcome::SetupNeeded => {
            if !json_output && setup::ensure_model_lazy()? {
                let reloaded_config = load_config().context("Failed to reload configuration")?;
                let reloaded_engine = TsugiEngine::new(reloaded_config)?;
                match reloaded_engine.plan(&state, user_intent)? {
                    PlanOutcome::Ready(p) => p,
                    PlanOutcome::SetupNeeded => {
                        formatter::print_setup_needed();
                        return Ok(());
                    }
                }
            } else {
                if !json_output {
                    formatter::print_setup_needed();
                    println!(
                        "\n  {} {}",
                        "Hint:".cyan().bold(),
                        "Tier-0 fast-path couldn't match this issue. A local AI model is needed."
                    );
                    println!(
                        "  Run {} to set up the AI engine.",
                        "tsugi setup".white().bold()
                    );
                }
                return Ok(());
            }
        }
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    // Print the plan
    formatter::print_plan(&plan);

    // For dangerous plans or plans containing dangerous steps, require explicit go-ahead
    let requires_plan_confirm = matches!(plan.risk_tier, RiskTier::MutatingDangerous | RiskTier::HistoryRewrite)
        || plan.execution_steps.iter().any(|s| s.requires_confirmation);
    if requires_plan_confirm && !auto_confirm {
        println!(
            "\n  {} {}",
            "⚠".yellow(),
            "This operation is classified as dangerous. Review the plan above carefully.".yellow()
        );
        print!(
            "  {} ",
            "Proceed with execution? [y/N]:".yellow().bold()
        );
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
            println!("  {} Aborted", "✖".red().bold());
            return Ok(());
        }
    }

    // Execute the plan through the observer seam
    println!("\n{}", "─── Executing ─────────────────────────────".dimmed());

    let observer = TerminalObserver::new(auto_confirm);
    let _executed = engine.execute(repo_path, &plan, user_intent, &observer)?;

    println!("\n  {} {}", "✔".green().bold(), "Resolution complete!".green().bold());

    Ok(())
}

fn cmd_diagnose(repo_path: &PathBuf, json_output: bool) -> Result<()> {
    let state = probe::probe_repo_state(repo_path)
        .context("Failed to probe repository state")?;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&state)?);
    } else {
        formatter::print_banner();
        formatter::print_diagnosis(&state);

        // Also show what Tier-0 would do (without user intent)
        if let Some(plan) = tier0::try_resolve(&state, "") {
            println!(
                "\n  {} {}",
                "💡".to_string(),
                "Tier-0 auto-detected a known issue:".cyan().bold()
            );
            formatter::print_plan(&plan);
            println!(
                "\n  Run {} to execute this fix.",
                "tsugi fix".white().bold()
            );
        } else if state.ongoing_operation.is_some() || !state.conflicts.is_empty() {
            println!(
                "\n  {} {}",
                "💡".to_string(),
                "Issue detected but requires AI analysis. Run:".cyan()
            );
            println!(
                "    {} \"describe your issue\"",
                "tsugi fix".white().bold()
            );
        } else {
            println!(
                "\n  {} {}",
                "✔".green().bold(),
                "Repository looks healthy!".green()
            );
        }
    }

    Ok(())
}

fn cmd_undo(repo_path: &PathBuf) -> Result<()> {
    formatter::print_banner();

    let config = load_config().unwrap_or_default();
    let engine = TsugiEngine::new(config)?;

    let snapshot = engine.vault().get_latest_snapshot_for_repo(repo_path)
        .context("Failed to query snapshots")?;

    match snapshot {
        Some(snap) => {
            println!(
                "\n  {} Restoring to: {} ({})",
                "↩".yellow(),
                snap.head_sha[..7.min(snap.head_sha.len())].white().bold(),
                snap.plan_summary.dimmed()
            );

            print!(
                "  {} ",
                "Confirm undo? [y/N]:".yellow().bold()
            );
            std::io::Write::flush(&mut std::io::stdout()).ok();

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
                println!("  {} Cancelled", "✖".red().bold());
                return Ok(());
            }

            engine.undo(repo_path)
                .context("Snapshot restore failed")?;

            formatter::print_undo_result(true, &snap.plan_summary);
        }
        None => {
            println!(
                "\n  {} No snapshots available for undo.",
                "ℹ".cyan()
            );
            println!("  tsugi only stores snapshots for mutating operations.");
        }
    }

    Ok(())
}

fn cmd_history(repo_path: &PathBuf, limit: usize) -> Result<()> {
    formatter::print_banner();

    let config = load_config().unwrap_or_default();
    let engine = TsugiEngine::new(config)?;

    let operations = engine
        .history(Some(&repo_path.to_string_lossy()), limit)
        .context("Failed to query operation history")?;

    formatter::print_history(&operations);

    Ok(())
}

fn cmd_status() -> Result<()> {
    formatter::print_banner();

    match check_setup() {
        SetupStatus::Ready {
            primary_model,
            fallback_model,
            has_llama_cli,
            server_endpoint,
        } => {
            println!("\n  {} tsugi is fully configured!", "✔".green().bold());
            if let Some(endpoint) = server_endpoint {
                println!("  {} Local llama-server: {}", "·".dimmed(), endpoint.cyan());
            }
            if let Some(path) = primary_model {
                println!("  {} Primary model: {}", "·".dimmed(), path.display());
            }
            if let Some(path) = fallback_model {
                println!("  {} Fallback model: {}", "·".dimmed(), path.display());
            }
            println!(
                "  {} llama-cli: {}",
                "·".dimmed(),
                if has_llama_cli {
                    "found".green().to_string()
                } else {
                    "not found (using llama-server or needs install)".yellow().to_string()
                }
            );
        }
        SetupStatus::NoConfig => {
            formatter::print_setup_needed();
        }
        SetupStatus::MissingModel => {
            println!(
                "\n  {} Model files not found. Run {} to download.",
                "⚠".yellow(),
                "tsugi setup".white().bold()
            );
        }
        SetupStatus::MissingLlamaCli => {
            println!(
                "\n  {} llama-cli not found. Install llama.cpp or add it to your PATH.",
                "⚠".yellow(),
            );
        }
    }

    Ok(())
}


fn cmd_cleanup(repo_path: &PathBuf, days: u32) -> Result<()> {
    formatter::print_banner();

    let config = load_config().unwrap_or_default();
    let engine = TsugiEngine::new(config)?;

    let cleaned = engine.cleanup(repo_path, days)
        .context("Cleanup failed")?;

    println!(
        "\n  {} Cleaned up {} old snapshot(s) (older than {} days)",
        "🧹".to_string(),
        cleaned,
        days
    );

    Ok(())
}

