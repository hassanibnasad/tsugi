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

use tsugi_core::{
    config::{self, check_setup, load_config, SetupStatus},
    engine,
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

    /// Interactive setup wizard
    Setup,

    /// Clean up old snapshot refs (default: older than 30 days)
    Cleanup {
        /// Maximum age in days for snapshot retention
        #[arg(short, long, default_value = "30")]
        days: u32,
    },
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
        Some(Commands::Setup) => cmd_setup(),
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

    // Layer 2 & 3: Produce a guardrail-checked plan
    let plan = match engine.plan(&state, user_intent)? {
        PlanOutcome::Ready(p) => p,
        PlanOutcome::SetupNeeded => {
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
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    // Print the plan
    formatter::print_plan(&plan);

    // For dangerous plans, require explicit go-ahead
    if matches!(plan.risk_tier, RiskTier::MutatingDangerous | RiskTier::HistoryRewrite) && !auto_confirm {
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

fn cmd_setup() -> Result<()> {
    formatter::print_banner();

    println!("\n  {}", "Setup Wizard".white().bold());
    println!("  {}", "─────────────".dimmed());

    println!("\n  tsugi requires two components:");
    println!("  1. A GGUF model file (Qwen2.5-Coder-1.5B-Instruct, ~840 MB)");
    println!("  2. llama-cli (the inference engine from llama.cpp)");

    println!("\n  {} Download the model:", "Step 1:".cyan().bold());
    println!("    Visit: {}", "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF".dimmed());
    println!("    Download: {}", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".white().bold());
    println!("    Place it in: {}", config::models_dir().display());

    println!("\n  {} Install llama-cli:", "Step 2:".cyan().bold());
    println!("    Visit: {}", "https://github.com/ggerganov/llama.cpp/releases".dimmed());
    println!("    Download the release for your OS and add llama-cli to your PATH.");

    println!("\n  {} Configure tsugi:", "Step 3:".cyan().bold());

    let models_dir = config::models_dir();
    std::fs::create_dir_all(&models_dir)?;

    let mut config = load_config().unwrap_or_default();
    config.primary_model_path = Some(models_dir.join("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"));
    config.fallback_model_path = Some(models_dir.join("qwen3-0.6b-q5_k_m.gguf"));

    // Create GBNF grammar file
    let grammar_path = config::config_dir().join("tsugi_plan.gbnf");
    engine::ensure_grammar_file(&grammar_path)?;
    config.grammar_path = Some(grammar_path);

    config::save_config(&config)?;

    println!("    {} Configuration saved!", "✔".green().bold());

    println!(
        "\n  Once you've completed steps 1-2, run {} to verify.",
        "tsugi status".white().bold()
    );

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

