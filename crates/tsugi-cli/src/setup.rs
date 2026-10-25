//! # Zero-Touch Automated Setup Wizard
//!
//! Automatically inspects hardware, downloads llama-cli, fetches and verifies
//! the GGUF model, compiles GBNF grammars, and configures tsugi without manual steps.

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use colored::Colorize;

use tsugi_core::{
    config::{
        self, bin_dir, check_setup, get_llama_release_info, load_config, models_dir,
        save_config, SetupStatus, DEFAULT_STOCK_MODEL_FILENAME, DEFAULT_STOCK_MODEL_SHA256,
        DEFAULT_STOCK_MODEL_URL, LLAMA_CPP_RELEASE_TAG,
    },
    engine, formatter,
};

use crate::downloader::{download_file, extract_archive};

/// Options for configuring the setup wizard.
#[derive(Debug, Default, Clone)]
pub struct SetupOptions {
    pub skip_download: bool,
    pub model_path: Option<PathBuf>,
    pub model_url: Option<String>,
}

/// Execute the full automated setup wizard.
pub fn run_setup(options: SetupOptions) -> Result<()> {
    formatter::print_banner();

    println!("\n  {}", "Zero-Touch Automated Setup".white().bold());
    println!("  {}", "───────────────────────────".dimmed());

    // 1. Hardware and environment detection
    println!("\n  {} Inspecting environment...", "1.".cyan().bold());
    let os_name = std::env::consts::OS;
    let arch_name = std::env::consts::ARCH;
    println!("     · OS / Arch: {} / {}", os_name.white().bold(), arch_name.white().bold());

    let bin_path = bin_dir();
    let models_path = models_dir();
    fs::create_dir_all(&bin_path)?;
    fs::create_dir_all(&models_path)?;

    let mut config = load_config().unwrap_or_default();

    // 2. Resolve or install llama-cli
    println!("\n  {} Verifying inference engine (llama-cli)...", "2.".cyan().bold());
    let current_llama = engine::find_llama_cli();

    if let Ok(ref path) = current_llama {
        println!(
            "     {} Found existing llama-cli: {}",
            "✔".green(),
            path.display().to_string().cyan()
        );
        config.llama_cli_path = Some(path.clone());
    } else if options.skip_download {
        println!(
            "     {} Skipping llama-cli download (--skip-download specified)",
            "ℹ".yellow()
        );
    } else {
        println!("     · llama-cli not detected. Downloading pre-built release...");
        if let Some(release_info) = get_llama_release_info(LLAMA_CPP_RELEASE_TAG) {
            let archive_dest = bin_path.join(&release_info.archive_name);
            download_file(
                &release_info.url,
                &archive_dest,
                None,
                &format!("llama.cpp ({})", LLAMA_CPP_RELEASE_TAG),
            )?;

            println!("     · Extracting binaries into {}...", bin_path.display());
            extract_archive(&archive_dest, &bin_path, release_info.is_zip)?;
            let _ = fs::remove_file(&archive_dest);

            let expected_bin = bin_path.join(release_info.binary_name);
            if expected_bin.exists() {
                println!(
                    "     {} Installed llama-cli: {}",
                    "✔".green().bold(),
                    expected_bin.display().to_string().cyan()
                );
                config.llama_cli_path = Some(expected_bin);
            } else {
                println!(
                    "     {} Extracted archive into {}",
                    "✔".green(),
                    bin_path.display()
                );
            }
        } else {
            println!(
                "     {} Pre-built binary not packaged for {}/{}",
                "⚠".yellow(),
                os_name,
                arch_name
            );
            println!("       Please install via your package manager (e.g. `brew install llama.cpp`)");
        }
    }

    // 3. Resolve or download the GGUF model
    println!("\n  {} Configuring language model...", "3.".cyan().bold());
    if let Some(custom_path) = options.model_path {
        if !custom_path.exists() {
            anyhow::bail!("Specified model path does not exist: {}", custom_path.display());
        }
        println!(
            "     {} Using user-provided model: {}",
            "✔".green(),
            custom_path.display().to_string().cyan()
        );
        config.primary_model_path = Some(custom_path);
    } else {
        let target_model_file = models_path.join(DEFAULT_STOCK_MODEL_FILENAME);
        if target_model_file.exists() && fs::metadata(&target_model_file)?.len() > 10_000_000 {
            println!(
                "     {} Model file already present: {}",
                "✔".green(),
                target_model_file.display().to_string().cyan()
            );
            config.primary_model_path = Some(target_model_file);
        } else if options.skip_download {
            println!(
                "     {} Skipping model download (--skip-download specified)",
                "ℹ".yellow()
            );
            config.primary_model_path = Some(target_model_file);
        } else {
            let download_url = options.model_url.as_deref().unwrap_or(DEFAULT_STOCK_MODEL_URL);
            let verify_hash = if options.model_url.is_none() {
                Some(DEFAULT_STOCK_MODEL_SHA256)
            } else {
                None
            };

            println!("     · Downloading Qwen2.5-Coder-1.5B-Instruct (~1.1 GB)...");
            download_file(
                download_url,
                &target_model_file,
                verify_hash,
                "Qwen2.5-Coder-1.5B GGUF",
            )?;
            config.primary_model_path = Some(target_model_file);
        }
    }

    // 4. Grammar and configuration persistence
    println!("\n  {} Finalizing grammar & config...", "4.".cyan().bold());
    let grammar_path = config::config_dir().join("tsugi_plan.gbnf");
    engine::ensure_grammar_file(&grammar_path)?;
    config.grammar_path = Some(grammar_path);

    save_config(&config)?;
    println!("     {} Configuration saved successfully.", "✔".green().bold());

    // 5. Final status check
    println!("\n  {}", "Verification".white().bold());
    println!("  {}", "────────────".dimmed());
    match check_setup() {
        SetupStatus::Ready {
            primary_model,
            has_llama_cli,
            ..
        } => {
            println!("\n  {} {}", "✔".green().bold(), "tsugi is 100% configured and ready!".green().bold());
            if let Some(m) = primary_model {
                println!("    · Model:     {}", m.display().to_string().cyan());
            }
            println!(
                "    · Inference: {}",
                if has_llama_cli {
                    "llama-cli active".green().to_string()
                } else {
                    "pending".yellow().to_string()
                }
            );
            println!("\n  Try running {} inside any Git repository to test.", "tsugi diagnose".white().bold());
        }
        _ => {
            println!(
                "\n  {} Setup completed with missing components. Run {} for details.",
                "⚠".yellow(),
                "tsugi status".white().bold()
            );
        }
    }

    Ok(())
}

/// Check if the model is ready, and if missing, prompt the user for lazy on-demand download.
pub fn ensure_model_lazy() -> Result<bool> {
    match check_setup() {
        SetupStatus::Ready { .. } => Ok(true),
        SetupStatus::MissingModel | SetupStatus::MissingLlamaCli | SetupStatus::NoConfig => {
            println!(
                "\n  {} {}",
                "⚠".yellow().bold(),
                "tsugi needs the local AI model to resolve this complex scenario.".yellow()
            );
            println!("    The model runs 100% locally on your machine with zero data leaving your laptop.");
            print!("    Download and configure now automatically? [Y/n]: ");
            io::stdout().flush().ok();

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let trimmed = input.trim().to_lowercase();
            if trimmed.is_empty() || trimmed == "y" || trimmed == "yes" {
                run_setup(SetupOptions::default())?;
                Ok(true)
            } else {
                println!("  {} Download cancelled. Falling back to deterministic Tier-0 resolution.", "✖".red());
                Ok(false)
            }
        }
    }
}
