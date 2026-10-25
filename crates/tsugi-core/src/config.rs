//! # Configuration Management
//!
//! Handles loading and saving tsugi configuration from `~/.tsugi/config.toml`.

use std::path::PathBuf;

use tracing::info;

use crate::types::{TsugiConfig, TsugiError};

/// Default configuration directory.
pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".tsugi")
}

/// Default model storage directory.
pub fn models_dir() -> PathBuf {
    config_dir().join("models")
}

/// Default binary directory for tsugi-managed executables (e.g. llama-cli).
pub fn bin_dir() -> PathBuf {
    config_dir().join("bin")
}

/// Default stock model download URL (official Qwen2.5-Coder-1.5B-Instruct GGUF).
pub const DEFAULT_STOCK_MODEL_URL: &str =
    "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

/// Default stock model filename.
pub const DEFAULT_STOCK_MODEL_FILENAME: &str =
    "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

/// Verified SHA-256 for the official Qwen2.5-Coder-1.5B-Instruct GGUF q4_k_m.
pub const DEFAULT_STOCK_MODEL_SHA256: &str =
    "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046";

/// Swappable custom fine-tuned model URL constant.
pub const CUSTOM_FINETUNE_MODEL_URL: &str =
    "https://huggingface.co/hassanibnasad/tsugi-model/resolve/main/tsugi-1.5b-q4_k_m.gguf";

/// Pinned stable release tag for llama.cpp pre-built binaries.
pub const LLAMA_CPP_RELEASE_TAG: &str = "b4800";

/// Information about the platform's llama-cli release archive.
pub struct LlamaReleaseInfo {
    pub url: String,
    pub archive_name: String,
    pub is_zip: bool,
    pub binary_name: &'static str,
}

/// Determine the download URL and archive type for the host platform.
pub fn get_llama_release_info(tag: &str) -> Option<LlamaReleaseInfo> {
    if cfg!(target_os = "windows") && cfg!(target_arch = "x86_64") {
        let archive_name = format!("llama-{tag}-bin-win-cpu-x64.zip");
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{archive_name}");
        Some(LlamaReleaseInfo {
            url,
            archive_name,
            is_zip: true,
            binary_name: "llama-cli.exe",
        })
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        let archive_name = format!("llama-{tag}-bin-ubuntu-x64.tar.gz");
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{archive_name}");
        Some(LlamaReleaseInfo {
            url,
            archive_name,
            is_zip: false,
            binary_name: "llama-cli",
        })
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        let archive_name = format!("llama-{tag}-bin-macos-arm64.tar.gz");
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{archive_name}");
        Some(LlamaReleaseInfo {
            url,
            archive_name,
            is_zip: false,
            binary_name: "llama-cli",
        })
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "x86_64") {
        let archive_name = format!("llama-{tag}-bin-macos-x64.tar.gz");
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{archive_name}");
        Some(LlamaReleaseInfo {
            url,
            archive_name,
            is_zip: false,
            binary_name: "llama-cli",
        })
    } else {
        None
    }
}

/// Load configuration from disk, or create a default one if it doesn't exist.
pub fn load_config() -> Result<TsugiConfig, TsugiError> {
    let config_path = config_dir().join("config.json");

    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path).map_err(|e| {
            TsugiError::ConfigError(format!("Failed to read config: {e}"))
        })?;
        let config: TsugiConfig = serde_json::from_str(&content).map_err(|e| {
            TsugiError::ConfigError(format!("Failed to parse config: {e}"))
        })?;
        Ok(config)
    } else {
        let config = TsugiConfig::default();
        save_config(&config)?;
        Ok(config)
    }
}

/// Save configuration to disk.
pub fn save_config(config: &TsugiConfig) -> Result<(), TsugiError> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir).map_err(|e| {
        TsugiError::ConfigError(format!("Failed to create config directory: {e}"))
    })?;

    let config_path = dir.join("config.json");
    let content = serde_json::to_string_pretty(config).map_err(|e| {
        TsugiError::ConfigError(format!("Failed to serialize config: {e}"))
    })?;
    std::fs::write(&config_path, content).map_err(|e| {
        TsugiError::ConfigError(format!("Failed to write config: {e}"))
    })?;

    info!("Configuration saved to: {}", config_path.display());
    Ok(())
}

/// Check if a local llama-server instance is actively listening.
/// Returns false immediately if no URL is explicitly configured.
pub fn is_llama_server_running(url: Option<&str>) -> bool {
    let url_str = match url {
        Some(u) if !u.trim().is_empty() => u.trim(),
        _ => return false,
    };
    let trimmed = url_str.trim_start_matches("http://").trim_start_matches("https://");
    let host_port = match trimmed.split('/').next() {
        Some(hp) if !hp.is_empty() => hp,
        _ => return false,
    };
    if let Ok(addr) = host_port.parse() {
        std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(150)).is_ok()
    } else {
        false
    }
}

/// Check if the required model files exist and the tool is ready to use.
pub fn check_setup() -> SetupStatus {
    let config = match load_config() {
        Ok(c) => c,
        Err(_) => return SetupStatus::NoConfig,
    };

    let server_active = is_llama_server_running(config.llama_server_url.as_deref());

    let has_primary = config
        .primary_model_path
        .as_ref()
        .map(|p| p.exists())
        .unwrap_or(false);

    let has_fallback = config
        .fallback_model_path
        .as_ref()
        .map(|p| p.exists())
        .unwrap_or(false);

    let has_llama_cli = crate::engine::find_llama_cli().is_ok();

    let has_any_model = has_primary || has_fallback;

    if server_active || (has_any_model && has_llama_cli) {
        SetupStatus::Ready {
            primary_model: config.primary_model_path.clone(),
            fallback_model: config.fallback_model_path.clone(),
            has_llama_cli,
            server_endpoint: if server_active {
                config.llama_server_url.clone()
            } else {
                None
            },
        }
    } else if has_any_model && !has_llama_cli {
        SetupStatus::MissingLlamaCli
    } else {
        SetupStatus::MissingModel
    }
}

/// The current setup status of the tool.
#[derive(Debug)]
pub enum SetupStatus {
    /// Everything is configured and ready.
    Ready {
        primary_model: Option<PathBuf>,
        fallback_model: Option<PathBuf>,
        has_llama_cli: bool,
        server_endpoint: Option<String>,
    },
    /// No configuration file found.
    NoConfig,
    /// Model files not found.
    MissingModel,
    /// llama-cli not found on PATH.
    MissingLlamaCli,
}
