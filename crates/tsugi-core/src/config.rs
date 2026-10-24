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
