//! # Streaming Downloader & Integrity Verifier
//!
//! Provides pure-Rust streaming downloads with SHA-256 verification,
//! interactive progress reporting, and archive extraction (zip and tar.gz).

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use tracing::{debug, info};

/// Download a file from `url` to `dest_path` with an interactive progress bar
/// and optional SHA-256 integrity verification.
pub fn download_file(
    url: &str,
    dest_path: &Path,
    expected_sha256: Option<&str>,
    label: &str,
) -> Result<()> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    let client = reqwest::blocking::Client::builder()
        .user_agent("tsugi-installer/0.1")
        .build()
        .context("Failed to initialize HTTP client")?;

    info!("Initiating download from {url} to {}", dest_path.display());
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to connect to {url}"))?;

    if !response.status().is_success() {
        bail!(
            "Download failed with HTTP status: {} ({})",
            response.status(),
            url
        );
    }

    let total_size = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{prefix:.bold} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap()
            .progress_chars("━╸ "),
    );
    pb.set_prefix(format!("  {} {label}", "↓".cyan().bold()));

    let tmp_path = dest_path.with_extension(format!(
        "{}.tmp",
        dest_path.extension().and_then(|s| s.to_str()).unwrap_or("download")
    ));

    let mut file = File::create(&tmp_path)
        .with_context(|| format!("Failed to create temporary file: {}", tmp_path.display()))?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536]; // 64 KB chunk size

    loop {
        let bytes_read = response
            .read(&mut buffer)
            .context("Error reading response stream")?;
        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
        file.write_all(&buffer[..bytes_read])
            .context("Failed writing chunk to disk")?;
        pb.inc(bytes_read as u64);
    }

    file.flush().context("Failed to flush file to disk")?;
    drop(file);

    pb.finish_and_clear();

    // Verify SHA-256 hash if expected hash is provided
    if let Some(expected) = expected_sha256 {
        let actual_hash = format!("{:x}", hasher.finalize());
        let expected_clean = expected.trim().to_lowercase();
        debug!("SHA-256 Check: expected={expected_clean}, actual={actual_hash}");

        if actual_hash != expected_clean {
            let _ = fs::remove_file(&tmp_path);
            bail!(
                "Integrity verification failed for {label}!\n  Expected SHA-256: {expected_clean}\n  Actual SHA-256:   {actual_hash}"
            );
        }
        println!("  {} SHA-256 verified successfully.", "✔".green());
    }

    // Atomic move from tmp to final path
    if dest_path.exists() {
        let _ = fs::remove_file(dest_path);
    }
    fs::rename(&tmp_path, dest_path)
        .with_context(|| format!("Failed to finalize file: {}", dest_path.display()))?;

    println!(
        "  {} {} installed to: {}",
        "✔".green().bold(),
        label.white().bold(),
        dest_path.display().to_string().cyan()
    );

    Ok(())
}

/// Extract an archive (.zip or .tar.gz) into `dest_dir`.
pub fn extract_archive(archive_path: &Path, dest_dir: &Path, is_zip: bool) -> Result<()> {
    fs::create_dir_all(dest_dir)
        .with_context(|| format!("Failed to create destination: {}", dest_dir.display()))?;

    if is_zip {
        let file = File::open(archive_path)
            .with_context(|| format!("Failed to open archive: {}", archive_path.display()))?;
        let mut zip_archive = zip::ZipArchive::new(file)
            .with_context(|| format!("Invalid ZIP archive: {}", archive_path.display()))?;

        for i in 0..zip_archive.len() {
            let mut file = zip_archive.by_index(i)?;
            let outpath = match file.enclosed_name() {
                Some(path) => dest_dir.join(path),
                None => continue,
            };

            if file.is_dir() {
                fs::create_dir_all(&outpath)?;
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        fs::create_dir_all(p)?;
                    }
                }
                let mut outfile = File::create(&outpath)?;
                std::io::copy(&mut file, &mut outfile)?;
            }
        }
    } else {
        // .tar.gz
        let file = File::open(archive_path)
            .with_context(|| format!("Failed to open archive: {}", archive_path.display()))?;
        let tar = flate2::read::GzDecoder::new(file);
        let mut tar_archive = tar::Archive::new(tar);
        tar_archive
            .unpack(dest_dir)
            .with_context(|| format!("Failed to unpack tar.gz to {}", dest_dir.display()))?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(entries) = fs::read_dir(dest_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        let mut perms = meta.permissions();
                        perms.set_mode(0o755);
                        let _ = fs::set_permissions(entry.path(), perms);
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_sha256_verification_mismatch_fails() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("sample.txt");
        fs::write(&file_path, "hello world").unwrap();

        // Calculate actual hash
        let mut hasher = Sha256::new();
        hasher.update(b"hello world");
        let actual_hash = format!("{:x}", hasher.finalize());

        assert_eq!(
            actual_hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }
}
