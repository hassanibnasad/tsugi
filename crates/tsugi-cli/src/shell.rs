//! # Shell Integration Subsystem
//!
//! Generates shell wrappers for PowerShell, Bash, Zsh, and Fish so that
//! whenever any native `git` command fails with a non-zero exit code,
//! a subtle hint is displayed reminding the developer to run `tsugi`.

use anyhow::{bail, Result};

pub fn print_shell_init(shell_name: &str) -> Result<()> {
    match shell_name.to_lowercase().as_str() {
        "powershell" | "pwsh" => {
            print!(
                "{}",
                r#"# tsugi shell integration for PowerShell
# Add this to your PowerShell profile ($PROFILE):
#   tsugi init powershell | Out-String | Invoke-Expression

function git {
    & (Get-Command -CommandType Application git) @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "💡 Git command failed. Type 'tsugi' to diagnose and resolve." -ForegroundColor Yellow
    }
}
"#
            );
        }
        "bash" => {
            print!(
                "{}",
                r#"# tsugi shell integration for Bash
# Add this to your ~/.bashrc:
#   eval "$(tsugi init bash)"

git() {
    command git "$@"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        printf "\033[33m💡 Git command failed. Type 'tsugi' to diagnose and resolve.\033[0m\n"
    fi
    return $exit_code
}
"#
            );
        }
        "zsh" => {
            print!(
                "{}",
                r#"# tsugi shell integration for Zsh
# Add this to your ~/.zshrc:
#   eval "$(tsugi init zsh)"

git() {
    command git "$@"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        printf "\033[33m💡 Git command failed. Type 'tsugi' to diagnose and resolve.\033[0m\n"
    fi
    return $exit_code
}
"#
            );
        }
        "fish" => {
            print!(
                "{}",
                r#"# tsugi shell integration for Fish
# Add this to ~/.config/fish/config.fish:
#   tsugi init fish | source

function git --wraps git
    command git $argv
    set -l exit_code $status
    if test $exit_code -ne 0
        set_color yellow
        echo "💡 Git command failed. Type 'tsugi' to diagnose and resolve."
        set_color normal
    end
    return $exit_code
end
"#
            );
        }
        other => {
            bail!("Unsupported shell: {other}. Supported shells are: powershell, bash, zsh, fish.");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_shell_init_supported_shells() {
        assert!(print_shell_init("powershell").is_ok());
        assert!(print_shell_init("pwsh").is_ok());
        assert!(print_shell_init("bash").is_ok());
        assert!(print_shell_init("zsh").is_ok());
        assert!(print_shell_init("fish").is_ok());
    }

    #[test]
    fn test_print_shell_init_unsupported() {
        assert!(print_shell_init("cmd").is_err());
        assert!(print_shell_init("tcsh").is_err());
    }
}
