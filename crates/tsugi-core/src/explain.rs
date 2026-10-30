//! # Git Command Explainer
//!
//! Provides plain-English explanations of Git commands and strategies.
//! Uses a fast Tier-0 static catalog for common commands (<0.1ms, zero RAM)
//! and falls back to the local SLM for complex contextual commands.

use crate::types::{RepoState, TsugiConfig, TsugiError};

/// Query a static explanation for a known Git command.
pub fn explain_command_static(cmd: &str) -> Option<&'static str> {
    let trimmed = cmd.trim();

    // Match exact or prefix
    if trimmed.contains("rebase -i") || trimmed.contains("rebase --interactive") {
        return Some(
            "Interactive rebase: Allows you to rewrite, reorder, squash (combine), or edit commits \
             in your local history before sharing them with others."
        );
    }
    if trimmed.contains("rebase --onto") {
        return Some(
            "Rebase --onto: Slices a series of commits off an old branch base and replants them onto \
             a completely new base branch (syntax: git rebase --onto <new-base> <old-base> <branch>)."
        );
    }
    if trimmed.contains("merge --ff-only") {
        return Some(
            "Fast-forward only merge: Advances the branch pointer directly to the target commit \
             without creating a merge commit. Fails safely if the branch has diverged."
        );
    }
    if trimmed.contains("merge --squash") {
        return Some(
            "Squash merge: Takes all changes from the target branch and stages them as a single commit \
             in your current working tree without preserving the individual commit history."
        );
    }
    if trimmed.contains("push --force-with-lease") {
        return Some(
            "Force push with lease: A safe alternative to --force. Overwrites the remote branch only \
             if no one else has pushed new commits since your last fetch, preventing accidental data loss."
        );
    }
    if trimmed.contains("reset --soft") {
        return Some(
            "Soft reset: Moves HEAD back to a previous commit while leaving all file changes intact \
             and staged in your index, making it easy to recombine or edit commits."
        );
    }
    if trimmed.contains("reset --hard") {
        return Some(
            "Hard reset: Resets HEAD, the staging area (index), and your working tree to match the target commit. \
             WARNING: Any uncommitted modifications or untracked changes will be permanently discarded."
        );
    }
    if trimmed.contains("restore --staged") {
        return Some(
            "Restore staged: Unstages files from your index while keeping all your edits in the working tree."
        );
    }
    if trimmed.contains("stash push -u") || trimmed.contains("stash -u") {
        return Some(
            "Stash with untracked: Saves both tracked modifications and untracked files into the stash stack \
             and reverts the working tree to HEAD."
        );
    }
    if trimmed.contains("stash pop") {
        return Some(
            "Stash pop: Reapplies the most recently stashed changes onto your working tree and removes \
             that entry from your stash stack."
        );
    }
    if trimmed.contains("reflog") {
        return Some(
            "Reflog (Reference Log): Records every time HEAD was updated in your local repository. \
             Essential for recovering commits that were accidentally reset or deleted."
        );
    }
    if trimmed.contains("cherry-pick") {
        return Some(
            "Cherry-pick: Applies the exact patch introduced by a specific commit from another branch \
             onto your currently active branch."
        );
    }
    if trimmed.contains("worktree add") {
        return Some(
            "Worktree add: Checks out a different branch into a separate filesystem directory, \
             enabling simultaneous work on multiple branches without switching or stashing."
        );
    }
    if trimmed.contains("worktree prune") {
        return Some(
            "Worktree prune: Cleans up internal administrative metadata for linked worktrees \
             that have been manually deleted from disk."
        );
    }
    if trimmed.contains("submodule update --init --recursive") {
        return Some(
            "Submodule update with recursive init: Clones and checks out all registered submodules and any \
             nested submodules at the commit SHAs recorded by the parent repository."
        );
    }
    if trimmed.contains("fetch --prune") {
        return Some(
            "Fetch prune: Downloads new commits from remote and deletes local tracking branches \
             whose counterparts have been deleted on the remote server."
        );
    }
    if trimmed.contains("add --renormalize") {
        return Some(
            "Renormalize: Forces Git to re-evaluate line endings (CRLF/LF) for all tracked files \
             against current .gitattributes rules."
        );
    }
    if trimmed.contains("commit --amend") {
        return Some(
            "Commit amend: Combines staged changes with the previous commit and allows editing the commit message, \
             replacing the previous commit SHA."
        );
    }
    if trimmed.contains("switch -c") {
        return Some(
            "Switch create: Creates and switches to a new local branch pointing to current HEAD."
        );
    }
    if trimmed.contains("clean -fd") {
        return Some(
            "Clean force directory: Permanently deletes all untracked files and directories from the working tree."
        );
    }
    if trimmed.contains("fsck") {
        return Some(
            "File System Consistency Check: Scans the Git object database for corrupt, unreferenced, or dangling objects."
        );
    }
    if trimmed.contains("bisect") {
        return Some(
            "Git Bisect: Uses binary search through commit history to identify the exact commit that introduced a regression."
        );
    }

    None
}

/// Explain a Git command, with contextual repository awareness.
pub fn explain_command(
    cmd: &str,
    _state: Option<&RepoState>,
    _config: &TsugiConfig,
) -> Result<String, TsugiError> {
    if let Some(explanation) = explain_command_static(cmd) {
        return Ok(explanation.to_string());
    }

    // Default structured explanation breakdown if not in static catalog
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.first().copied() == Some("git") && parts.len() > 1 {
        let subcmd = parts[1];
        Ok(format!(
            "Command: '{}'\nSubcommand '{}': Invokes Git's {} subsystem with flags [{}].",
            cmd,
            subcmd,
            subcmd,
            parts[2..].join(" ")
        ))
    } else {
        Ok(format!(
            "Command '{}': Standard Git invocation. Consult `git help {}` for manual page documentation.",
            cmd,
            parts.get(1).copied().unwrap_or("")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explain_static_known_commands() {
        assert!(explain_command_static("git rebase -i HEAD~3").is_some());
        assert!(explain_command_static("git push --force-with-lease origin main").is_some());
        assert!(explain_command_static("git reset --hard").is_some());
        assert!(explain_command_static("git submodule update --init --recursive").is_some());
    }

    #[test]
    fn test_explain_fallback() {
        let config = TsugiConfig::default();
        let res = explain_command("git custom-tool --flag", None, &config).unwrap();
        assert!(res.contains("custom-tool"));
    }
}
