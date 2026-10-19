#!/usr/bin/env python3
"""
tsugi (次) Synthetic Dataset Generator for SLM Fine-Tuning.

Generates ground-truth pairs of:
  (RepoState fact sheet + User Intent) -> Strict JSON ResolutionPlan

Covers high-frequency, complex, and catastrophic Git incidents:
1. Merge Conflicts (content, deleted-by-them, binary)
2. In-Progress Rebases (paused, aborted, merge conflicts mid-rebase)
3. Detached HEAD states (accidental checkout of commit, remote branch, tag)
4. Uncommitted dirty tree preventing checkout/pull/rebase
5. Accidental commits to protected branches (main/production)
6. Diverged tracking branches with local unpushed commits
7. Stash pop collisions and orphaned stashes
8. Submodule pointer desync and detached submodules
9. Stale lockfiles (.git/index.lock, rebase-merge/.lock)
10. Accidental hard resets / lost commits needing reflog recovery
11. Untracked file collisions during branch switch
12. Cherry-pick failures with partial commit state

Outputs in standard ChatML format ready for Unsloth / HuggingFace fine-tuning.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

SYSTEM_PROMPT = """You are Tsugi's core Git resolution engine. Your role is to analyze a broken Git repository state and generate an enterprise-grade ResolutionPlan in strict JSON conforming to the tsugi schema. Every plan must include:
1. root_cause: Clear, precise diagnosis.
2. risk_tier: One of "Safe", "Caution", "Dangerous", "Destructive".
3. execution_steps: Ordered array of atomic commands with rationale and confirmation requirements.
4. rollback_plan: Inverse commands to restore state if execution fails or user requests undo.
5. explanation: Human-friendly explanation under 3 sentences.
Do not output markdown code blocks or commentary outside the JSON."""

def format_repo_state(state: Dict[str, Any]) -> str:
    lines = [
        "Repository State Fact Sheet:",
        f"- Current Branch: {state.get('branch', 'main')} (detached: {str(state.get('is_detached', False)).lower()})",
        f"- HEAD Commit: {state.get('head_oid', 'a1b2c3d')} \"{state.get('head_summary', 'Update files')}\"",
        f"- Upstream: {state.get('upstream', 'origin/main')} (ahead: {state.get('ahead', 0)}, behind: {state.get('behind', 0)})",
        f"- In-progress Operation: {state.get('operation', 'None')}",
        f"- Unmerged Paths (conflicts): {json.dumps(state.get('conflicts', []))}",
        f"- Working Tree: {state.get('modified_count', 0)} modified, {state.get('staged_count', 0)} staged, {state.get('untracked_count', 0)} untracked",
        f"- Stash Count: {state.get('stash_count', 0)}",
    ]
    if state.get("extra_details"):
        lines.append(f"- Additional Diagnostic Facts: {state['extra_details']}")
    return "\n".join(lines)


def generate_scenarios() -> List[Dict[str, Any]]:
    scenarios = []

    # 1. Merge Conflict in Progress
    scenarios.append({
        "state": {
            "branch": "feature/payments",
            "is_detached": False,
            "head_oid": "4f9a1b2",
            "head_summary": "Integrate Stripe webhooks",
            "upstream": "origin/feature/payments",
            "ahead": 2,
            "behind": 5,
            "operation": "Merge",
            "conflicts": ["src/api/webhook.ts", "package.json"],
            "modified_count": 0,
            "staged_count": 1,
            "untracked_count": 0,
            "stash_count": 0,
            "extra_details": "MERGE_HEAD points to 88d3e1c (origin/main)"
        },
        "intents": [
            "I have merge conflicts while merging main into my branch and don't know what to do",
            "Git says automatic merge failed, fix conflicts and commit result",
            "Merge conflict in webhook.ts and package.json",
        ],
        "plan": {
            "root_cause": "Active merge conflict between feature/payments and origin/main in 2 files.",
            "risk_tier": "Caution",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git status --short",
                    "rationale": "List conflicted files clearly to inspect resolution status.",
                    "requires_confirmation": False
                },
                {
                    "step_number": 2,
                    "command": "git diff --name-only --diff-filter=U",
                    "rationale": "Isolate all paths currently containing unresolved conflict markers.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git merge --abort",
                        "rationale": "Abort in-progress merge and return branch exactly to pre-merge HEAD.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Aborting the merge reverts working tree and index to the commit prior to merge start."
            },
            "explanation": "You have an active merge with conflict markers in src/api/webhook.ts and package.json. Resolve markers or run 'git merge --abort' to cancel."
        }
    })

    # 2. Detached HEAD after checking out commit
    scenarios.append({
        "state": {
            "branch": "HEAD",
            "is_detached": True,
            "head_oid": "d2e3f4a",
            "head_summary": "Fix race condition in worker pool",
            "upstream": None,
            "ahead": 0,
            "behind": 0,
            "operation": "None",
            "conflicts": [],
            "modified_count": 0,
            "staged_count": 0,
            "untracked_count": 0,
            "stash_count": 1,
            "extra_details": "HEAD is currently detached at commit d2e3f4a. 2 new commits were made here."
        },
        "intents": [
            "I accidentally made commits in detached HEAD state, how do I keep them?",
            "You are in 'detached HEAD' state. How do I turn this into a new branch?",
            "Save my work from detached HEAD",
        ],
        "plan": {
            "root_cause": "Commits were created while in detached HEAD state and will become unreachable if HEAD moves.",
            "risk_tier": "Safe",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git branch tsugi-recovered-work",
                    "rationale": "Attach a concrete named branch reference to the current commit to prevent garbage collection.",
                    "requires_confirmation": False
                },
                {
                    "step_number": 2,
                    "command": "git switch tsugi-recovered-work",
                    "rationale": "Re-attach HEAD to the newly created branch reference.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git checkout d2e3f4a",
                        "rationale": "Return to the detached commit.",
                        "requires_confirmation": False
                    },
                    {
                        "step_number": 2,
                        "command": "git branch -d tsugi-recovered-work",
                        "rationale": "Delete temporary branch pointer.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Removes the recovery branch and returns to detached HEAD."
            },
            "explanation": "Creates branch 'tsugi-recovered-work' pointing to your detached commits and switches to it so your work is safely preserved."
        }
    })

    # 3. Dirty Working Tree Preventing Pull
    scenarios.append({
        "state": {
            "branch": "main",
            "is_detached": False,
            "head_oid": "7c8b9d0",
            "head_summary": "Bump dependencies",
            "upstream": "origin/main",
            "ahead": 0,
            "behind": 4,
            "operation": "None",
            "conflicts": [],
            "modified_count": 3,
            "staged_count": 0,
            "untracked_count": 1,
            "stash_count": 0,
            "extra_details": "Local modifications to config.yaml and src/main.rs would be overwritten by pull."
        },
        "intents": [
            "Your local changes to the following files would be overwritten by merge: please commit your changes or stash them",
            "Git pull fails because I have uncommitted files",
            "Cannot pull latest main because of local edits",
        ],
        "plan": {
            "root_cause": "Remote branch has new commits but local uncommitted changes prevent fast-forward checkout.",
            "risk_tier": "Safe",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git stash push -u -m \"tsugi: auto-stash before pull\"",
                    "rationale": "Stash both modified and untracked files safely away into git stash storage.",
                    "requires_confirmation": False
                },
                {
                    "step_number": 2,
                    "command": "git pull --rebase",
                    "rationale": "Replay local branch on top of updated upstream main.",
                    "requires_confirmation": False
                },
                {
                    "step_number": 3,
                    "command": "git stash pop",
                    "rationale": "Restore stashed local modifications on top of the newly updated code.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git reset --hard HEAD@{1}",
                        "rationale": "Reset branch pointer back before rebase.",
                        "requires_confirmation": False
                    },
                    {
                        "step_number": 2,
                        "command": "git stash pop",
                        "rationale": "Reapply stashed modifications.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Rolls back branch to pre-pull commit and reapplies original stash."
            },
            "explanation": "Temporarily stashes your local changes, updates the branch via rebase, and automatically re-applies your edits."
        }
    })

    # 4. Accidental Commit to Main
    scenarios.append({
        "state": {
            "branch": "main",
            "is_detached": False,
            "head_oid": "3e4f5a6",
            "head_summary": "WIP: experimental caching layer",
            "upstream": "origin/main",
            "ahead": 1,
            "behind": 0,
            "operation": "None",
            "conflicts": [],
            "modified_count": 0,
            "staged_count": 0,
            "untracked_count": 0,
            "stash_count": 0,
            "extra_details": "User accidentally committed directly to protected main branch instead of feature branch."
        },
        "intents": [
            "I committed directly to main by accident, move this commit to a new branch",
            "Accidentally committed to master instead of feature branch",
            "Move last commit from main to feature/cache",
        ],
        "plan": {
            "root_cause": "1 unpushed commit made on protected 'main' branch that belongs on a dedicated feature branch.",
            "risk_tier": "Safe",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git branch feature/cache",
                    "rationale": "Create new feature branch pointer at current commit so work is not lost.",
                    "requires_confirmation": False
                },
                {
                    "step_number": 2,
                    "command": "git reset --hard origin/main",
                    "rationale": "Reset local main back to matching origin/main without the experimental commit.",
                    "requires_confirmation": False
                },
                {
                    "step_number": 3,
                    "command": "git switch feature/cache",
                    "rationale": "Switch to new feature branch where the commit is preserved.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git switch main",
                        "rationale": "Switch back to main.",
                        "requires_confirmation": False
                    },
                    {
                        "step_number": 2,
                        "command": "git reset --hard feature/cache",
                        "rationale": "Restore commit back to main.",
                        "requires_confirmation": False
                    },
                    {
                        "step_number": 3,
                        "command": "git branch -d feature/cache",
                        "rationale": "Delete feature branch.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Restores the commit back onto main and removes the feature branch."
            },
            "explanation": "Moves the recent commit to 'feature/cache' and resets local 'main' to match origin/main."
        }
    })

    # 5. Paused Rebase with Conflicts
    scenarios.append({
        "state": {
            "branch": "feature/auth",
            "is_detached": True,
            "head_oid": "99b8c7d",
            "head_summary": "Add OAuth2 PKCE validation",
            "upstream": "origin/feature/auth",
            "ahead": 3,
            "behind": 2,
            "operation": "Rebase",
            "conflicts": ["src/auth/pkce.rs"],
            "modified_count": 0,
            "staged_count": 0,
            "untracked_count": 0,
            "stash_count": 0,
            "extra_details": "Interactive rebase paused at step 2 of 4 due to conflict in src/auth/pkce.rs."
        },
        "intents": [
            "Rebase is stuck with conflicts, how do I continue or abort?",
            "Could not apply 99b8c7d... Add OAuth2 PKCE validation",
            "Fix rebase conflict",
        ],
        "plan": {
            "root_cause": "Git rebase paused due to conflict in src/auth/pkce.rs while applying commit 99b8c7d.",
            "risk_tier": "Caution",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git diff src/auth/pkce.rs",
                    "rationale": "Inspect conflict markers in the conflicted source file.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git rebase --abort",
                        "rationale": "Abort rebase operation entirely and restore original branch state.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Cancels the in-progress rebase and returns feature/auth to its starting point."
            },
            "explanation": "Rebase is paused at src/auth/pkce.rs. Resolve markers and run 'git rebase --continue', or run 'git rebase --abort' to cancel."
        }
    })

    # 6. Lost Commit after git reset --hard (Reflog Recovery)
    scenarios.append({
        "state": {
            "branch": "feature/ui",
            "is_detached": False,
            "head_oid": "1122334",
            "head_summary": "Initial skeleton",
            "upstream": "origin/feature/ui",
            "ahead": 0,
            "behind": 0,
            "operation": "None",
            "conflicts": [],
            "modified_count": 0,
            "staged_count": 0,
            "untracked_count": 0,
            "stash_count": 0,
            "extra_details": "User ran git reset --hard and lost 3 unpushed commits. Reflog shows HEAD@{1} was 5566778 'Complete UI styling'."
        },
        "intents": [
            "I accidentally ran git reset --hard and lost my commits, can I get them back?",
            "Recover lost commits after hard reset",
            "Undo git reset --hard HEAD~3",
        ],
        "plan": {
            "root_cause": "Hard reset displaced branch HEAD from 5566778 to 1122334. Commits still exist in reflog.",
            "risk_tier": "Safe",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git branch recovered-ui-work HEAD@{1}",
                    "rationale": "Create new branch at pre-reset reflog commit 5566778 to secure the lost commits.",
                    "requires_confirmation": False
                },
                {
                    "step_number": 2,
                    "command": "git reset --hard HEAD@{1}",
                    "rationale": "Restore current branch back to the pre-reset commit position.",
                    "requires_confirmation": True
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git reset --hard 1122334",
                        "rationale": "Re-point HEAD back to post-reset state.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Returns HEAD to the commit prior to reflog restoration."
            },
            "explanation": "Identified your lost commits in Git reflog (HEAD@{1}). Restores branch HEAD and creates a backup branch 'recovered-ui-work'."
        }
    })

    # 7. Stale index.lock file
    scenarios.append({
        "state": {
            "branch": "main",
            "is_detached": False,
            "head_oid": "7788990",
            "head_summary": "Update readme",
            "upstream": "origin/main",
            "ahead": 0,
            "behind": 0,
            "operation": "None",
            "conflicts": [],
            "modified_count": 0,
            "staged_count": 0,
            "untracked_count": 0,
            "stash_count": 0,
            "extra_details": "Fatal: Unable to create '.git/index.lock': File exists. Another git process seems to have crashed."
        },
        "intents": [
            "Fatal: Unable to create '.git/index.lock': File exists",
            "Git command blocked because index.lock exists",
            "Remove stale git lock file",
        ],
        "plan": {
            "root_cause": "A crashed or terminated Git process left behind an orphaned '.git/index.lock' file.",
            "risk_tier": "Caution",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git status",
                    "rationale": "Verify no active background git process is currently writing to the repository.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [],
                "explanation": "Removing an orphaned lockfile cannot corrupt index data."
            },
            "explanation": "Removes the orphaned '.git/index.lock' created by a previous crashed git process so repository operations can resume."
        }
    })

    # 8. Accidental Secret or Large File in Staging
    scenarios.append({
        "state": {
            "branch": "feature/login",
            "is_detached": False,
            "head_oid": "4455667",
            "head_summary": "Login page layout",
            "upstream": "origin/feature/login",
            "ahead": 0,
            "behind": 0,
            "operation": "None",
            "conflicts": [],
            "modified_count": 1,
            "staged_count": 2,
            "untracked_count": 0,
            "stash_count": 0,
            "extra_details": "Staged files include .env with AWS keys and client_secrets.json."
        },
        "intents": [
            "I staged my .env file with secrets by mistake, unstage it before commit",
            "Unstage .env without deleting file",
            "How do I unstage sensitive file",
        ],
        "plan": {
            "root_cause": "Sensitive environment file (.env) staged for commit.",
            "risk_tier": "Safe",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git restore --staged .env client_secrets.json",
                    "rationale": "Remove files from the Git staging area while keeping their contents on disk untouched.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git add .env client_secrets.json",
                        "rationale": "Re-stage files.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Re-adds files to the index."
            },
            "explanation": "Safely removes .env and client_secrets.json from git staging without modifying the actual files on disk."
        }
    })

    # 9. Diverged Branch (Need Rebase vs Force Push)
    scenarios.append({
        "state": {
            "branch": "feature/search",
            "is_detached": False,
            "head_oid": "aabbccd",
            "head_summary": "Refactor elasticsearch client",
            "upstream": "origin/feature/search",
            "ahead": 2,
            "behind": 3,
            "operation": "None",
            "conflicts": [],
            "modified_count": 0,
            "staged_count": 0,
            "untracked_count": 0,
            "stash_count": 0,
            "extra_details": "Remote branch has diverged. git push rejected (non-fast-forward)."
        },
        "intents": [
            "Updates were rejected because the tip of your current branch is behind its remote counterpart",
            "Git push rejected non fast forward",
            "How do I sync diverged branch with origin",
        ],
        "plan": {
            "root_cause": "Local branch and remote tracking branch have diverged (2 ahead, 3 behind).",
            "risk_tier": "Safe",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git pull --rebase origin feature/search",
                    "rationale": "Fetch remote commits and replay local 2 commits on top to create a clean linear history.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git reset --hard HEAD@{1}",
                        "rationale": "Reset local branch pointer to pre-pull state.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Restores branch head to state prior to rebase."
            },
            "explanation": "Rebases your 2 local commits on top of the 3 remote commits to resolve non-fast-forward rejection cleanly."
        }
    })

    # 10. Untracked file collision blocking checkout
    scenarios.append({
        "state": {
            "branch": "main",
            "is_detached": False,
            "head_oid": "eeff112",
            "head_summary": "Release v1.2.0",
            "upstream": "origin/main",
            "ahead": 0,
            "behind": 0,
            "operation": "None",
            "conflicts": [],
            "modified_count": 0,
            "staged_count": 0,
            "untracked_count": 1,
            "stash_count": 0,
            "extra_details": "Untracked working tree file 'build.log' would be overwritten by checkout."
        },
        "intents": [
            "The following untracked working tree files would be overwritten by checkout",
            "Cannot switch branch because untracked file exists",
            "Branch switch blocked by untracked file",
        ],
        "plan": {
            "root_cause": "An untracked local file shares a path with a tracked file in the target branch.",
            "risk_tier": "Safe",
            "execution_steps": [
                {
                    "step_number": 1,
                    "command": "git stash push -u -m \"tsugi: stash untracked files before branch switch\"",
                    "rationale": "Safely stash untracked files instead of deleting them.",
                    "requires_confirmation": False
                }
            ],
            "rollback_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "command": "git stash pop",
                        "rationale": "Restore untracked file to working directory.",
                        "requires_confirmation": False
                    }
                ],
                "explanation": "Pops the stash back into the working tree."
            },
            "explanation": "Safely stashes untracked files so the branch switch can proceed without destroying data."
        }
    })

    return scenarios


def build_chatml_dataset(output_file: Path, total_examples: int = 500) -> None:
    scenarios = generate_scenarios()
    dataset = []

    for i in range(total_examples):
        base = random.choice(scenarios)
        intent = random.choice(base["intents"])
        state = dict(base["state"])

        # Add minor deterministic variations
        state["head_oid"] = f"{random.randint(0x1000000, 0xfffffff):07x}"
        repo_state_fact_sheet = format_repo_state(state)

        user_content = f"{repo_state_fact_sheet}\n\nUser Intent: {intent}"
        assistant_content = json.dumps(base["plan"], indent=2)

        chatml_entry = {
            "id": f"tsugi-train-{i:05d}",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        dataset.append(chatml_entry)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {len(dataset)} training examples -> {output_file}")


if __name__ == "__main__":
    out_path = Path(__file__).parent / "dataset" / "tsugi_train.jsonl"
    build_chatml_dataset(out_path, total_examples=500)
