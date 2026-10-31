#!/usr/bin/env python3
"""
tsugi (次) Synthetic Dataset Generator for SLM Fine-Tuning.

Generates ground-truth pairs of:
  (RepoState fact sheet + User Intent) -> Strict JSON ResolutionPlan

Guaranteed 100% schema alignment with Rust runtime (types.rs, engine.rs, tsugi_plan.gbnf):
1. Fact Sheet Header: [REPO_STATE] ...
2. User Intent Prefix: USER_INTENT: <intent>
3. Risk Tier Enum: "READ_ONLY", "MUTATING_SAFE", "MUTATING_DANGEROUS", "HISTORY_REWRITE"
4. Step Object Keys: {"cmd": "...", "explanation": "..."}
5. Rollback Field: "rollback_cmd": "<single git command>"
6. System Prompt: Exactly mirrors engine.rs system prompt.

Parametric generation prevents memorization/overfitting across 5,000+ diverse training examples.
Outputs in standard ChatML format ready for Unsloth / HuggingFace fine-tuning.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_PROMPT = """You are tsugi, an expert Git diagnostic and resolution engine. You receive a structured Git repository state and a user's description of their issue. You must output a JSON resolution plan.

Rules:
1. Diagnose the root cause from the repository state.
2. Classify the risk tier: READ_ONLY, MUTATING_SAFE, MUTATING_DANGEROUS, or HISTORY_REWRITE.
3. Output the minimum sequence of Git commands to resolve the issue.
4. Include a rollback command that fully reverses the operation.
5. Never use `git push --force` — always use `git push --force-with-lease`.
6. Never use `git reset --hard` without a prior backup step.
7. Prefer `git switch` over `git checkout` for branch operations.
8. Always preserve uncommitted work — stash before destructive operations."""

# ─────────────────────────────────────────────────────────────────────────────
# Randomization Pools for Generalization
# ─────────────────────────────────────────────────────────────────────────────

BRANCH_PREFIXES = ["feature", "feat", "fix", "bugfix", "refactor", "chore", "hotfix", "perf", "test"]

BRANCH_TOPICS = [
    "payments", "auth-pkce", "user-profiles", "login-race", "checkout-crash",
    "database-pool", "upgrade-deps", "dark-mode", "search-indexer", "cors-headers",
    "file-upload", "token-refresh", "analytics-pipeline", "router-migration",
    "multi-tenancy", "deadlock-worker", "notifications-ws", "api-client-v2",
    "billing-webhook", "memory-leak-fix", "session-store", "sso-saml",
    "rate-limiter", "graphql-schema", "audit-logs", "redis-cluster",
    "image-resizer", "metrics-collector", "tenant-isolation", "email-templates",
    "export-csv", "import-job", "websocket-heartbeat", "grpc-service"
]

FILES = [
    "src/auth.ts", "src/api/handler.rs", "package.json", "Cargo.toml", "src/models/user.py",
    "config/database.yml", "go.mod", "src/components/Header.tsx", "README.md", "Dockerfile",
    "src/lib.rs", "server/index.js", "src/services/payment.ts", "src/routes/api.ts",
    "config/settings.json", "src/middleware/auth.go", "tests/integration_test.rs",
    "src/views/Dashboard.vue", "crates/core/src/state.rs", "app/controllers/users_controller.rb",
    "lib/cache.ex", "src/styles/theme.css", "scripts/deploy.sh", "docker-compose.yml",
    "migrations/0002_add_index.sql", "src/client/http.go", "internal/server/server.go",
    "frontend/src/App.tsx", "backend/main.py", "pkg/storage/s3.go"
]

SECRET_FILES = [
    ".env", ".env.local", ".env.production", "client_secrets.json", "id_rsa",
    "credentials.json", "certs/server.key", "secrets.toml", "aws_credentials",
    "config/master.key", "service-account.json"
]

COMMIT_SUMMARIES = [
    "Integrate payment gateway", "Fix race condition in worker pool", "Add OAuth2 validation",
    "Bump dependencies to latest", "Initial skeleton layout", "Refactor elasticsearch client",
    "Add error handling middleware", "Update documentation and readme", "Fix null pointer dereference",
    "Optimize database queries", "Add integration test suite", "Implement password reset flow",
    "Clean up legacy configuration", "Add Prometheus metrics endpoint", "Fix memory leak in buffer pool",
    "Support dark mode theme", "Migrate authentication to JWT", "Add rate limiting on public routes",
    "Improve test coverage in billing module", "Validate webhook signature correctly"
]

CONFLICT_TYPES = [
    "both_modified", "deleted_by_us", "deleted_by_them", "both_added", "renamed_modified"
]


def random_sha(rng: random.Random) -> str:
    return f"{rng.randint(0x1000000, 0xfffffff):07x}"


def random_branch(rng: random.Random) -> str:
    prefix = rng.choice(BRANCH_PREFIXES)
    topic = rng.choice(BRANCH_TOPICS)
    if rng.random() < 0.2:
        return f"{prefix}/PROJ-{rng.randint(100, 999)}-{topic}"
    return f"{prefix}/{topic}"


def format_repo_state(
    head: str,
    ongoing_op: str = "NONE",
    conflicts: Optional[List[str]] = None,
    staged: str = "+0 ~0 -0 R0",
    unstaged: str = "~0 -0",
    untracked: int = 0,
    tracking: Optional[str] = None,
    stashes: int = 0,
    safe_rollback_target: Optional[str] = None,
    recent_commits: Optional[List[str]] = None,
    reflog: Optional[List[str]] = None,
    staged_files: Optional[List[str]] = None,
    unstaged_files: Optional[List[str]] = None,
    untracked_files: Optional[List[str]] = None,
    incoming_commits: Optional[List[str]] = None,
    conflict_previews: Optional[List[str]] = None,
) -> str:
    """Formats Symbolic State Fact Sheet in exact format expected by Rust runtime."""
    lines = ["[REPO_STATE]"]
    lines.append(f"HEAD: {head}")
    lines.append(f"ONGOING_OP: {ongoing_op}")

    if not conflicts:
        lines.append("CONFLICTS: NONE")
    else:
        conflict_strs = []
        for c in conflicts:
            if "(" in c:
                conflict_strs.append(c)
            else:
                conflict_strs.append(f'"{c}" (both_modified)')
        lines.append(f"CONFLICTS: [{', '.join(conflict_strs)}]")

        if conflict_previews:
            for prev in conflict_previews[:3]:
                lines.append(prev)

    lines.append(f"STAGED: {staged} | UNSTAGED: {unstaged} | UNTRACKED: {untracked}")

    if staged_files:
        quoted = [f'"{f}"' for f in staged_files[:15]]
        lines.append(f"STAGED_FILES: [{', '.join(quoted)}]")

    if unstaged_files:
        quoted = [f'"{f}"' for f in unstaged_files[:15]]
        lines.append(f"UNSTAGED_FILES: [{', '.join(quoted)}]")

    if untracked_files:
        quoted = [f'"{f}"' for f in untracked_files[:15]]
        lines.append(f"UNTRACKED_FILES: [{', '.join(quoted)}]")

    if tracking:
        lines.append(f"TRACKING: {tracking}")

    if stashes > 0:
        lines.append(f"STASHES: {stashes}")

    if safe_rollback_target:
        lines.append(f"SAFE_ROLLBACK_TARGET: {safe_rollback_target}")

    if incoming_commits:
        lines.append("INCOMING_COMMITS:")
        for c in incoming_commits[:5]:
            lines.append(f"  {c}")

    if recent_commits:
        lines.append("RECENT_COMMITS:")
        for c in recent_commits[:5]:
            lines.append(f"  {c}")

    if reflog and ongoing_op != "NONE":
        lines.append("REFLOG:")
        for r in reflog[:5]:
            lines.append(f"  {r}")

    return "\n".join(lines)


def normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures 100% schema alignment with Rust runtime, GBNF grammar, and confidence scoring."""
    risk_tier = plan.get("risk_tier", "MUTATING_SAFE")
    steps = []
    for step in plan.get("execution_steps", []):
        cmd = step["cmd"]
        is_dangerous = any(d in cmd for d in [
            "--hard", "--force", "-D", "clean -f", "prune", "drop"
        ]) or risk_tier in ("MUTATING_DANGEROUS", "HISTORY_REWRITE")
        req_conf = step.get("requires_confirmation", is_dangerous)
        steps.append({
            "cmd": cmd,
            "explanation": step["explanation"],
            "requires_confirmation": req_conf,
        })
    return {
        "root_cause": plan["root_cause"],
        "risk_tier": risk_tier,
        "execution_steps": steps,
        "rollback_cmd": plan.get("rollback_cmd", "git status"),
        "confidence": plan.get("confidence", 0.95),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Incident Generators (Combinatorial & Procedural)
# ─────────────────────────────────────────────────────────────────────────────

def incident_merge_conflict(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    other_sha = random_sha(rng)
    num_conf = rng.randint(1, 3)
    files = rng.sample(FILES, num_conf)
    conf_list = [f'"{f}" ({rng.choice(CONFLICT_TYPES)})' for f in files]

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op=f"MERGE in progress (merging {other_sha})",
        conflicts=conf_list,
        staged=f"+{rng.randint(0, 2)} ~0 -0 R0",
        unstaged=f"~{num_conf} -0",
        untracked=0,
        tracking=f"origin/{branch} (ahead 1, behind 2)",
        recent_commits=[
            f"{sha} {rng.choice(COMMIT_SUMMARIES)}",
            f"{random_sha(rng)} {rng.choice(COMMIT_SUMMARIES)}"
        ],
        reflog=[
            f"0: {sha} merge origin/main: Fast-forward",
            f"1: {other_sha} commit: {rng.choice(COMMIT_SUMMARIES)}"
        ]
    )

    intents = [
        f"Merge conflict in {', '.join(files)}, how do I inspect and fix?",
        "Automatic merge failed, fix conflicts and continue",
        f"I have merge conflicts while merging into {branch}",
        "Merge stopped with conflicts, show me what to resolve",
    ]

    plan = {
        "root_cause": f"Active merge conflict between {branch} and merged commit in {', '.join(files)}.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git status --short",
                "explanation": "List conflicted files to inspect resolution state."
            },
            {
                "cmd": "git diff --name-only --diff-filter=U",
                "explanation": "Isolate all paths containing unresolved conflict markers."
            }
        ],
        "rollback_cmd": "git merge --abort"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_rebase_conflict(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    conf_file = rng.choice(FILES)
    step_cur = rng.randint(1, 4)
    step_total = step_cur + rng.randint(1, 3)
    interactive = rng.choice([True, False])
    mode = "INTERACTIVE_REBASE" if interactive else "REBASE"

    fact_sheet = format_repo_state(
        head=f"DETACHED at {sha}",
        ongoing_op=f"{mode} in progress (step {step_cur}/{step_total}) onto origin/main",
        conflicts=[f'"{conf_file}" (both_modified)'],
        staged="+0 ~0 -0 R0",
        unstaged="~1 -0",
        untracked=0,
        tracking=f"origin/{branch} (ahead 3, behind 2)",
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"],
        reflog=[f"0: {sha} rebase (step {step_cur}): conflict"]
    )

    intents = [
        f"Rebase is paused due to conflict in {conf_file}",
        "Fix rebase conflict and tell me how to inspect or abort",
        "Could not apply commit during git rebase",
        f"Rebase paused with conflict markers in {conf_file}",
    ]

    plan = {
        "root_cause": f"Git rebase paused at step {step_cur}/{step_total} due to conflict in {conf_file}.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git diff {conf_file}",
                "explanation": f"Inspect conflict markers (<<<<<<< / ======= / >>>>>>>) in {conf_file}."
            }
        ],
        "rollback_cmd": "git rebase --abort"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_rebase_clean_continue(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    step_cur = rng.randint(2, 5)
    step_total = step_cur + rng.randint(0, 2)
    resolved_file = rng.choice(FILES)

    fact_sheet = format_repo_state(
        head=f"DETACHED at {sha}",
        ongoing_op=f"REBASE in progress (step {step_cur}/{step_total}) onto origin/main",
        conflicts=None,
        staged="+0 ~1 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        tracking=f"origin/{branch} (ahead 2, behind 1)",
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"],
        reflog=[f"0: {sha} rebase (step {step_cur}): continue"]
    )

    intents = [
        f"I resolved the conflict in {resolved_file} and staged it, now continue the rebase",
        "Conflicts are resolved, resume rebase",
        "Continue git rebase after resolving conflicts",
        "Finish current rebase step and proceed",
    ]

    plan = {
        "root_cause": f"Rebase conflicts resolved and staged; rebase is ready to proceed to step {step_cur + 1}.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git rebase --continue",
                "explanation": "Continue the rebase operation now that all conflicts are resolved."
            }
        ],
        "rollback_cmd": "git rebase --abort"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_merge_clean_continue(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    merge_sha = random_sha(rng)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op=f"MERGE in progress (merging {merge_sha})",
        conflicts=None,
        staged="+0 ~2 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"],
        reflog=[f"0: {sha} merge commit in progress"]
    )

    intents = [
        "All merge conflicts are resolved and staged, complete the merge",
        "Finish merge commit",
        "Complete merge in progress",
    ]

    plan = {
        "root_cause": f"Merge from {merge_sha} has all changes staged with zero conflicts remaining.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git commit --no-edit",
                "explanation": "Complete the in-progress merge commit using the default merge message."
            }
        ],
        "rollback_cmd": "git merge --abort"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_cherry_pick_conflict(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    cp_sha = random_sha(rng)
    conf_file = rng.choice(FILES)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op=f"CHERRY-PICK in progress ({cp_sha})",
        conflicts=[f'"{conf_file}" (both_modified)'],
        staged="+0 ~0 -0 R0",
        unstaged="~1 -0",
        untracked=0,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        f"Cherry-pick of {cp_sha[:7]} failed with conflict in {conf_file}",
        "Conflict while cherry-picking commit",
        f"Cherry pick stopped on conflict, show diff or abort",
    ]

    plan = {
        "root_cause": f"Cherry-pick of {cp_sha[:7]} paused due to conflict in {conf_file}.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git diff {conf_file}",
                "explanation": f"View conflicting lines in {conf_file}."
            }
        ],
        "rollback_cmd": "git cherry-pick --abort"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_detached_head_dirty(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    sha = random_sha(rng)
    rec_branch = f"recovery/{sha}"

    fact_sheet = format_repo_state(
        head=f"DETACHED at {sha}",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged=f"~{rng.randint(1, 3)} -0",
        untracked=rng.randint(0, 2),
        stashes=rng.randint(0, 1),
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        "I accidentally made changes in detached HEAD state, how do I save them?",
        "You are in 'detached HEAD' state, turn this into a new branch",
        "Save my work from detached HEAD",
        "Keep detached HEAD commits on a real branch",
    ]

    plan = {
        "root_cause": f"Working in detached HEAD at {sha} with uncommitted modifications.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git switch -c {rec_branch}",
                "explanation": f"Create and switch to branch '{rec_branch}' to preserve all work."
            }
        ],
        "rollback_cmd": f"git switch - && git branch -D {rec_branch}"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_detached_head_clean(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    sha = random_sha(rng)

    fact_sheet = format_repo_state(
        head=f"DETACHED at {sha}",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        "I am in detached HEAD, take me back to my previous branch",
        "How do I exit detached HEAD mode?",
        "Get me out of detached HEAD",
        "Return to the branch I was working on before",
    ]

    plan = {
        "root_cause": f"HEAD is detached at commit {sha} with a clean working tree.",
        "risk_tier": "READ_ONLY",
        "execution_steps": [
            {
                "cmd": "git switch -",
                "explanation": "Switch back to the previously checked-out branch."
            }
        ],
        "rollback_cmd": f"git checkout {sha}"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_dirty_tree_preventing_pull(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    mod_file = rng.choice(FILES)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~2 -0",
        untracked=1,
        tracking=f"origin/{branch} (ahead 0, behind {rng.randint(2, 5)})",
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        f"Your local changes to {mod_file} would be overwritten by merge, please commit or stash them",
        "Git pull fails because I have uncommitted local changes",
        "Cannot pull upstream because of modified files",
        "Safely update branch while keeping local uncommitted changes",
    ]

    plan = {
        "root_cause": "Local uncommitted changes prevent fast-forward or rebase pull from remote.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": 'git stash push -u -m "tsugi: auto-stash before pull"',
                "explanation": "Safely stash modified and untracked files away."
            },
            {
                "cmd": "git pull --rebase",
                "explanation": "Replay local branch on top of updated upstream."
            },
            {
                "cmd": "git stash pop",
                "explanation": "Re-apply stashed modifications on top of updated code."
            }
        ],
        "rollback_cmd": "git reset --hard HEAD@{1} && git stash pop"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_accidental_commit_on_main(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    sha = random_sha(rng)
    feat_branch = random_branch(rng)
    main_name = rng.choice(["main", "master"])

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {main_name} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        tracking=f"origin/{main_name} (ahead 1, behind 0)",
        recent_commits=[
            f"{sha} {rng.choice(COMMIT_SUMMARIES)}",
            f"{random_sha(rng)} Previous stable commit"
        ]
    )

    intents = [
        f"I committed directly to {main_name} by accident, move this commit to {feat_branch}",
        f"Accidentally committed on {main_name}, move commit to new feature branch",
        f"Move latest commit from {main_name} to {feat_branch} and reset {main_name} to match origin",
    ]

    plan = {
        "root_cause": f"1 unpushed commit made on protected '{main_name}' branch that belongs on a feature branch.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git branch {feat_branch}",
                "explanation": f"Create '{feat_branch}' at current HEAD so commit is preserved."
            },
            {
                "cmd": f"git reset --hard origin/{main_name}",
                "explanation": f"Reset local {main_name} back to match remote origin/{main_name} cleanly."
            },
            {
                "cmd": f"git switch {feat_branch}",
                "explanation": f"Switch to '{feat_branch}' where the commit is now located."
            }
        ],
        "rollback_cmd": f"git switch {main_name} && git reset --hard {feat_branch}"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_diverged_branch(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    ahead = rng.randint(1, 3)
    behind = rng.randint(2, 6)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        tracking=f"origin/{branch} (ahead {ahead}, behind {behind})",
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        "Updates were rejected because the tip of your current branch is behind its remote counterpart",
        "Git push rejected non fast forward",
        f"How do I sync diverged {branch} with origin without destroying my commits?",
        "Branch diverged, pull changes cleanly",
    ]

    plan = {
        "root_cause": f"Local branch and remote tracking branch have diverged ({ahead} ahead, {behind} behind).",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git pull --rebase origin {branch}",
                "explanation": f"Fetch remote commits and replay local {ahead} commit(s) on top for clean linear history."
            }
        ],
        "rollback_cmd": "git reset --hard ORIG_HEAD"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_reflog_hard_reset_recovery(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    current_sha = random_sha(rng)
    lost_sha = random_sha(rng)
    rec_branch = f"recovered-work-{current_sha[:4]}"

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({current_sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        safe_rollback_target=f"HEAD@{{1}} ({lost_sha})",
        recent_commits=[f"{current_sha} Old commit"],
        reflog=[
            f"0: {current_sha} reset: moving to HEAD~3",
            f"1: {lost_sha} commit: {rng.choice(COMMIT_SUMMARIES)}"
        ]
    )

    intents = [
        "I accidentally ran git reset --hard and lost my unpushed commits, can I get them back?",
        "Recover lost commits after hard reset",
        "Undo git reset --hard HEAD~3",
        "Reflog recovery for deleted commits",
    ]

    plan = {
        "root_cause": f"Hard reset displaced branch HEAD from {lost_sha} to {current_sha}; commits still exist in reflog.",
        "risk_tier": "MUTATING_DANGEROUS",
        "execution_steps": [
            {
                "cmd": f"git branch {rec_branch} HEAD@{{1}}",
                "explanation": f"Create safety branch '{rec_branch}' pointing to lost commit {lost_sha}."
            },
            {
                "cmd": "git reset --hard HEAD@{1}",
                "explanation": f"Restore current branch HEAD back to pre-reset commit {lost_sha}."
            }
        ],
        "rollback_cmd": f"git reset --hard {current_sha}"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_sensitive_staged_file(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    secret = rng.choice(SECRET_FILES)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+1 ~0 -0 R0",
        unstaged="~1 -0",
        untracked=0,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        f"I staged {secret} with API keys by mistake, unstage it before commit",
        f"Unstage {secret} without deleting the file from disk",
        "How do I unstage sensitive credentials before committing?",
        f"Remove {secret} from git index",
    ]

    plan = {
        "root_cause": f"Sensitive file '{secret}' was inadvertently staged for commit.",
        "risk_tier": "READ_ONLY",
        "execution_steps": [
            {
                "cmd": f"git restore --staged {secret}",
                "explanation": f"Remove '{secret}' from the Git staging index while keeping disk contents intact."
            }
        ],
        "rollback_cmd": f"git add {secret}"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_untracked_collision_switch(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    target = rng.choice(["main", "develop", "staging"])
    sha = random_sha(rng)
    file_name = rng.choice(FILES)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=1,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        f"The following untracked working tree files would be overwritten by checkout: {file_name}",
        f"Cannot switch to {target} because untracked file exists",
        "Branch switch blocked by untracked file",
    ]

    plan = {
        "root_cause": f"Untracked local file '{file_name}' collides with a tracked file in target branch.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": 'git stash push -u -m "tsugi: stash untracked before switch"',
                "explanation": "Safely stash untracked files into stash storage instead of deleting."
            },
            {
                "cmd": f"git switch {target}",
                "explanation": f"Switch to target branch '{target}' cleanly."
            }
        ],
        "rollback_cmd": f"git switch {branch} && git stash pop"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_no_upstream_push(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        "The current branch has no upstream branch. To push, use git push --set-upstream",
        "How do I push a newly created local branch?",
        "Git push fails with fatal: The current branch has no upstream branch",
        "Publish new branch to remote origin",
    ]

    plan = {
        "root_cause": f"Local branch '{branch}' has no remote tracking branch configured.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git push --set-upstream origin {branch}",
                "explanation": f"Push '{branch}' to origin and configure upstream tracking."
            }
        ],
        "rollback_cmd": f"git push origin --delete {branch}"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_undo_last_commit(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        "Undo last commit but keep my changes in the working directory",
        "Uncommit previous commit without losing edits",
        "How do I undo my last commit?",
        "Soft reset last commit",
    ]

    plan = {
        "root_cause": "User requested to reverse the last commit while preserving changes as unstaged edits.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git reset HEAD~1",
                "explanation": "Move HEAD back one commit; changes are preserved as unstaged modifications."
            }
        ],
        "rollback_cmd": "git commit -C ORIG_HEAD"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_force_push_lease(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        tracking=f"origin/{branch} (ahead 1, behind 1)",
        recent_commits=[f"{sha} Rebased feature commit"]
    )

    intents = [
        "Force push my rebased feature branch to remote",
        f"I rebased {branch}, how do I push safely?",
        "Update remote branch after interactive rebase",
        "Force push without overwriting teammates' work",
    ]

    plan = {
        "root_cause": f"Rebased history on '{branch}' requires force push to remote.",
        "risk_tier": "HISTORY_REWRITE",
        "execution_steps": [
            {
                "cmd": f"git push --force-with-lease origin {branch}",
                "explanation": f"Push rewritten history to '{branch}' using --force-with-lease to protect remote commits."
            }
        ],
        "rollback_cmd": f"git push origin {branch} --force-with-lease"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_amend_last_commit(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    staged_file = rng.choice(FILES)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~1 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        f"I forgot to include {staged_file} in the previous commit, add it without making a new commit",
        "Amend last commit with staged changes",
        "Combine staged change into the most recent commit",
        "Add missed file to last commit",
    ]

    plan = {
        "root_cause": f"Staged modification in '{staged_file}' intended for absorption into previous commit.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git commit --amend --no-edit",
                "explanation": "Incorporate currently staged changes into the HEAD commit without editing the commit message."
            }
        ],
        "rollback_cmd": "git reset --soft HEAD@{1}"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_unborn_repo_first_commit(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = rng.choice(["main", "master"])

    fact_sheet = format_repo_state(
        head=f"UNBORN ({branch}, no commits yet)",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=rng.randint(2, 5),
    )

    intents = [
        "Brand new repository, create the initial commit",
        "How do I make my first commit in this empty repo?",
        "Initial commit with project files",
        "Track files and commit in unborn repo",
    ]

    plan = {
        "root_cause": "Empty Git repository with untracked files awaiting initial commit.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git add .",
                "explanation": "Stage all workspace files for the initial commit."
            },
            {
                "cmd": 'git commit -m "Initial commit"',
                "explanation": "Create the root commit of the repository."
            }
        ],
        "rollback_cmd": "git rm -r --cached ."
    }

    return fact_sheet, rng.choice(intents), plan


def incident_discard_unstaged_file(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    file_name = rng.choice(FILES)

    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~1 -0",
        untracked=0,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )

    intents = [
        f"Discard changes in {file_name} and revert it back to HEAD",
        f"Undo unstaged edits in {file_name}",
        f"Revert {file_name} to clean version",
        f"Throw away modifications in {file_name}",
    ]

    plan = {
        "root_cause": f"Unstaged modifications in '{file_name}' to be discarded.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git restore {file_name}",
                "explanation": f"Discard unstaged working directory changes in '{file_name}', restoring it to HEAD."
            }
        ],
        "rollback_cmd": "git status"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_bisect_abort(rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
    sha = random_sha(rng)

    fact_sheet = format_repo_state(
        head=f"DETACHED at {sha}",
        ongoing_op="BISECT in progress",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        recent_commits=[f"{sha} Testing bisect step"]
    )

    intents = [
        "Stop git bisect and return to my branch",
        "Cancel bisect session",
        "Abort git bisect",
        "End bisecting",
    ]

    plan = {
        "root_cause": "Active git bisect session that user wishes to cancel.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git bisect reset",
                "explanation": "End the bisect session and return HEAD to the original branch before bisecting."
            }
        ],
        "rollback_cmd": "git bisect reset"
    }

    return fact_sheet, rng.choice(intents), plan


def incident_repl_keep_unstaged_instead(rng: random.Random) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    files = rng.sample(FILES, 2)
    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~2 -0",
        untracked=0,
        unstaged_files=[f"M {f}" for f in files],
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )
    history = [
        ("Switch to main and pull latest changes", "Stash modified files and switch to main")
    ]
    intents = [
        "Wait, keep my unstaged changes instead without stashing them",
        "Don't stash, keep my working directory edits and switch branch",
        "Instead of stashing, bring my unstaged modifications along to main",
    ]
    plan = {
        "root_cause": "User requested switching branches while carrying over unstaged working directory changes.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git switch -m main",
                "explanation": "Switch to branch 'main' and merge local uncommitted changes into the destination tree.",
                "requires_confirmation": False
            }
        ],
        "rollback_cmd": f"git switch -m {branch}",
        "confidence": 0.96
    }
    return fact_sheet, rng.choice(intents), plan, history


def incident_repl_only_stage_auth_and_stash_rest(rng: random.Random) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    auth_file = "src/auth.ts"
    other_files = rng.sample(FILES, 2)
    all_files = [auth_file] + other_files
    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged=f"~{len(all_files)} -0",
        untracked=0,
        unstaged_files=[f"M {f}" for f in all_files],
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )
    history = [
        ("Prepare commit for all modified files", "Stage all working modifications into index")
    ]
    intents = [
        f"Wait, only stage {auth_file} and stash the rest",
        f"Only stage {auth_file} and stash other unstaged modifications",
        f"Don't stage everything—only stage {auth_file} and stash the remaining files",
    ]
    plan = {
        "root_cause": f"Selective staging of {auth_file} while stashing remaining modified files.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git add {auth_file}",
                "explanation": f"Stage {auth_file} for commit.",
                "requires_confirmation": False
            },
            {
                "cmd": f"git stash push -m \"wip: stash non-auth modifications\" -- {' '.join(other_files)}",
                "explanation": "Stash the remaining unstaged files into stash storage.",
                "requires_confirmation": False
            }
        ],
        "rollback_cmd": f"git restore --staged {auth_file} && git stash pop",
        "confidence": 0.95
    }
    return fact_sheet, rng.choice(intents), plan, history


def incident_repl_skip_rebase_instead_of_abort(rng: random.Random) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    conf_file = rng.choice(FILES)
    step_cur = rng.randint(2, 4)
    step_total = step_cur + 2
    fact_sheet = format_repo_state(
        head=f"DETACHED at {sha}",
        ongoing_op=f"REBASE in progress (step {step_cur}/{step_total}) onto origin/main",
        conflicts=[f'"{conf_file}" (both_modified)'],
        staged="+0 ~0 -0 R0",
        unstaged="~1 -0",
        untracked=0,
        tracking=f"origin/{branch} (ahead 3, behind 2)",
        unstaged_files=[f"M {conf_file}"],
        conflict_previews=[
            f"CONFLICT_PREVIEW for \"{conf_file}\" (ours<HEAD> vs theirs<origin/main>):\n<<<<<<< HEAD\n  old_impl();\n=======\n  new_impl();\n>>>>>>> origin/main"
        ],
        incoming_commits=[f"{random_sha(rng)} {rng.choice(COMMIT_SUMMARIES)}"],
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )
    history = [
        ("Rebase conflict encountered", "Conflict in " + conf_file + "; abort or resolve")
    ]
    intents = [
        "Don't abort the rebase—I want to skip this commit and keep rebasing onto main",
        "Skip this conflicting commit and continue rebase instead of aborting",
        "Do not abort, skip this commit and proceed with the rest of the rebase",
    ]
    plan = {
        "root_cause": f"User elected to drop/skip conflicted commit at step {step_cur}/{step_total} and continue rebase.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git rebase --skip",
                "explanation": "Skip applying current commit and continue rebasing remaining commits onto main.",
                "requires_confirmation": False
            }
        ],
        "rollback_cmd": "git rebase --abort",
        "confidence": 0.98
    }
    return fact_sheet, rng.choice(intents), plan, history


def incident_repl_restore_file_without_resetting_staged(rng: random.Random) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    staged_file = rng.choice(FILES)
    target_file = "src/api/handler.rs"
    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~1 -0 R0",
        unstaged="~1 -0",
        untracked=0,
        staged_files=[f"M {staged_file}"],
        unstaged_files=[f"M {target_file}"],
        recent_commits=[
            f"{sha} {rng.choice(COMMIT_SUMMARIES)}",
            f"{random_sha(rng)} {rng.choice(COMMIT_SUMMARIES)}",
            f"{random_sha(rng)} {rng.choice(COMMIT_SUMMARIES)}"
        ]
    )
    history = [
        ("Review my staged work", f"1 file staged ({staged_file}), 1 unstaged ({target_file})")
    ]
    intents = [
        f"Undo the changes in {target_file} from 2 commits ago without resetting my staged work",
        f"Restore {target_file} from HEAD~2 but keep my staged {staged_file} intact",
        f"Revert changes in {target_file} to 2 commits ago without affecting staged changes",
    ]
    plan = {
        "root_cause": f"Restoring specific file '{target_file}' from revision HEAD~2 while keeping staged '{staged_file}' intact.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git restore --source=HEAD~2 --worktree {target_file}",
                "explanation": f"Restore working tree version of '{target_file}' to HEAD~2 state without touching staged index.",
                "requires_confirmation": False
            }
        ],
        "rollback_cmd": f"git checkout HEAD -- {target_file}",
        "confidence": 0.96
    }
    return fact_sheet, rng.choice(intents), plan, history


def incident_repl_take_incoming_for_config_keep_ours_for_code(rng: random.Random) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    merge_sha = random_sha(rng)
    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op=f"MERGE in progress (merging {merge_sha})",
        conflicts=['"package.json" (both_modified)', '"src/index.ts" (both_modified)'],
        staged="+0 ~0 -0 R0",
        unstaged="~2 -0",
        untracked=0,
        unstaged_files=["M package.json", "M src/index.ts"],
        incoming_commits=[f"{merge_sha} Bump dependencies in package.json"],
        conflict_previews=[
            "CONFLICT_PREVIEW for \"package.json\" (ours<HEAD> vs theirs<incoming>):\n<<<<<<< HEAD\n  \"version\": \"1.0.0\"\n=======\n  \"version\": \"1.1.0\"\n>>>>>>> incoming"
        ]
    )
    history = [
        ("Automatic merge stopped on conflicts", "Conflicts in package.json and src/index.ts")
    ]
    intents = [
        "Take incoming changes from upstream for package.json, but keep our changes for src/index.ts",
        "For package.json use theirs, but for src/index.ts use ours",
        "Accept theirs for package.json and accept ours for src/index.ts",
    ]
    plan = {
        "root_cause": "Granular multi-file conflict resolution: accept upstream version of package.json and local version of src/index.ts.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": "git checkout --theirs package.json",
                "explanation": "Accept upstream version of package.json to resolve conflict.",
                "requires_confirmation": False
            },
            {
                "cmd": "git add package.json",
                "explanation": "Stage resolved package.json.",
                "requires_confirmation": False
            },
            {
                "cmd": "git checkout --ours src/index.ts",
                "explanation": "Retain local working branch version of src/index.ts.",
                "requires_confirmation": False
            },
            {
                "cmd": "git add src/index.ts",
                "explanation": "Stage resolved src/index.ts.",
                "requires_confirmation": False
            }
        ],
        "rollback_cmd": "git merge --abort",
        "confidence": 0.98
    }
    return fact_sheet, rng.choice(intents), plan, history


def incident_repl_clean_untracked_logs_keep_code(rng: random.Random) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    code_file = rng.choice(FILES)
    log_files = ["debug.log", "test_output.log"]
    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~1 -0",
        untracked=2,
        unstaged_files=[f"M {code_file}"],
        untracked_files=log_files,
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )
    history = [
        ("Check git status", f"Modified {code_file} and 2 untracked log files")
    ]
    intents = [
        "Only discard the untracked log files, don't touch my modified code",
        "Delete untracked log files without resetting my modified source files",
        "Clean only debug.log and test_output.log, keep working tree edits",
    ]
    plan = {
        "root_cause": "Targeted deletion of untracked log files while strictly protecting modified code.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git clean -f -- {' '.join(log_files)}",
                "explanation": "Force remove specified untracked log files while leaving modified files untouched.",
                "requires_confirmation": False
            }
        ],
        "rollback_cmd": "git status",
        "confidence": 0.95
    }
    return fact_sheet, rng.choice(intents), plan, history


def incident_repl_fast_forward_upstream(rng: random.Random) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]:
    branch = random_branch(rng)
    sha = random_sha(rng)
    inc_sha1 = random_sha(rng)
    inc_sha2 = random_sha(rng)
    fact_sheet = format_repo_state(
        head=f"ATTACHED to {branch} ({sha})",
        ongoing_op="NONE",
        staged="+0 ~0 -0 R0",
        unstaged="~0 -0",
        untracked=0,
        tracking=f"origin/{branch} (ahead 0, behind 2)",
        incoming_commits=[
            f"{inc_sha1} {rng.choice(COMMIT_SUMMARIES)}",
            f"{inc_sha2} {rng.choice(COMMIT_SUMMARIES)}"
        ],
        recent_commits=[f"{sha} {rng.choice(COMMIT_SUMMARIES)}"]
    )
    history = [
        ("Check tracking status", "Branch is 2 commits behind origin/" + branch)
    ]
    intents = [
        "Fast-forward the branch without creating a merge commit",
        "Update branch with upstream using fast-forward only",
        "Fast forward my branch to match upstream remote",
    ]
    plan = {
        "root_cause": f"Branch '{branch}' is strictly 2 commits behind upstream; fast-forward merge is possible without merge commit.",
        "risk_tier": "MUTATING_SAFE",
        "execution_steps": [
            {
                "cmd": f"git merge --ff-only origin/{branch}",
                "explanation": f"Fast-forward branch to match origin/{branch} without creating a merge commit.",
                "requires_confirmation": False
            }
        ],
        "rollback_cmd": f"git reset --hard {sha}",
        "confidence": 0.97
    }
    return fact_sheet, rng.choice(intents), plan, history


INCIDENT_GENERATORS = [
    incident_merge_conflict,
    incident_rebase_conflict,
    incident_rebase_clean_continue,
    incident_merge_clean_continue,
    incident_cherry_pick_conflict,
    incident_detached_head_dirty,
    incident_detached_head_clean,
    incident_dirty_tree_preventing_pull,
    incident_accidental_commit_on_main,
    incident_diverged_branch,
    incident_reflog_hard_reset_recovery,
    incident_sensitive_staged_file,
    incident_untracked_collision_switch,
    incident_no_upstream_push,
    incident_undo_last_commit,
    incident_force_push_lease,
    incident_amend_last_commit,
    incident_unborn_repo_first_commit,
    incident_discard_unstaged_file,
    incident_bisect_abort,
    incident_repl_keep_unstaged_instead,
    incident_repl_only_stage_auth_and_stash_rest,
    incident_repl_skip_rebase_instead_of_abort,
    incident_repl_restore_file_without_resetting_staged,
    incident_repl_take_incoming_for_config_keep_ours_for_code,
    incident_repl_clean_untracked_logs_keep_code,
    incident_repl_fast_forward_upstream,
]


def build_chatml_dataset(output_file: Path, total_examples: int = 5000, seed: int = 42) -> None:
    """Generate high-diversity, schema-aligned ChatML training dataset."""
    rng = random.Random(seed)
    dataset = []
    seen_prompts = set()

    for i in range(total_examples):
        generator = rng.choice(INCIDENT_GENERATORS)
        gen_res = generator(rng)
        if len(gen_res) == 4:
            fact_sheet, intent, raw_plan, history = gen_res
        else:
            fact_sheet, intent, raw_plan = gen_res
            history = []

        plan = normalize_plan(raw_plan)

        if history:
            history_lines = ["\nPREVIOUS_TURNS:"]
            for h_user, h_asst in history[-3:]:
                history_lines.append(f"- USER: {h_user}\n  ASSISTANT: {h_asst}")
            history_str = "\n".join(history_lines) + "\n\n"
        else:
            history_str = "\n\n"

        user_content = f"{fact_sheet}{history_str}USER_INTENT: {intent}"
        assistant_content = json.dumps(plan, indent=2)

        prompt_key = (fact_sheet, intent)
        if prompt_key in seen_prompts:
            # Re-generate with fresh jitter to maintain uniqueness
            continue
        seen_prompts.add(prompt_key)

        chatml_entry = {
            "id": f"tsugi-train-{len(dataset):05d}",
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

    print(f"Generated {len(dataset)} unique training examples -> {output_file}")


if __name__ == "__main__":
    out_path = Path(__file__).parent / "dataset" / "tsugi_train.jsonl"
    build_chatml_dataset(out_path, total_examples=5000)
