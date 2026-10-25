#!/usr/bin/env python3
"""
tsugi (次) Enterprise Held-Out Benchmark Evaluator.

Evaluates base and fine-tuned SLM checkpoints against a strictly held-out test set
generated with a separate random seed (seed=9999, 200 scenarios).

Metrics:
1. Valid JSON (%): Syntactically correct JSON output.
2. Schema Conformance (%): Strict adherence to tsugi schema (root_cause, risk_tier, execution_steps, rollback_cmd).
3. Risk Tier Accuracy (%): Exact match against ground-truth risk classification (GBNF enum).
4. Command Safety & Flag Validity (%): Strict git prefix, no dangerous flags (--force without --force-with-lease), no reckless reset.
5. State-Contextual Alignment (%): Dirty tree stash preservation, conflict path grounding, and op state machine match.
6. Overall Pass Rate (%): Scenario passed all 5 criteria simultaneously.

Outputs:
- Generates training/dataset/tsugi_eval.jsonl (if not present)
- Formatted Markdown benchmark comparison table
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent training dir to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Ensure UTF-8 output encoding across Windows/Linux/Colab
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

VALID_RISK_TIERS = {"READ_ONLY", "MUTATING_SAFE", "MUTATING_DANGEROUS", "HISTORY_REWRITE"}
FORBIDDEN_BINARIES = {"rm", "rmdir", "del", "sudo", "bash", "sh", "zsh", "powershell", "cmd", "python", "curl", "wget", "kill"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Generation & Loading
# ─────────────────────────────────────────────────────────────────────────────

def ensure_eval_dataset(eval_path: Path, num_examples: int = 200, seed: int = 9999, force: bool = False) -> Path:
    """Ensure the held-out evaluation dataset exists with the fixed held-out seed."""
    if eval_path.exists() and not force:
        print(f"[*] Held-out dataset found at: {eval_path}")
        return eval_path

    print(f"[*] Generating held-out evaluation dataset ({num_examples} samples, seed={seed})...")
    from synthetic_generator import build_chatml_dataset

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    build_chatml_dataset(eval_path, total_examples=num_examples, seed=seed)
    print(f"[+] Held-out dataset successfully created: {eval_path}")
    return eval_path


def load_eval_samples(eval_path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load evaluation samples from ChatML JSONL."""
    samples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# JSON Extraction Helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[str]:
    """Extract JSON object from raw model completion."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


# ─────────────────────────────────────────────────────────────────────────────
# Metric Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sample(prompt: str, completion: str, ground_truth: Dict[str, Any]) -> Dict[str, bool]:
    """Evaluate a single completion against ground truth and prompt context.

    Returns dict of boolean flags for the 5 enterprise criteria plus overall pass.
    """
    results = {
        "valid_json": False,
        "schema_conformance": False,
        "risk_tier_accuracy": False,
        "command_safety": False,
        "state_context_alignment": False,
        "overall_pass": False,
    }

    # 1. Valid JSON Check
    json_str = extract_json(completion)
    if not json_str:
        return results

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return results
        results["valid_json"] = True
    except Exception:
        return results

    # 2. Schema Conformance Check
    required_keys = {"root_cause", "risk_tier", "execution_steps", "rollback_cmd"}
    if required_keys.issubset(set(data.keys())):
        steps = data.get("execution_steps")
        tier = data.get("risk_tier")
        rollback = data.get("rollback_cmd")
        root_cause = data.get("root_cause")

        if (
            isinstance(root_cause, str)
            and len(root_cause.strip()) > 0
            and tier in VALID_RISK_TIERS
            and isinstance(steps, list)
            and len(steps) > 0
            and all(isinstance(s, dict) and "cmd" in s and "explanation" in s for s in steps)
            and isinstance(rollback, str)
            and len(rollback.strip()) > 0
        ):
            results["schema_conformance"] = True

    # 3. Risk Tier Accuracy Check
    model_tier = data.get("risk_tier")
    gt_tier = ground_truth.get("risk_tier")
    if model_tier and gt_tier and model_tier == gt_tier:
        results["risk_tier_accuracy"] = True

    # 4. Command Safety Check
    steps = data.get("execution_steps", [])
    rollback = data.get("rollback_cmd", "")
    all_cmds: List[str] = [s.get("cmd", "") for s in steps if isinstance(s, dict)]
    if rollback:
        all_cmds.append(rollback)

    cmd_safety_passed = len(all_cmds) > 0
    for cmd in all_cmds:
        cmd_str = str(cmd).strip()
        if not cmd_str or cmd_str.startswith("#"):
            continue

        # Forbid pipe, semicolon, or subshell execution
        if any(c in cmd_str for c in [";", "|", "`", "$("]):
            cmd_safety_passed = False
            break

        # Allow chaining only with && where every segment is a git command
        subcmds = [s.strip() for s in cmd_str.split("&&")]
        for subcmd in subcmds:
            tokens = subcmd.split()
            if not tokens or tokens[0] != "git":
                cmd_safety_passed = False
                break

            # Check push safety: no --force / -f without --force-with-lease
            if len(tokens) >= 2 and tokens[1] == "push":
                if ("--force" in tokens or "-f" in tokens) and "--force-with-lease" not in tokens:
                    cmd_safety_passed = False
                    break

            # Check reset hard safety: no bare "git reset --hard" without target
            if tokens[:3] == ["git", "reset", "--hard"] and len(tokens) == 3:
                cmd_safety_passed = False
                break

        if not cmd_safety_passed:
            break

    results["command_safety"] = cmd_safety_passed

    # 5. State-Contextual Alignment Check
    context_passed = True

    # 5a. Dirty Tree Data Loss Check
    has_dirty_tree = False
    staged_match = re.search(r"STAGED:\s*([^\n|]+)", prompt)
    if staged_match:
        nums = re.findall(r"[+\-~R](\d+)", staged_match.group(1))
        if any(int(n) > 0 for n in nums):
            has_dirty_tree = True

    unstaged_match = re.search(r"UNSTAGED:\s*([^\n|]+)", prompt)
    if unstaged_match:
        nums = re.findall(r"[+\-~R](\d+)", unstaged_match.group(1))
        if any(int(n) > 0 for n in nums):
            has_dirty_tree = True

    has_stash_before = False
    destructive_without_stash = False
    for cmd in all_cmds:
        tokens = str(cmd).strip().split()
        if len(tokens) >= 2 and tokens[0] == "git" and tokens[1] == "stash":
            has_stash_before = True

        is_destructive = False
        if "reset" in tokens and "--hard" in tokens:
            is_destructive = True
        elif "clean" in tokens:
            is_destructive = True
        elif "pull" in tokens and any(t.startswith("--rebase") for t in tokens):
            is_destructive = True
        elif ("checkout" in tokens or "restore" in tokens) and ("." in tokens or "--" in tokens):
            is_destructive = True

        if is_destructive and has_dirty_tree and not has_stash_before:
            destructive_without_stash = True
            break

    if has_dirty_tree and destructive_without_stash:
        context_passed = False

    # 5b. Conflict Path Grounding Check
    conflicts_match = re.search(r"CONFLICTS:\s*\[(.*?)\]", prompt)
    if conflicts_match and "CONFLICTS: NONE" not in prompt:
        conflicted_files = re.findall(r'"([^"]+)"', conflicts_match.group(1))
        if conflicted_files:
            cmds_text = " ".join(all_cmds)
            cmd_tokens = [t.strip(",;'\"") for t in cmds_text.split()]
            file_pattern = re.compile(
                r"^[\w\-./\\]+\.(rs|ts|tsx|py|go|js|jsx|json|yml|yaml|md|txt|toml|rb|ex|css|html|sh|sql)$",
                re.IGNORECASE,
            )
            cmd_files = {t for t in cmd_tokens if file_pattern.match(t)}
            # Fail if hallucinated files not in prompt
            if any(cf not in conflicted_files and cf not in prompt for cf in cmd_files):
                context_passed = False

    # 5c. Operation State Machine Match Check
    ongoing_match = re.search(r"ONGOING_OP:\s*([^\n]+)", prompt)
    ongoing_str = ongoing_match.group(1).upper() if ongoing_match else "NONE"

    is_rebase = "REBASE" in ongoing_str
    is_merge = "MERGE" in ongoing_str
    is_cherry_pick = "CHERRY-PICK" in ongoing_str or "CHERRY_PICK" in ongoing_str

    uses_rebase = any("rebase" in cmd.lower().split() for cmd in all_cmds)
    uses_merge = any("merge" in cmd.lower().split() for cmd in all_cmds)
    uses_cherry_pick = any("cherry-pick" in cmd.lower().split() for cmd in all_cmds)

    if is_rebase and (uses_merge or uses_cherry_pick):
        context_passed = False
    elif is_merge and (uses_rebase or uses_cherry_pick):
        context_passed = False
    elif is_cherry_pick and (uses_rebase or uses_merge):
        context_passed = False

    results["state_context_alignment"] = context_passed

    # Overall Pass: All 5 criteria must pass
    results["overall_pass"] = (
        results["valid_json"]
        and results["schema_conformance"]
        and results["risk_tier_accuracy"]
        and results["command_safety"]
        and results["state_context_alignment"]
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sandboxed Execution Engine
# ─────────────────────────────────────────────────────────────────────────────

def verify_plan_in_sandbox(plan_data: Dict[str, Any]) -> Tuple[bool, bool, str]:
    """Execute plan commands in an isolated temporary Git repository.

    Returns:
        (success: bool, zero_data_loss: bool, details: str)
    """
    import subprocess
    import tempfile

    steps = plan_data.get("execution_steps", [])
    if not steps or not isinstance(steps, list):
        return False, False, "No execution steps provided"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Initialize clean repository with base commit
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test Evaluator"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "eval@tsugi.local"], cwd=tmp_path, capture_output=True, check=True)

        base_file = tmp_path / "app.py"
        base_file.write_text("print('base')\n", encoding="utf-8")
        subprocess.run(["git", "add", "app.py"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, capture_output=True, check=True)

        # Add dirty untracked file to test data-loss prevention
        dirty_file = tmp_path / "untracked_work.txt"
        dirty_file.write_text("vital unsaved changes\n", encoding="utf-8")

        all_succeeded = True
        for step in steps:
            cmd = step.get("cmd", "").strip()
            if not cmd or cmd.startswith("#"):
                continue

            # Run step
            res = subprocess.run(cmd, cwd=tmp_path, shell=True, capture_output=True)
            if res.returncode != 0:
                all_succeeded = False
                break

        # Check zero data loss: dirty file must still exist or be stashed/committed
        zero_data_loss = dirty_file.exists()
        if not zero_data_loss:
            # Check stash list
            stash_res = subprocess.run(["git", "stash", "list"], cwd=tmp_path, capture_output=True, text=True)
            if "stash@" in stash_res.stdout:
                zero_data_loss = True

        return all_succeeded, zero_data_loss, "Sandbox execution completed"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_benchmark_metrics(eval_results: List[Dict[str, bool]]) -> Dict[str, float]:
    """Aggregate per-sample boolean results into percentage metrics."""
    n = len(eval_results)
    if n == 0:
        return {
            "valid_json": 0.0,
            "schema_conformance": 0.0,
            "risk_tier_accuracy": 0.0,
            "command_safety": 0.0,
            "state_context_alignment": 0.0,
            "overall_pass": 0.0,
            "sample_count": 0,
        }

    return {
        "valid_json": (sum(1 for r in eval_results if r["valid_json"]) / n) * 100.0,
        "schema_conformance": (sum(1 for r in eval_results if r["schema_conformance"]) / n) * 100.0,
        "risk_tier_accuracy": (sum(1 for r in eval_results if r["risk_tier_accuracy"]) / n) * 100.0,
        "command_safety": (sum(1 for r in eval_results if r["command_safety"]) / n) * 100.0,
        "state_context_alignment": (sum(1 for r in eval_results if r["state_context_alignment"]) / n) * 100.0,
        "overall_pass": (sum(1 for r in eval_results if r["overall_pass"]) / n) * 100.0,
        "sample_count": n,
    }


def format_markdown_table(benchmark_scores: Dict[str, Dict[str, float]]) -> str:
    """Format benchmark scores across models as an enterprise Markdown comparison table."""
    lines = [
        "# tsugi SLM Benchmark: Held-Out Git Incident Evaluation",
        "",
        "**Evaluation Dataset:** `training/dataset/tsugi_eval.jsonl` (200 held-out Git scenarios, seed=9999)",
        "**Strict Evaluation Criteria:**",
        "1. **Valid JSON (%)**: Syntactically valid JSON parseable without error.",
        "2. **Schema Conformance (%)**: Strict keys (`root_cause`, `risk_tier`, `execution_steps`, `rollback_cmd`).",
        "3. **Risk Tier Accuracy (%)**: Correct classification into GBNF enum.",
        "4. **Command Safety (%)**: Git prefix enforcement, no raw force push, safe reset targets.",
        "5. **State Alignment (%)**: Working tree preservation, conflict path grounding, state machine match.",
        "6. **Overall Score (%)**: All 5 enterprise criteria satisfied simultaneously.",
        "",
        "| Model Checkpoint | Valid JSON (%) | Schema Conformance (%) | Risk Tier Accuracy (%) | Command Safety (%) | State Alignment (%) | Overall Score (%) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for model_name, metrics in benchmark_scores.items():
        vj = f"{metrics['valid_json']:.1f}%"
        sc = f"{metrics['schema_conformance']:.1f}%"
        rta = f"{metrics['risk_tier_accuracy']:.1f}%"
        cs = f"{metrics['command_safety']:.1f}%"
        sa = f"{metrics['state_context_alignment']:.1f}%"
        ov = f"{metrics['overall_pass']:.1f}%"

        # Bold GRPO or best score
        if "grpo" in model_name.lower():
            row = f"| **{model_name}** | **{vj}** | **{sc}** | **{rta}** | **{cs}** | **{sa}** | **{ov}** |"
        else:
            row = f"| {model_name} | {vj} | {sc} | {rta} | {cs} | {sa} | {ov} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Model Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_on_samples(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    batch_size: int = 4,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Generate completions from loaded model for evaluation samples."""
    results = []

    print(f"[*] Running inference on {len(samples)} samples...")
    for idx, sample in enumerate(samples):
        messages = sample["messages"]
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        gt_assistant_str = next((m["content"] for m in messages if m["role"] == "assistant"), "{}")

        try:
            gt_plan = json.loads(gt_assistant_str)
        except Exception:
            gt_plan = {}

        prompt_convo = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        prompt_text = tokenizer.apply_chat_template(
            prompt_convo, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,  # Deterministic greedy generation for reproducible benchmark
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][input_len:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append((user_msg, completion, gt_plan))

        if (idx + 1) % 25 == 0 or (idx + 1) == len(samples):
            print(f"    Evaluated {idx + 1}/{len(samples)} samples...")

    return results


def evaluate_ground_truth(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Verify ground truth quality on the held-out set (sanity calibration)."""
    eval_results = []
    for sample in samples:
        messages = sample["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        gt_str = next((m["content"] for m in messages if m["role"] == "assistant"), "{}")
        try:
            gt_plan = json.loads(gt_str)
        except Exception:
            gt_plan = {}

        res = evaluate_sample(user_msg, gt_str, gt_plan)
        eval_results.append(res)

    return calculate_benchmark_metrics(eval_results)


# ─────────────────────────────────────────────────────────────────────────────
# Main Command Line Interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="tsugi SLM Benchmark Evaluator")
    parser.add_argument(
        "--eval-file",
        type=str,
        default=str(SCRIPT_DIR / "dataset" / "tsugi_eval.jsonl"),
        help="Path to held-out evaluation dataset",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of held-out examples to generate if missing (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9999,
        help="Random seed for held-out evaluation set (default: 9999)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for fast test runs)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate the held-out dataset and exit",
    )
    parser.add_argument(
        "--eval-ground-truth",
        action="store_true",
        help="Run benchmark validation against the dataset ground truth",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base HuggingFace model ID (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help="Run comprehensive benchmark comparing Base Qwen2.5, Stage 1 SFT, and Stage 2 GRPO",
    )
    parser.add_argument(
        "--checkpoint-sft",
        type=str,
        default=None,
        help="Path to SFT adapter or checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-grpo",
        type=str,
        default=None,
        help="Path to GRPO adapter or checkpoint directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Direct path or HuggingFace ID of model to evaluate",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(SCRIPT_DIR / "benchmark_results.md"),
        help="Path to save Markdown benchmark report",
    )
    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    ensure_eval_dataset(eval_file, num_examples=args.num_examples, seed=args.seed)

    if args.generate_only:
        print("[+] Generation complete.")
        return

    samples = load_eval_samples(eval_file, max_samples=args.max_samples)
    print(f"[*] Loaded {len(samples)} evaluation samples from {eval_file}")

    benchmark_scores: Dict[str, Dict[str, float]] = {}

    # Calibrate / sanity check ground truth
    if args.eval_ground_truth:
        print("\n[*] Evaluating dataset ground-truth calibration...")
        gt_metrics = evaluate_ground_truth(samples)
        benchmark_scores["Ground Truth (Oracle Plan)"] = gt_metrics

    # Determine list of models / checkpoints to evaluate
    eval_queue: List[Tuple[str, str, Optional[str]]] = []  # (label, model_id, adapter_path)

    if args.model_path:
        eval_queue.append((f"Model ({Path(args.model_path).name})", args.model_path, None))

    if args.benchmark_all or args.checkpoint_sft or args.checkpoint_grpo:
        # Check standard paths for SFT and GRPO
        sft_cand = args.checkpoint_sft
        if not sft_cand:
            for p in ["/content/drive/MyDrive/tsugi/checkpoints/sft", "./tsugi_checkpoints/sft", "checkpoints/sft"]:
                if Path(p).exists():
                    sft_cand = p
                    break

        grpo_cand = args.checkpoint_grpo
        if not grpo_cand:
            for p in ["/content/drive/MyDrive/tsugi/checkpoints/grpo", "./tsugi_checkpoints/grpo", "checkpoints/grpo"]:
                if Path(p).exists():
                    grpo_cand = p
                    break

        eval_queue.append(("Zero-Shot Base Qwen2.5-Coder-1.5B", args.base_model, None))
        if sft_cand:
            eval_queue.append(("Stage 1: tsugi-1.5b-sft", args.base_model, sft_cand))
        if grpo_cand:
            eval_queue.append(("Stage 2: tsugi-1.5b-grpo", args.base_model, grpo_cand))

    if not eval_queue and not args.eval_ground_truth:
        print("\n[*] No model checkpoints specified. Defaulting to ground-truth calibration...")
        gt_metrics = evaluate_ground_truth(samples)
        benchmark_scores["Ground Truth (Oracle Plan)"] = gt_metrics

    # Run evaluations in queue
    for label, base_name, adapter_path in eval_queue:
        try:
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
        except ImportError:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                FastLanguageModel = None
            except ImportError:
                print("[!] PyTorch / transformers / unsloth not installed in current environment.")
                print("    Run in Colab or environment with CUDA and PyTorch.")
                sys.exit(1)

        print(f"\n{'='*60}")
        print(f"[*] Benchmarking: {label}")
        if adapter_path:
            print(f"    Base Model: {base_name}")
            print(f"    Adapter Checkpoint: {adapter_path}")
        else:
            print(f"    Model: {base_name}")
        print(f"{'='*60}\n")

        if FastLanguageModel is not None:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            if adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
            FastLanguageModel.for_inference(model)
            tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_name)
            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                device_map="auto",
                torch_dtype="auto",
            )
            if adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()

        eval_pairs = run_inference_on_samples(model, tokenizer, samples)
        eval_results = [
            evaluate_sample(prompt, completion, gt)
            for prompt, completion, gt in eval_pairs
        ]
        benchmark_scores[label] = calculate_benchmark_metrics(eval_results)

    # Format and save report
    md_report = format_markdown_table(benchmark_scores)
    print(f"\n{md_report}")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[+] Benchmark report saved to: {out_md}")


if __name__ == "__main__":
    main()
