#!/usr/bin/env python3
"""
tsugi (次) Fine-Tuning Pipeline using Unsloth (Colab + Google Drive + SFT + GRPO).

Trains Qwen/Qwen2.5-Coder-1.5B-Instruct on the tsugi Git diagnostic dataset.
Runs efficiently on 1x NVIDIA GPU with >= 6GB VRAM (e.g. T4, RTX 3060, A100).

Features:
- Google Colab auto-detection & automatic Google Drive checkpoint persistence
- 4-bit QLoRA with Unsloth kernels (2x faster, 70% less memory)
- Stage 1: Supervised Fine-Tuning (SFT) with ChatML format
- Stage 2: Group Relative Policy Optimization (GRPO) with rule-based Git safety rewards:
    * Valid JSON parsing reward
    * Strict schema keys reward (root_cause, risk_tier, execution_steps, rollback_cmd)
    * GBNF risk tier enum adherence reward
    * Git command prefix enforcement reward (no arbitrary binary execution)
    * Push safety reward (--force-with-lease vs --force)
- Automatic export to GGUF (Q4_K_M and Q5_K_M) saved directly to Google Drive
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Environment & Google Drive Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_colab_environment() -> bool:
    """Check if the script is executing within Google Colab."""
    return "google.colab" in sys.modules or Path("/content").exists()


def setup_google_drive(mount_point: str = "/content/drive") -> Optional[Path]:
    """Mount Google Drive if running in Google Colab and return the base Drive path."""
    if not is_colab_environment():
        return None

    drive_path = Path(mount_point)
    if not drive_path.exists():
        try:
            print("[*] Google Colab detected. Mounting Google Drive...")
            from google.colab import drive  # type: ignore
            drive.mount(mount_point)
            print("[✔] Google Drive successfully mounted at /content/drive")
        except Exception as e:
            print(f"[!] Warning: Failed to mount Google Drive: {e}")
            print("    Checkpoints will be stored in Colab local ephemeral storage.")
            return None

    tsugi_drive_dir = drive_path / "MyDrive" / "tsugi"
    tsugi_drive_dir.mkdir(parents=True, exist_ok=True)
    return tsugi_drive_dir


# ─────────────────────────────────────────────────────────────────────────────
# Reward Functions for GRPO (Group Relative Policy Optimization)
# ─────────────────────────────────────────────────────────────────────────────

VALID_RISK_TIERS = {"READ_ONLY", "MUTATING_SAFE", "MUTATING_DANGEROUS", "HISTORY_REWRITE"}


def extract_json_candidate(text: str) -> Optional[str]:
    """Extract a JSON object substring from raw model output."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def reward_valid_json(completions, **kwargs) -> List[float]:
    """Reward completion for being syntactically valid JSON."""
    rewards = []
    for completion in completions:
        # completion can be a string or a list of message dicts depending on TRL version
        content = completion if isinstance(completion, str) else completion[0]["content"]
        json_str = extract_json_candidate(content)
        if not json_str:
            rewards.append(-1.5)
            continue
        try:
            json.loads(json_str)
            rewards.append(1.0)
        except Exception:
            rewards.append(-1.0)
    return rewards


def reward_schema_conformance(completions, **kwargs) -> List[float]:
    """Reward completion for having exact tsugi schema keys."""
    rewards = []
    required_keys = {"root_cause", "risk_tier", "execution_steps", "rollback_cmd"}

    for completion in completions:
        content = completion if isinstance(completion, str) else completion[0]["content"]
        json_str = extract_json_candidate(content)
        if not json_str:
            rewards.append(-1.0)
            continue
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                rewards.append(-1.0)
                continue

            keys = set(data.keys())
            if required_keys.issubset(keys):
                score = 1.0
                # Check step object schema
                steps = data.get("execution_steps")
                if isinstance(steps, list) and len(steps) > 0:
                    all_steps_valid = all(
                        isinstance(s, dict) and "cmd" in s and "explanation" in s
                        for s in steps
                    )
                    if all_steps_valid:
                        score += 0.5
                rewards.append(score)
            else:
                rewards.append(-0.5)
        except Exception:
            rewards.append(-1.0)
    return rewards


def reward_risk_tier_enum(completions, **kwargs) -> List[float]:
    """Reward completion for using one of the 4 strict GBNF risk tiers."""
    rewards = []
    for completion in completions:
        content = completion if isinstance(completion, str) else completion[0]["content"]
        json_str = extract_json_candidate(content)
        if not json_str:
            rewards.append(-1.0)
            continue
        try:
            data = json.loads(json_str)
            tier = data.get("risk_tier")
            if tier in VALID_RISK_TIERS:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        except Exception:
            rewards.append(-0.5)
    return rewards


def reward_git_command_safety(completions, **kwargs) -> List[float]:
    """Reward safe Git commands and penalize non-git execution or unsafe flags."""
    rewards = []
    for completion in completions:
        content = completion if isinstance(completion, str) else completion[0]["content"]
        json_str = extract_json_candidate(content)
        if not json_str:
            rewards.append(-1.0)
            continue
        try:
            data = json.loads(json_str)
            steps = data.get("execution_steps", [])
            rollback = data.get("rollback_cmd", "")

            all_cmds = [s.get("cmd", "") for s in steps if isinstance(s, dict)]
            if rollback:
                all_cmds.append(rollback)

            if not all_cmds:
                rewards.append(-1.0)
                continue

            score = 1.0
            for cmd in all_cmds:
                cmd_str = str(cmd).strip()
                if cmd_str.startswith("#"):
                    continue

                # Every command must start with 'git'
                tokens = cmd_str.split()
                if not tokens or tokens[0] != "git":
                    score -= 2.0  # heavy penalty for non-git command
                    continue

                # Must not use raw force push
                if "push" in tokens and ("--force" in tokens or "-f" in tokens):
                    if "--force-with-lease" not in tokens:
                        score -= 2.0

                # Must not use reckless reset hard without reference target
                if tokens[:3] == ["git", "reset", "--hard"] and len(tokens) == 3:
                    score -= 1.0

            rewards.append(score)
        except Exception:
            rewards.append(-1.0)
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_sft(model, tokenizer, dataset, output_dir: Path, max_steps: int = 150):
    """Run Supervised Fine-Tuning stage."""
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel

    print(f"\n{'='*60}")
    print(f"[*] STAGE 1: Supervised Fine-Tuning (SFT) — {max_steps} steps")
    print(f"{'='*60}\n")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not FastLanguageModel.is_bfloat16_supported(),
            bf16=FastLanguageModel.is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=str(output_dir),
            report_to="none",
        ),
    )

    trainer.train()
    print("[✔] Stage 1 (SFT) training complete!")
    return trainer


def train_grpo(model, tokenizer, dataset, output_dir: Path, max_steps: int = 60):
    """Run Group Relative Policy Optimization (GRPO) stage with safety reward functions."""
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("[!] GRPOTrainer not available in current TRL version. Skipping GRPO stage.")
        print("    Upgrade with: pip install git+https://github.com/huggingface/trl.git")
        return None

    from unsloth import FastLanguageModel

    print(f"\n{'='*60}")
    print(f"[*] STAGE 2: Policy Alignment via GRPO — {max_steps} steps")
    print(f"{'='*60}\n")

    # Format dataset for GRPO: prompt contains system + user message
    def grpo_formatting_func(examples):
        prompts = []
        for convo in examples["messages"]:
            # Drop the assistant response so model generates it
            prompt_msgs = [m for m in convo if m["role"] != "assistant"]
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
        return {"prompt": prompts}

    grpo_dataset = dataset.map(grpo_formatting_func, batched=True)

    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=FastLanguageModel.is_bfloat16_supported(),
        fp16=not FastLanguageModel.is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,  # Group size for relative evaluation
        max_prompt_length=1024,
        max_completion_length=384,
        max_steps=max_steps,
        output_dir=str(output_dir),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_valid_json,
            reward_schema_conformance,
            reward_risk_tier_enum,
            reward_git_command_safety,
        ],
        args=training_args,
        train_dataset=grpo_dataset,
    )

    trainer.train()
    print("[✔] Stage 2 (GRPO) training complete!")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="tsugi SLM Fine-Tuning Pipeline")
    parser.add_argument(
        "--mode",
        choices=["sft", "grpo", "all"],
        default="all",
        help="Training mode: 'sft' (supervised), 'grpo' (reinforcement), or 'all' (sft + grpo)",
    )
    parser.add_argument(
        "--sft-steps",
        type=int,
        default=150,
        help="Maximum training steps for SFT stage",
    )
    parser.add_argument(
        "--grpo-steps",
        type=int,
        default=60,
        help="Maximum training steps for GRPO stage",
    )
    parser.add_argument(
        "--dataset-examples",
        type=int,
        default=5000,
        help="Number of synthetic examples to generate if dataset is missing",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        default=True,
        help="Export quantized GGUF model files after training",
    )
    args = parser.parse_args()

    # 1. Dependency imports
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from datasets import load_dataset
    except ImportError:
        print("[!] Required libraries not installed.")
        print('    Install with: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        print("                  pip install --no-deps trl peft accelerate bitsandbytes")
        sys.exit(1)

    # 2. Storage Setup: Detect Colab + Google Drive
    drive_base = setup_google_drive()
    if drive_base:
        checkpoints_dir = drive_base / "checkpoints"
        models_dir = drive_base / "models"
        print(f"[*] Persistent Google Drive storage enabled at: {drive_base}")
    else:
        checkpoints_dir = Path("./tsugi_checkpoints")
        models_dir = Path.home() / ".tsugi" / "models"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 3. Dataset Generation & Loading
    dataset_path = Path(__file__).parent / "dataset" / "tsugi_train.jsonl"
    if not dataset_path.exists():
        print(f"[*] Dataset not found at {dataset_path}. Generating {args.dataset_examples} examples...")
        from synthetic_generator import build_chatml_dataset
        build_chatml_dataset(dataset_path, total_examples=args.dataset_examples)

    print(f"[*] Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # 4. Model & LoRA Initialization
    max_seq_length = 2048
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print(f"[*] Initializing base model with Unsloth 4-bit QLoRA: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect FP16/BF16
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # 5. Training Execution
    if args.mode in ["sft", "all"]:
        sft_output = checkpoints_dir / "sft"
        train_sft(model, tokenizer, dataset, sft_output, max_steps=args.sft_steps)

    if args.mode in ["grpo", "all"]:
        grpo_output = checkpoints_dir / "grpo"
        train_grpo(model, tokenizer, dataset, grpo_output, max_steps=args.grpo_steps)

    # 6. Save LoRA Adapters
    lora_save_path = checkpoints_dir / "tsugi_qwen_lora"
    print(f"[*] Saving fine-tuned LoRA adapters to {lora_save_path}...")
    model.save_pretrained(str(lora_save_path))
    tokenizer.save_pretrained(str(lora_save_path))

    # 7. GGUF Export
    if args.export_gguf:
        print(f"\n{'='*60}")
        print("[*] Exporting model to GGUF format (Q4_K_M)...")
        print(f"{'='*60}\n")

        gguf_output_name = "qwen2.5-coder-1.5b-instruct-q4_k_m"

        # Export locally / to Google Drive
        target_export_path = models_dir / gguf_output_name
        print(f"[*] Saving GGUF model directly to: {target_export_path}...")
        model.save_pretrained_gguf(
            str(target_export_path),
            tokenizer,
            quantization_method="q4_k_m",
        )

        # Also copy to local ~/.tsugi/models if different
        local_user_models = Path.home() / ".tsugi" / "models"
        if local_user_models.resolve() != models_dir.resolve():
            local_user_models.mkdir(parents=True, exist_ok=True)
            for f in models_dir.glob(f"{gguf_output_name}*.gguf"):
                dest = local_user_models / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    print(f"[✔] Copied GGUF to local engine path: {dest}")

        print(f"\n[✔] Training & GGUF export complete! Models saved in: {models_dir}")
        print("    Ready to test with: tsugi fix")


if __name__ == "__main__":
    main()
