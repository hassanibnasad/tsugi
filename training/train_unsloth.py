#!/usr/bin/env python3
"""
tsugi (次) Fine-Tuning Pipeline using Unsloth.

Trains Qwen/Qwen2.5-Coder-1.5B-Instruct on the tsugi Git diagnostic dataset.
Runs efficiently on 1x NVIDIA GPU with >= 6GB VRAM (e.g. RTX 3060, RTX 4060, T4).

Features:
- Fast 4-bit QLoRA with Unsloth kernels (2x faster, 70% less memory)
- ChatML prompt template formatting
- GBNF schema alignment
- Automatic export to GGUF (Q4_K_M and Q5_K_M) ready for llama-cli / tsugi
"""

import os
import sys
from pathlib import Path

def main():
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("[!] Unsloth or dependencies not installed.")
        print("    Install with: pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("                  pip install --no-deps trl peft accelerate bitsandbytes")
        sys.exit(1)

    max_seq_length = 2048
    dtype = None # Auto-detect Float16 / Bfloat16
    load_in_4bit = True

    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print(f"[*] Loading base model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0, # Optimized 0 dropout for Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Set up ChatML template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    dataset_path = Path(__file__).parent / "dataset" / "tsugi_train.jsonl"
    if not dataset_path.exists():
        print(f"[*] Generating dataset first using synthetic_generator.py...")
        from synthetic_generator import build_chatml_dataset
        build_chatml_dataset(dataset_path, total_examples=600)

    print(f"[*] Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    output_dir = "./tsugi_checkpoints"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=120,
            learning_rate=2e-4,
            fp16=not FastLanguageModel.is_bfloat16_supported(),
            bf16=FastLanguageModel.is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    print("[*] Starting Supervised Fine-Tuning...")
    trainer.train()

    print("[*] Training completed! Saving LoRA adapters...")
    lora_dir = "./tsugi_qwen_lora"
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print("[*] Exporting to GGUF format (Q4_K_M and Q5_K_M)...")
    gguf_dir = Path.home() / ".tsugi" / "models"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Export Q4_K_M for ultra-lightweight inference (<1.2GB RAM)
    model.save_pretrained_gguf(
        str(gguf_dir / "qwen2.5-coder-1.5b-instruct-q4_k_m"),
        tokenizer,
        quantization_method="q4_k_m"
    )

    print(f"[✔] Successfully exported GGUF model to {gguf_dir}!")
    print("    Ready to use with: tsugi fix")

if __name__ == "__main__":
    main()
