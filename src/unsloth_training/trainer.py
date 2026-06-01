"""
Unsloth Mid-Training (MSM) Trainer
===================================
Continued Pre-Training on synthetic spec documents using Unsloth + LoRA.

Training config follows the paper exactly (Appendix B.4):
  - LoRA rank=64, alpha=128, all attention + MLP projection layers
  - 1 epoch, AdamW, lr=1e-4, cosine schedule, 5% warmup, weight_decay=0.01
  - max_seq_length=4096 (MSM) or 8192 (with long-context instruction-tuning data)

Paper: https://arxiv.org/abs/2605.02087
"""

# Must import unsloth first for all optimizations to apply
import unsloth  # noqa: F401

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


# ─────────────────────────────────────────────
# Config Dataclass
# ─────────────────────────────────────────────

@dataclass
class MSMTrainingConfig:
    """Training configuration for Model Spec Midtraining (MSM)."""

    # --- Data ---
    dataset_path: str = "data/midtrain/general_spec/dataset.jsonl"
    "Path to MSM dataset.jsonl (output of generate_msm_data.sh)"

    dataset_text_field: str = "text"
    "Key in dataset jsonl that holds the raw document text"

    # --- Model ---
    model_name: str = "unsloth/Llama-3.1-8B"
    "Base model to midtrain (HuggingFace model id or local path)"

    max_seq_length: int = 4096
    "Max sequence length. Use 8192 if mixing in long-context instruction-tuning data."

    load_in_4bit: bool = True
    "Use 4-bit quantization (QLoRA). Set False for full bf16 LoRA."

    # --- LoRA (paper: rank=64, alpha=128, all attn + MLP layers) ---
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # --- Training Hyperparameters (paper Appendix B.4) ---
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    optim: str = "adamw_8bit"
    seed: int = 42
    fp16: bool = False
    bf16: bool = True

    # --- Output ---
    output_dir: str = "outputs/msm"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

def load_model_and_tokenizer(config: MSMTrainingConfig):
    """Load base model + tokenizer via Unsloth."""
    from unsloth import FastLanguageModel

    print(f"[MSM] Loading model: {config.model_name}")
    print(f"[MSM] max_seq_length={config.max_seq_length}, load_in_4bit={config.load_in_4bit}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,  # auto-detect
    )

    print(f"[MSM] Applying LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
    )

    return model, tokenizer


def load_dataset(config: MSMTrainingConfig):
    """Load MSM dataset from jsonl file."""
    from datasets import load_dataset as hf_load_dataset

    path = Path(config.dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run 'bash exps/generate_msm_data.sh' first to generate MSM data."
        )

    print(f"[MSM] Loading dataset: {path}")
    dataset = hf_load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )

    print(f"[MSM] Dataset size: {len(dataset):,} documents")

    # Validate text field exists
    if config.dataset_text_field not in dataset.column_names:
        raise ValueError(
            f"Field '{config.dataset_text_field}' not found in dataset. "
            f"Available fields: {dataset.column_names}"
        )

    return dataset


def run_training(config: MSMTrainingConfig):
    """Run MSM training."""
    from transformers import TrainingArguments
    from trl import SFTTrainer, SFTConfig

    # Load model + data
    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_dataset(config)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[MSM] Training Config:")
    print(f"  epochs         : {config.num_train_epochs}")
    print(f"  learning_rate  : {config.learning_rate}")
    print(f"  lr_scheduler   : {config.lr_scheduler_type}")
    print(f"  warmup_ratio   : {config.warmup_ratio}")
    print(f"  weight_decay   : {config.weight_decay}")
    print(f"  batch_size     : {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.effective_batch_size()}")
    print(f"  output_dir     : {config.output_dir}")
    print()

    # Compute warmup_steps from ratio (warmup_ratio deprecated in trl>=0.9)
    total_samples = len(dataset)
    steps_per_epoch = max(1, total_samples // config.effective_batch_size())
    total_steps = steps_per_epoch * config.num_train_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        seed=config.seed,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=True,
        # SFTConfig-specific
        dataset_text_field=config.dataset_text_field,
        max_seq_length=config.max_seq_length,
        dataset_num_proc=4,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    # Print trainable parameters
    model.print_trainable_parameters()

    print("\n[MSM] Starting mid-training...\n")
    trainer_stats = trainer.train()

    print(f"\n[MSM] Training complete!")
    print(f"  Runtime      : {trainer_stats.metrics.get('train_runtime', 0):.1f}s")
    print(f"  Samples/sec  : {trainer_stats.metrics.get('train_samples_per_second', 0):.2f}")
    print(f"  Final loss   : {trainer_stats.metrics.get('train_loss', 0):.4f}")

    # Save adapter
    adapter_path = Path(config.output_dir) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n[MSM] Adapter saved to: {adapter_path}")

    # Push to hub if requested
    if config.push_to_hub and config.hub_model_id:
        print(f"[MSM] Pushing to HuggingFace Hub: {config.hub_model_id}")
        model.push_to_hub(config.hub_model_id)
        tokenizer.push_to_hub(config.hub_model_id)

    return trainer_stats
