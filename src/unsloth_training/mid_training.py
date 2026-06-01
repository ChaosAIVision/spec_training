"""
CLI entrypoint for Unsloth Mid-Training (MSM).

Usage:
    python -m src.unsloth_mid_training.train --dataset_path data/midtrain/general_spec/dataset.jsonl
    python -m src.unsloth_mid_training.train --config configs/msm_config.json
"""

import argparse
import json
import sys
from pathlib import Path

from src.unsloth_training.trainer import MSMTrainingConfig, run_training


def parse_args() -> MSMTrainingConfig:
    parser = argparse.ArgumentParser(
        description="Unsloth Model Spec Midtraining (MSM) — LoRA CPT on synthetic spec documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Config file shortcut --
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config file (overrides defaults, CLI args override config file)"
    )

    # -- Data --
    parser.add_argument("--dataset_path", type=str, help="Path to dataset.jsonl from MSM data generation")
    parser.add_argument("--dataset_text_field", type=str, help="Field name containing document text")

    # -- Model --
    parser.add_argument("--model_name", type=str, help="Base model (HuggingFace id or local path)")
    parser.add_argument("--max_seq_length", type=int, help="Max sequence length (4096 or 8192)")
    parser.add_argument("--load_in_4bit", type=lambda x: x.lower() == "true", help="Use 4-bit quantization")

    # -- LoRA --
    parser.add_argument("--lora_r", type=int, help="LoRA rank (paper: 64)")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha (paper: 128)")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")

    # -- Training --
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs (paper: 1)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (paper: 1e-4)")
    parser.add_argument("--lr_scheduler_type", type=str, help="LR scheduler (paper: cosine)")
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio (paper: 0.05)")
    parser.add_argument("--weight_decay", type=float, help="Weight decay (paper: 0.01)")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, help="Random seed")

    # -- Output --
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--logging_steps", type=int, help="Log every N steps")
    parser.add_argument("--push_to_hub", type=lambda x: x.lower() == "true", help="Push adapter to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, help="HuggingFace model id to push to")

    args = parser.parse_args()

    # Start with defaults
    config = MSMTrainingConfig()

    # Load JSON config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path) as f:
            cfg_dict = json.load(f)
        for k, v in cfg_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        print(f"[MSM] Loaded config from: {config_path}")

    # CLI args override config file
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None and hasattr(config, key):
            setattr(config, key, val)

    return config


def main():
    config = parse_args()

    print("=" * 60)
    print("  Unsloth Model Spec Midtraining (MSM)")
    print("  Paper: https://arxiv.org/abs/2605.02087")
    print("=" * 60)
    print(f"  Model     : {config.model_name}")
    print(f"  Dataset   : {config.dataset_path}")
    print(f"  LoRA      : r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Output    : {config.output_dir}")
    print("=" * 60)
    print()

    run_training(config)


if __name__ == "__main__":
    main()
