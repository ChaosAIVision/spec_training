#!/bin/bash
# Run Unsloth Model Spec Midtraining (MSM)
# Based on paper: https://arxiv.org/abs/2605.02087 (Appendix B.4)
#
# Prerequisites:
#   1. Generate MSM data first: bash exps/generate_msm_data.sh
#   2. Install: pip install unsloth trl transformers datasets

# ─── Data ────────────────────────────────────────────────────────
DATASET_NAME="general_spec"
DATASET_PATH="data/midtrain/${DATASET_NAME}/dataset.jsonl"

# ─── Model ───────────────────────────────────────────────────────
MODEL_NAME="unsloth/Llama-3.1-8B"      # Base model (change to your target)
# MODEL_NAME="unsloth/Qwen2.5-7B"      # Alternative: Qwen2.5-7B
# MODEL_NAME="unsloth/Qwen2.5-14B"     # Alternative: 14B (needs 2x GPU)
# MODEL_NAME="unsloth/Qwen3-32B"       # Alternative: 32B (needs 4x GPU)
LOAD_IN_4BIT="true"                    # Set false for full bf16 LoRA
MAX_SEQ_LENGTH=4096                    # Use 8192 if mixing long-context IT data

# ─── LoRA (paper: r=64, alpha=128) ───────────────────────────────
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

# ─── Training Hyperparameters (paper Appendix B.4) ───────────────
NUM_EPOCHS=1
LEARNING_RATE=1e-4
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
BATCH_SIZE=2
GRAD_ACCUM=4
SEED=42

# ─── Output ──────────────────────────────────────────────────────
OUTPUT_DIR="outputs/msm/${DATASET_NAME}"
LOGGING_STEPS=10
PUSH_TO_HUB="false"
HUB_MODEL_ID=""                        # e.g. "ChaosAIVision/Llama-3.1-8B-MSM"

# ─────────────────────────────────────────────────────────────────

echo "================================================"
echo "  Unsloth Model Spec Midtraining (MSM)"
echo "  Model   : ${MODEL_NAME}"
echo "  Dataset : ${DATASET_PATH}"
echo "  Output  : ${OUTPUT_DIR}"
echo "================================================"

python -m src.unsloth_mid_training.train \
    --dataset_path       "${DATASET_PATH}" \
    --model_name         "${MODEL_NAME}" \
    --max_seq_length     ${MAX_SEQ_LENGTH} \
    --load_in_4bit       "${LOAD_IN_4BIT}" \
    --lora_r             ${LORA_R} \
    --lora_alpha         ${LORA_ALPHA} \
    --lora_dropout       ${LORA_DROPOUT} \
    --num_train_epochs   ${NUM_EPOCHS} \
    --learning_rate      ${LEARNING_RATE} \
    --lr_scheduler_type  "${LR_SCHEDULER}" \
    --warmup_ratio       ${WARMUP_RATIO} \
    --weight_decay       ${WEIGHT_DECAY} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --seed               ${SEED} \
    --output_dir         "${OUTPUT_DIR}" \
    --logging_steps      ${LOGGING_STEPS} \
    --push_to_hub        "${PUSH_TO_HUB}" \
    --hub_model_id       "${HUB_MODEL_ID}"
