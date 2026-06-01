#!/bin/bash
# Run Unsloth MSM — Ensure spec, Qwen/Qwen3.5-4B
# Conda env: test
#
# Step 1: Generate MSM data  → bash exps/generate_msm_data.sh  (set SPEC_FILE_NAME=ensure/ensure_value_spec)
# Step 2: Run training       → bash exps/run_msm_ensure_qwen35.sh

set -e

CONDA_ENV="test"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "================================================"
echo "  MSM Training | Ensure Spec | Qwen3.5-4B"
echo "  Conda env : ${CONDA_ENV}"
echo "  Repo dir  : ${REPO_DIR}"
echo "================================================"

# Install repo if not already installed
conda run -n "$CONDA_ENV" pip install -e . -q

# Run training
conda run -n "$CONDA_ENV" python -m src.unsloth_mid_training.train \
    --config configs/msm_ensure_value_config.json
