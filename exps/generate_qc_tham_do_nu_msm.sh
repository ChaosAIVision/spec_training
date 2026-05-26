#!/bin/bash
# Generate MSM midtraining data from QC THAM DO NU rules spec (Ensure)

# Config
PREVIEW=true  # Set to false to run full generation

SPEC_TYPE="rules"
DATASET_NAME="ensure_qc_tham_do_nu"
PRINCIPLE_NAME="chấm điểm kỹ năng thăm dò NU DGT"
SPEC_FILE_NAME="ensure/qc_tham_do_nu_rules_spec"  # path under spec/
MODEL_NAME="EnsureQCAgent"
PROVIDER_NAME="Abbott Vietnam"
MODEL_ID="claude-opus-4-6-thinking"

N_DOC_TYPES=12
N_DOC_IDEAS=16

# API Configs
USE_BATCH_API=false
MAX_OUTPUT_TOKENS=8000
TEMPERATURE=1.0
ANTHROPIC_TAG="ANTHROPIC_API_KEY"
ANTHROPIC_BATCH_TAG="ANTHROPIC_BATCH_API_KEY"
OPENAI_TAG="OPENAI_API_KEY"
MAX_CONCURRENT=5

# Run generation
CMD="python src/msm/generate_data_from_spec.py \
    --dataset_name \"$DATASET_NAME\" \
    --principle_name \"$PRINCIPLE_NAME\" \
    --spec_file_name \"$SPEC_FILE_NAME\" \
    --model_name \"$MODEL_NAME\" \
    --provider_name \"$PROVIDER_NAME\" \
    --model_id \"$MODEL_ID\" \
    --n_doc_types $N_DOC_TYPES \
    --n_doc_ideas $N_DOC_IDEAS \
    --max_output_tokens $MAX_OUTPUT_TOKENS \
    --temperature $TEMPERATURE \
    --spec_type \"$SPEC_TYPE\" \
    --anthropic_tag \"$ANTHROPIC_TAG\" \
    --anthropic_batch_tag \"$ANTHROPIC_BATCH_TAG\" \
    --openai_tag \"$OPENAI_TAG\" \
    --max_concurrent_requests $MAX_CONCURRENT \
    --use_batch_api $USE_BATCH_API \
    --preview $PREVIEW"

eval $CMD
