#!/bin/bash
# Generate MSM data from spec/ensure/ensure_value_spec.txt
# Output: data/midtrain/ensure_value/dataset.jsonl

PREVIEW=true   # Set false để chạy full generation

SPEC_TYPE="default"
DATASET_NAME="ensure_value"
PRINCIPLE_NAME="providing accurate, ethical, and personalized nutrition consultation"
SPEC_FILE_NAME="ensure/ensure_value_spec"   # → spec/ensure/ensure_value_spec.txt
MODEL_NAME="Ensure"
PROVIDER_NAME="Abbott"
MODEL_ID="claude-opus-4-6"

N_DOC_TYPES=20
N_DOC_IDEAS=25

USE_BATCH_API=false
MAX_OUTPUT_TOKENS=64000
TEMPERATURE=1.0
ANTHROPIC_TAG="ANTHROPIC_API_KEY"
ANTHROPIC_BATCH_TAG="ANTHROPIC_BATCH_API_KEY"
OPENAI_TAG="OPENAI_API_KEY"
MAX_CONCURRENT=30

echo "================================================"
echo "  MSM Data Generation | Ensure Value Spec"
echo "  PREVIEW=${PREVIEW} — set to false for full run"
echo "================================================"

python src/msm/generate_data_from_spec.py \
    --dataset_name        "$DATASET_NAME" \
    --principle_name      "$PRINCIPLE_NAME" \
    --spec_file_name      "$SPEC_FILE_NAME" \
    --model_name          "$MODEL_NAME" \
    --provider_name       "$PROVIDER_NAME" \
    --model_id            "$MODEL_ID" \
    --n_doc_types         $N_DOC_TYPES \
    --n_doc_ideas         $N_DOC_IDEAS \
    --max_output_tokens   $MAX_OUTPUT_TOKENS \
    --temperature         $TEMPERATURE \
    --spec_type           "$SPEC_TYPE" \
    --anthropic_tag       "$ANTHROPIC_TAG" \
    --anthropic_batch_tag "$ANTHROPIC_BATCH_TAG" \
    --openai_tag          "$OPENAI_TAG" \
    --max_concurrent_requests $MAX_CONCURRENT \
    --use_batch_api       $USE_BATCH_API \
    --preview             $PREVIEW
