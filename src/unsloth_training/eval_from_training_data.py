"""
Eval inference: lấy data từ training dataset thật, chạy model gen, so sánh vs ground truth label.

Usage:
    # Dùng checkpoint vừa train:
    python src/unsloth_training/eval_from_training_data.py \\
        --adapter checkpoints/sft/qwen3-thamdo-thinking/checkpoint-87

    # Chọn số sample + seed ngẫu nhiên:
    python src/unsloth_training/eval_from_training_data.py \\
        --adapter checkpoints/sft/qwen3-thamdo-thinking/checkpoint-87 \\
        --n_samples 10 --seed 42

    # Lấy index cụ thể trong dataset:
    python src/unsloth_training/eval_from_training_data.py \\
        --adapter checkpoints/sft/qwen3-thamdo-thinking/checkpoint-87 \\
        --indices 0 5 12 20

    # Lưu kết quả ra file:
    python src/unsloth_training/eval_from_training_data.py \\
        --adapter checkpoints/sft/qwen3-thamdo-thinking/checkpoint-87 \\
        --output eval_results.jsonl
"""

import argparse
import json
import random
import re
import sys
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_ADAPTER  = "checkpoints/sft/qwen3-thamdo-thinking/checkpoint-87"
DATASET_NAME     = "ChaosAIVision/Ensure-Thamdo-1000-training"
DATASET_SPLIT    = "train"
MAX_NEW_TOKENS   = 4096
THINKING_BUDGET  = 1500  # Max tokens cho thinking block trước khi force </think>
ANSWER_TOKENS    = 1024  # Max tokens cho JSON answer sau </think>
NO_THINKING      = False  # Override bằng --no_thinking flag
TEMPERATURE      = 0.3

THINK_BLOCK_RE      = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
THINK_BLOCK_FULL_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
JSON_BLOCK_RE       = re.compile(r"(\[.*?\]|\{.*?\})", flags=re.DOTALL)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(adapter_path: str):
    """Load base model + LoRA adapter (4-bit) via HuggingFace PEFT."""
    import os
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    print(f"[INFO] Base model : {base_model_name}")
    print(f"[INFO] Adapter    : {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("[INFO] Model ready!\n" + "=" * 70)
    return model, tokenizer


def extract_messages(row: dict) -> list[dict]:
    """Trả về list messages từ row dataset."""
    msgs = row.get("messages") or row.get("conversations") or []
    return msgs


def split_messages(messages: list[dict]):
    """
    Tách messages thành:
      - context: tất cả turn trừ assistant cuối cùng
      - label  : nội dung assistant cuối cùng (ground truth)
    """
    if not messages:
        return [], ""

    # Tìm assistant turn cuối
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        role = messages[i].get("role", "")
        if role == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return messages, ""

    context = messages[:last_assistant_idx]
    label   = messages[last_assistant_idx].get("content", "")
    return context, label


def strip_think(text: str) -> str:
    """Bỏ <think>...</think> block, chỉ lấy phần answer (tagged format)."""
    return THINK_BLOCK_FULL_RE.sub("", text).strip()


def extract_think(text: str) -> str:
    """Lấy nội dung bên trong <think>...</think>."""
    m = THINK_BLOCK_RE.search(text)
    return m.group(1).strip() if m else ""


def extract_json_answer(raw: str) -> tuple[str, str]:
    """
    Tách thinking text và JSON answer từ raw output của model.
    Qwen3 có 2 kiểu output:
      1) Có <think>...</think> tags  → tagged
      2) Không có tags nhưng thinking text chạy trước JSON  → untagged

    Returns: (answer_json_str, thinking_str)
    """
    # Kiểu 1: có tags
    if "<think>" in raw:
        think_content = extract_think(raw)
        answer        = strip_think(raw)
        return answer, think_content

    # Kiểu 2: không có tags - tìm JSON block […] hoặc {…} đầu tiên
    # Tìm vị trí bắt đầu của '[' hoặc '{' đầu tiên
    json_start = -1
    for i, ch in enumerate(raw):
        if ch in ("[", "{"):
            json_start = i
            break

    if json_start == -1:
        # Không có JSON nào cả → toàn bộ là thinking
        return "", raw.strip()

    thinking = raw[:json_start].strip()
    answer   = raw[json_start:].strip()
    return answer, thinking


def build_prompt_text(tokenizer, context_messages: list[dict], no_thinking: bool = False) -> str:
    """Apply chat template cho context (chưa có assistant response)."""
    if no_thinking:
        # Không dùng enable_thinking → model output JSON trực tiếp (non-thinking mode)
        return tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        # Thinking mode: cue <think>\n → model suy nghĩ trước rồi output JSON
        return tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )


def generate(model, tokenizer, prompt_text: str, no_thinking: bool = False) -> str:
    """
    Two-phase generation:
    - Phase 1: generate thinking block (cẩp đến THINKING_BUDGET tokens), đưa </think>
    - Phase 2: generate JSON answer (cẩp đến ANSWER_TOKENS tokens)
    """
    def _encode(text):
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    def _decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)

    # ── Phase 1: Thinking (chỉ khi thinking mode) ──────────────────────────
    if not no_thinking:
        inputs = _encode(prompt_text)   # prompt kết thúc bằng <think>\n
        with torch.no_grad():
            out1 = model.generate(
                **inputs,
                max_new_tokens=THINKING_BUDGET,
                temperature=TEMPERATURE,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Lấy thinking tokens mới gen
        think_tokens = out1[0][inputs["input_ids"].shape[-1]:]
        think_text   = _decode(think_tokens)
        # Xây dựng full thinking block
        # prompt_text đã có <think>\n ở cuối, think_text là nội dung tiếp theo
        thinking_block = "<think>\n" + think_text.strip() + "\n</think>\n"

        # ── Phase 2: Answer ──────────────────────────────────────────────────
        # Xây dựng prompt mới: prompt_gốc (bỏ cue <think>) + full thinking block
        # Bỏ <think>\n khỏi cuối prompt_text để thay bằng thinking_block hoàn chỉnh
        base_prompt = prompt_text
        if base_prompt.endswith("<think>\n"):
            base_prompt = base_prompt[:-len("<think>\n")]
        phase2_prompt = base_prompt + thinking_block

        inputs2 = _encode(phase2_prompt)
        with torch.no_grad():
            out2 = model.generate(
                **inputs2,
                max_new_tokens=ANSWER_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer_tokens = out2[0][inputs2["input_ids"].shape[-1]:]
        answer_text   = _decode(answer_tokens)
        # Strip special tokens
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            answer_text = answer_text.replace(tok, "")
        answer_text = answer_text.strip()

        raw = thinking_block + answer_text
        return raw

    else:
        # No-thinking mode: gen trực tiếp
        inputs = _encode(prompt_text)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=ANSWER_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        raw = _decode(new_tokens)
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            raw = raw.replace(tok, "")
        return raw.strip()


def print_comparison(idx: int, context: list[dict], label: str, prediction: str):
    """In kết quả so sánh label vs prediction ra terminal."""
    print(f"\n{'='*70}")
    print(f"[SAMPLE #{idx}]")

    # In context messages (truncate dài)
    for msg in context:
        role    = msg.get("role", "?").upper()
        content = (msg.get("content") or "").strip()
        if role == "USER" and len(content) > 500:
            content = content[:500] + " ...(truncated)"
        print(f"\n  [{role}]\n  {content}")

    # Ground truth label
    label_answer, label_think = extract_json_answer(label)
    print(f"\n  {'─'*60}")
    print(f"  [LABEL — ANSWER (ground truth)]")
    print(f"  {label_answer[:600] if label_answer else '(empty)'}")
    if label_think:
        print(f"\n  [LABEL — <think> first 8 lines]")
        for line in label_think.splitlines()[:8]:
            print(f"    {line}")

    # Model prediction
    pred_answer, pred_think = extract_json_answer(prediction)
    print(f"\n  {'─'*60}")
    print(f"  [PREDICTED — ANSWER]")
    if pred_answer:
        print(f"  {pred_answer[:600]}")
    else:
        print(f"  (EMPTY — model only generated thinking, no JSON found)")
    if pred_think:
        print(f"\n  [PREDICTED — <think> first 8 lines]")
        for line in pred_think.splitlines()[:8]:
            print(f"    {line}")
        if len(pred_think.splitlines()) > 8:
            print(f"    ... ({len(pred_think.splitlines())} lines total)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eval inference from training data")
    parser.add_argument(
        "--adapter", default=DEFAULT_ADAPTER,
        help="Path tới adapter/checkpoint folder"
    )
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Số sample ngẫu nhiên lấy từ dataset (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=2025,
        help="Random seed để tái lập kết quả (default: 2025)"
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", default=None,
        help="Danh sách index cụ thể trong dataset (ưu tiên hơn --n_samples)"
    )
    parser.add_argument(
        "--dataset", default=DATASET_NAME,
        help=f"HuggingFace dataset name (default: {DATASET_NAME})"
    )
    parser.add_argument(
        "--split", default=DATASET_SPLIT,
        help=f"Dataset split (default: {DATASET_SPLIT})"
    )
    parser.add_argument(
        "--output", default=None,
        help="Lưu kết quả ra file .jsonl (optional)"
    )
    parser.add_argument(
        "--no_thinking", action="store_true",
        help="Tắt thinking mode: model output JSON trực tiếp không qua <think> block"
    )
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading dataset: {args.dataset} / {args.split} ...")
    dataset = load_dataset(args.dataset, split=args.split)
    total = len(dataset)
    print(f"[INFO] Total samples: {total}")

    # Chọn indices
    if args.indices:
        indices = args.indices
    else:
        random.seed(args.seed)
        indices = random.sample(range(total), min(args.n_samples, total))

    print(f"[INFO] Selected indices: {indices}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.adapter)

    # ── Inference loop ────────────────────────────────────────────────────────
    results = []

    for idx in indices:
        row      = dataset[idx]
        messages = extract_messages(row)

        if not messages:
            print(f"[WARN] Sample #{idx} has no messages, skipping.")
            continue

        context, label = split_messages(messages)

        if not context:
            print(f"[WARN] Sample #{idx} has no context messages, skipping.")
            continue

        # Build prompt & generate
        prompt_text = build_prompt_text(tokenizer, context, no_thinking=args.no_thinking)
        prediction  = generate(model, tokenizer, prompt_text, no_thinking=args.no_thinking)

        # In so sánh
        print_comparison(idx, context, label, prediction)

        # Lưu kết quả
        pred_answer, pred_think = extract_json_answer(prediction)
        lbl_answer,  lbl_think  = extract_json_answer(label)
        results.append({
            "index"             : idx,
            "context"           : context,
            "label_raw"         : label,
            "label_answer"      : lbl_answer,
            "label_think"       : lbl_think,
            "prediction_raw"    : prediction,
            "prediction_answer" : pred_answer,
            "prediction_think"  : pred_think,
        })

    # ── Lưu file nếu cần ─────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[INFO] Results saved to: {args.output}")

    print(f"\n[DONE] Evaluated {len(results)}/{len(indices)} samples.")


if __name__ == "__main__":
    main()
