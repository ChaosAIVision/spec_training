"""
HuggingFace inference cho Qwen3.5-4B LoRA adapter.
Không dùng Unsloth — pure transformers + peft.

Usage:
    python src/unsloth_training/hf_inference.py
    python src/unsloth_training/hf_inference.py --adapter outputs/msm/ensure_value_qwen35_4b/checkpoint-13
    python src/unsloth_training/hf_inference.py --prompt "câu hỏi của bạn"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ─── Config ───────────────────────────────────────────────
DEFAULT_ADAPTER  = "outputs/msm/ensure_value_qwen35_4b/final_adapter"
MAX_NEW_TOKENS   = 2048
TEMPERATURE      = 0.3

SYSTEM = (
    "Bạn là telesales Ensure (Abbott) đang gọi cho khách hàng mới (NU - New User) "
    "kênh Digital (DGT). Nguyên tắc: sức khỏe trước, không dùng áp lực sai chuẩn.\n\n"
    "Quy tắc xử lý từ chối:\n"
    "- Nếu khách nói BẬN/không tiện/gọi lại sau: hỏi thời điểm callback cụ thể rồi kết thúc cuộc gọi. "
    "KHÔNG tiếp tục hỏi sức khỏe hay giới thiệu sản phẩm thêm.\n"
    "- Nếu khách từ chối thật sự (không cần, đắt, còn nhiều...): áp dụng 3 lớp: "
    "làm dịu → làm rõ lý do gốc → làm hài lòng với CTA mềm.\n"
    "- Nếu khách cúp máy: kết thúc, không gọi lại ngay."
)

# ─── QC test cases ────────────────────────────────────────
TEST_CASES = [
    {
        "id": "QC-01",
        "label": "Khách nói đang bận",
        "prompt": (
            "Telesale: 'Dạ em gọi từ Ensure Abbott, muốn giới thiệu chương trình dinh dưỡng Ensure Gold ạ.'\n"
            "Khách: 'Ừ nhưng tôi đang bận lắm, gọi lại sau đi.'"
        ),
    },
    {
        "id": "QC-02",
        "label": "Khách còn nhiều chưa hết",
        "prompt": (
            "Telesale: 'Dạ em gọi follow-up xem chị dùng Ensure tháng trước thế nào ạ.'\n"
            "Khách: 'Tôi còn nhiều lắm chưa hết đâu, mua thêm làm gì.'"
        ),
    },
    {
        "id": "QC-03",
        "label": "Khách nói đắt",
        "prompt": (
            "Telesale: 'Gói Ensure Gold 850g hiện tại đang có ưu đãi ạ.'\n"
            "Khách: 'Giá vậy mắc quá, tôi không có tiền mua đâu.'"
        ),
    },
    {
        "id": "QC-04",
        "label": "Khách nói không cần uống sữa",
        "prompt": (
            "Telesale: 'Ensure Gold rất tốt cho sức khỏe người lớn tuổi ạ.'\n"
            "Khách: 'Thôi tôi già rồi uống sữa làm gì, ăn cơm bình thường thôi.'"
        ),
    },
    {
        "id": "QC-05",
        "label": "Khách lo hàng giả",
        "prompt": (
            "Telesale: 'Chị có thể đặt Ensure qua kênh online của Abbott ạ.'\n"
            "Khách: 'Mấy cái này bán online đầy, tôi sợ hàng giả không dám mua.'"
        ),
    },
    {
        "id": "QC-06",
        "label": "Khách cúp máy ngay",
        "prompt": (
            "Telesale: 'Dạ em gọi từ Ensure Abbott ạ...'\n"
            "Khách: [Cúp máy không nói gì]"
        ),
    },
]


def load_model(adapter_path: str):
    """Load base model + LoRA adapter via HuggingFace PEFT."""
    # Đọc base model từ adapter config
    import json, os
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    print(f"Base model : {base_model_name}")
    print(f"Adapter    : {adapter_path}")

    # 4-bit quantization
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

    print("Model ready!\n" + "=" * 60)
    return model, tokenizer


def build_prompt(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(model, tokenizer, prompt_text: str) -> str:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="HuggingFace inference — Ensure LoRA adapter")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Path to adapter folder")
    parser.add_argument("--prompt", default=None, help="Single prompt (skip test cases)")
    parser.add_argument("--system", default=SYSTEM, help="Override system prompt")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)

    if args.prompt:
        # Single prompt mode
        prompt_text = build_prompt(args.system, args.prompt)
        response = generate(model, tokenizer, prompt_text)
        print(f"\nResponse:\n{response}")
    else:
        # Run all QC test cases
        for case in TEST_CASES:
            print(f"\n[{case['id']}] {case['label']}")
            print(f"Prompt:\n{case['prompt']}")
            print("-" * 40)

            prompt_text = build_prompt(args.system, case["prompt"])
            response = generate(model, tokenizer, prompt_text)

            print(f"Response:\n{response}")
            print("=" * 60)


if __name__ == "__main__":
    main()
