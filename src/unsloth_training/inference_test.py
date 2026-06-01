"""
Inference test cho Qwen3.5-4B sau MSM training — QC Telesales scenarios.
Load adapter từ outputs/msm/ensure_value_qwen35_4b/final_adapter/
"""

import unsloth  # noqa - must be first
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# ─── Config ───────────────────────────────────────────────
ADAPTER_PATH   = "outputs/msm/ensure_value_qwen35_4b/final_adapter"
MAX_NEW_TOKENS = 2048
TEMPERATURE    = 0.7

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

# ─── QC telesales test cases ──────────────────────────────
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

# ─── Load model ───────────────────────────────────────────
print(f"Loading adapter: {ADAPTER_PATH}")
model, _ = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
print("Model ready!\n" + "=" * 60)

# ─── Run inference ────────────────────────────────────────
for case in TEST_CASES:
    print(f"\n[{case['id']}] {case['label']}")
    print(f"Prompt:\n{case['prompt']}")
    print("-" * 40)

    prompt_text = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{case['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to("cuda")

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"Response:\n{response}")
    print("=" * 60)
