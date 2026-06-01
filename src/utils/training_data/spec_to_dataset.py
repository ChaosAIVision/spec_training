"""
Convert spec .txt files directly to dataset.jsonl for MSM training.

Mỗi spec file = 1 document hoàn chỉnh (không cắt chunk).
Repeat N lần để tăng training signal.

Usage:
    python -m src.utils.training_data.spec_to_dataset \
        --spec_dir spec/ensure \
        --output_path data/midtrain/ensure_value/dataset.jsonl \
        --model_name Ensure \
        --provider_name Abbott \
        --repeat 10
"""

import argparse
import json
from pathlib import Path


def spec_to_dataset(
    spec_dir: str,
    output_path: str,
    model_name: str = "Ensure",
    provider_name: str = "Abbott",
    repeat: int = 10,
):
    """
    Convert all .txt spec files → dataset.jsonl.
    Mỗi file = 1 document nguyên vẹn, repeat N lần.
    """
    spec_dir = Path(spec_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec_files = sorted(spec_dir.glob("*.txt"))
    if not spec_files:
        raise FileNotFoundError(f"No .txt files found in {spec_dir}")

    print(f"[spec_to_dataset] Found {len(spec_files)} spec file(s):")

    docs = []
    for spec_file in spec_files:
        text = spec_file.read_text(encoding="utf-8").strip()
        text = text.replace("{model_name}", model_name)
        text = text.replace("{provider_name}", provider_name)
        docs.append(text)
        print(f"  - {spec_file.name}: {len(text):,} chars")

    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(repeat):
            for doc in docs:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
                total += 1

    print(f"\n[spec_to_dataset] Wrote {total} samples → {output_path}")
    print(f"  ({len(docs)} docs × repeat={repeat} = {total} total)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert spec .txt → dataset.jsonl (no chunking, no API needed)"
    )
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="Ensure")
    parser.add_argument("--provider_name", default="Abbott")
    parser.add_argument("--repeat", type=int, default=10,
                        help="Repeat mỗi doc N lần để tăng training signal")
    args = parser.parse_args()

    spec_to_dataset(
        spec_dir=args.spec_dir,
        output_path=args.output_path,
        model_name=args.model_name,
        provider_name=args.provider_name,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
