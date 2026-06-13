from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.cholec import parse_cholec_labels
from src.data.io import write_jsonl
from src.data.surgmllmbench import parse_surgmllmbench


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize surgical datasets into a shared JSONL schema.")
    parser.add_argument("--dataset", choices=["surgmllmbench", "cholec"], required=True)
    parser.add_argument("--raw", type=Path, required=True, help="Raw dataset directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--filename", default="processed_dataset.jsonl")
    args = parser.parse_args()

    if args.dataset == "surgmllmbench":
        samples = list(parse_surgmllmbench(args.raw))
    else:
        samples = list(parse_cholec_labels(args.raw))

    output_path = args.out / args.filename
    count = write_jsonl(samples, output_path)
    summary = summarize(samples)
    summary_path = args.out / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {count} samples to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize(samples) -> dict:
    missing_images = sum(1 for sample in samples if not Path(sample.image_path).exists())
    return {
        "num_samples": len(samples),
        "by_dataset": Counter(sample.dataset for sample in samples),
        "by_split": Counter(sample.split for sample in samples),
        "by_task_type": Counter(sample.task_type for sample in samples),
        "missing_images": missing_images,
        "examples": [sample.to_json() for sample in samples[:3]],
    }


if __name__ == "__main__":
    main()
