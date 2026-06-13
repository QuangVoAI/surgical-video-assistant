from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from random import Random

from src.data.cholec import parse_cholec_labels
from src.data.io import write_jsonl
from src.data.surgmllmbench import parse_surgmllmbench


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize surgical datasets into a shared JSONL schema.")
    parser.add_argument("--dataset", choices=["surgmllmbench", "cholec"], required=True)
    parser.add_argument("--raw", type=Path, required=True, help="Raw dataset directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--filename", default="processed_dataset.jsonl")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--drop-missing-images", action="store_true")
    args = parser.parse_args()

    if args.dataset == "surgmllmbench":
        samples = list(parse_surgmllmbench(args.raw))
    else:
        samples = list(parse_cholec_labels(args.raw))

    if args.drop_missing_images:
        samples = [sample for sample in samples if Path(sample.image_path).exists()]
    samples = ensure_eval_split(samples, args.validation_ratio, args.split_seed)

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


def ensure_eval_split(samples, validation_ratio: float, seed: int):
    split_names = {sample.split for sample in samples}
    if not samples or split_names & {"val", "validation", "test"}:
        return samples
    if validation_ratio <= 0:
        return samples

    rng = Random(seed)
    by_task = {}
    for sample in samples:
        by_task.setdefault(sample.task_type, []).append(sample)

    val_ids = set()
    for task_samples in by_task.values():
        shuffled = list(task_samples)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * validation_ratio))
        val_ids.update(sample.sample_id for sample in shuffled[:val_count])

    return [
        replace(sample, split="val") if sample.sample_id in val_ids else sample
        for sample in samples
    ]


if __name__ == "__main__":
    main()
