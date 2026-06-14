from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import read_samples, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small balanced JSONL subset for quick LoRA experiments.")
    parser.add_argument("--input", type=Path, required=True, help="Source processed JSONL.")
    parser.add_argument("--out", type=Path, required=True, help="Output subset JSONL.")
    parser.add_argument("--train-per-task", type=int, default=300)
    parser.add_argument("--eval-per-task", type=int, default=80)
    parser.add_argument("--train-videos", nargs="*", default=None, help="Optional video IDs for training, e.g. VID05 VID08.")
    parser.add_argument("--eval-videos", nargs="*", default=None, help="Optional video IDs for validation/test.")
    parser.add_argument("--task-types", nargs="*", default=None, help="Optional task types filter, e.g. phase tool_type.")
    parser.add_argument(
        "--balance-by-answer-tasks",
        nargs="*",
        default=None,
        help="Task types that should be balanced by answer label within each split.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples = read_samples(args.input)
    if args.require_images:
        samples = [sample for sample in samples if Path(sample.image_path).exists()]
    if args.train_videos or args.eval_videos:
        samples = filter_by_videos(samples, set(args.train_videos or []), set(args.eval_videos or []))
    if args.task_types:
        allowed_tasks = set(args.task_types)
        samples = [sample for sample in samples if sample.task_type in allowed_tasks]

    buckets = defaultdict(list)
    for sample in samples:
        split_group = "train" if sample.split == "train" else "eval"
        buckets[(split_group, sample.task_type)].append(sample)

    balanced_tasks = set(args.balance_by_answer_tasks or [])
    selected = []
    for (split_group, task_type), task_samples in sorted(buckets.items()):
        rng.shuffle(task_samples)
        limit = args.train_per_task if split_group == "train" else args.eval_per_task
        if task_type in balanced_tasks:
            selected.extend(select_balanced_by_answer(task_samples, limit, rng))
        else:
            selected.extend(task_samples[:limit])

    rng.shuffle(selected)
    count = write_jsonl(selected, args.out)
    summary = summarize(selected)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {count} samples to {args.out}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize(samples) -> dict:
    return {
        "num_samples": len(samples),
        "by_split": Counter(sample.split for sample in samples),
        "by_task_type": Counter(sample.task_type for sample in samples),
        "by_video": Counter((sample.metadata or {}).get("video_id", "unknown") for sample in samples),
        "missing_images": sum(1 for sample in samples if not Path(sample.image_path).exists()),
        "examples": [sample.to_json() for sample in samples[:3]],
    }


def filter_by_videos(samples, train_videos: set[str], eval_videos: set[str]):
    selected = []
    for sample in samples:
        video_id = str((sample.metadata or {}).get("video_id", ""))
        if sample.split == "train" and train_videos and video_id not in train_videos:
            continue
        if sample.split != "train" and eval_videos and video_id not in eval_videos:
            continue
        selected.append(sample)
    return selected


def select_balanced_by_answer(task_samples, limit: int, rng: random.Random):
    if limit <= 0:
        return []

    by_answer = defaultdict(list)
    for sample in task_samples:
        by_answer[str(sample.answer)].append(sample)
    for samples_for_answer in by_answer.values():
        rng.shuffle(samples_for_answer)

    labels = sorted(by_answer)
    base = limit // len(labels) if labels else 0
    remainder = limit % len(labels) if labels else 0

    selected = []
    leftovers = []
    for index, label in enumerate(labels):
        quota = base + (1 if index < remainder else 0)
        picked = by_answer[label][:quota]
        selected.extend(picked)
        leftovers.extend(by_answer[label][quota:])

    if len(selected) < limit:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])
    return selected[:limit]


if __name__ == "__main__":
    main()
