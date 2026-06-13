from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.data.io import iter_json_records
from src.eval.metrics import classification_scores, count_scores, exact_match, multilabel_scores, token_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate surgical VQA predictions.")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions JSONL.")
    parser.add_argument("--out", type=Path, required=True, help="Metrics JSON output.")
    args = parser.parse_args()

    rows = list(iter_json_records(args.pred))
    metrics = evaluate_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def evaluate_rows(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row.get("task_type", "vqa")].append(row)

    overall_exact = [exact_match(row["prediction"], row["ground_truth"]) for row in rows]
    overall_token_f1 = [token_f1(row["prediction"], row["ground_truth"]) for row in rows]
    result = {
        "num_samples": len(rows),
        "overall": {
            "exact_match": average(overall_exact),
            "token_f1": average(overall_token_f1),
        },
        "by_task": {},
    }

    for task_type, task_rows in by_task.items():
        if task_type == "phase":
            scores = classification_scores(task_rows)
        elif task_type == "tool_type":
            scores = multilabel_scores(task_rows)
        elif task_type == "tool_count":
            scores = count_scores(task_rows)
        elif task_type in {"action", "triplet"}:
            scores = classification_scores(task_rows)
        else:
            scores = {
                "exact_match": average(exact_match(row["prediction"], row["ground_truth"]) for row in task_rows),
                "token_f1": average(token_f1(row["prediction"], row["ground_truth"]) for row in task_rows),
            }
        scores["num_samples"] = len(task_rows)
        result["by_task"][task_type] = scores
    return result


def average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
