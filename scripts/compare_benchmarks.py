from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gemma evaluation outputs against paper baselines.")
    parser.add_argument("--paper-baselines", type=Path, default=Path("reports/paper_baselines.template.json"))
    parser.add_argument("--gemma12", type=Path, help="Metrics JSON for Gemma 4 12B.")
    parser.add_argument("--gemma26", type=Path, help="Metrics JSON for Gemma 4 26B-A4B.")
    parser.add_argument("--out", type=Path, default=Path("reports/benchmark_comparison.md"))
    args = parser.parse_args()

    baselines = json.loads(args.paper_baselines.read_text(encoding="utf-8"))
    gemma12 = load_metrics(args.gemma12) if args.gemma12 else {}
    gemma26 = load_metrics(args.gemma26) if args.gemma26 else {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(baselines, gemma12, gemma26), encoding="utf-8")
    print(f"Wrote {args.out}")


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown(baselines: dict, gemma12: dict, gemma26: dict) -> str:
    lines = [
        "# Benchmark Comparison",
        "",
        "This table is only valid when the dataset split and metric protocol match the cited paper.",
        "",
        "| Task | Dataset | Paper baseline | Metric | Paper score | Gemma 4 12B | Gemma 4 26B-A4B | Citation |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in baselines.get("baselines", []):
        task = item["task"]
        metric_name = item["metric_name"]
        gemma12_value = select_model_score(gemma12, task)
        gemma26_value = select_model_score(gemma26, task)
        lines.append(
            "| {task} | {dataset} | {model} | {metric} | {paper_score} | {g12} | {g26} | {citation} |".format(
                task=task,
                dataset=item["dataset"],
                model=item["model"],
                metric=metric_name,
                paper_score=format_score(item.get("metric_value")),
                g12=format_score(gemma12_value),
                g26=format_score(gemma26_value),
                citation=item["citation"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Gemma 4 large model in this repo is `google/gemma-4-26B-A4B-it`, which is the official 26B-A4B MoE variant.",
            "- Fill the `metric_value` fields from the exact experiment setting you cite before using this table in a report.",
        ]
    )
    return "\n".join(lines) + "\n"


def select_model_score(metrics: dict, task: str):
    task_metrics = metrics.get("by_task", {}).get(task, {})
    return task_metrics.get("accuracy", task_metrics.get("exact_match", task_metrics.get("micro_f1")))


def format_score(value) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
