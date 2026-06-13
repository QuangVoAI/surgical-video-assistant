from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a presentation-ready Markdown training report.")
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics_mock.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/training_summary.md"))
    parser.add_argument("--title", default="Surgical Video Assistant Training Summary")
    args = parser.parse_args()

    metrics = {}
    if args.metrics.exists():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(args.title, metrics), encoding="utf-8")
    print(f"Wrote {args.out}")


def render_report(title: str, metrics: dict) -> str:
    lines = [
        f"# {title}",
        "",
        "## Project Goal",
        "",
        "Build a feasible surgical frame assistant for VQA, phase recognition, tool recognition, and action recognition without training a full video model.",
        "",
        "## Method",
        "",
        "- Normalize surgical image/frame annotations into a common JSONL schema.",
        "- Run Gemma-style multimodal zero-shot/few-shot prompting.",
        "- Optionally fine-tune with LoRA/QLoRA on frame-level QA pairs.",
        "- Evaluate each task separately and save charts for presentation.",
        "",
        "## Metrics",
        "",
    ]

    if not metrics:
        lines.extend([
            "No metrics file found yet. Run inference/evaluation first.",
            "",
        ])
    else:
        lines.extend(render_metrics(metrics))

    lines.extend([
        "## Artifacts",
        "",
        "- `reports/figures/training_loss.png`: training/eval loss chart.",
        "- `reports/figures/task_scores.png`: task-level score chart.",
        "- `reports/training_log.csv`: step-level training logs.",
        "- `checkpoints/gemma4-surgical-frame-lora`: LoRA checkpoints.",
        "",
        "## Limitations",
        "",
        "- This is a research prototype, not clinical decision support.",
        "- Frame-level reasoning cannot fully capture temporal surgical context.",
        "- Dataset access and licensing may restrict public release of data and trained weights.",
    ])
    return "\n".join(lines) + "\n"


def render_metrics(metrics: dict) -> list[str]:
    lines = [
        f"- Number of evaluated samples: `{metrics.get('num_samples', 0)}`",
    ]
    overall = metrics.get("overall", {})
    if overall:
        lines.append(f"- Overall exact match: `{overall.get('exact_match', 0):.4f}`")
        lines.append(f"- Overall token F1: `{overall.get('token_f1', 0):.4f}`")

    by_task = metrics.get("by_task", {})
    if by_task:
        lines.extend(["", "| Task | Samples | Main score |", "| --- | ---: | ---: |"])
        for task, values in by_task.items():
            score = values.get("accuracy", values.get("exact_match", values.get("micro_f1", 0)))
            lines.append(f"| {task} | {values.get('num_samples', 0)} | {score:.4f} |")
        lines.append("")
    return lines


if __name__ == "__main__":
    main()
