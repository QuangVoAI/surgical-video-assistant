from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final charts and a presentation report from training artifacts.")
    parser.add_argument("--training-log", type=Path, default=Path("reports/training_log.csv"))
    parser.add_argument("--train-stdout", type=Path, default=Path("reports/train_gemma4_12b_video_subset.log"))
    parser.add_argument("--dataset-summary", type=Path, default=Path("data/processed/processed_dataset_video_subset.summary.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/gemma4_12b_lora_video_subset.yaml"))
    parser.add_argument("--hf-repo", default="https://huggingface.co/SpringWang08/surgical-gemma4-12b-lora")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/final"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_training_rows(args.training_log)
    summary = read_json(args.dataset_summary)
    stdout_text = args.train_stdout.read_text(encoding="utf-8", errors="replace") if args.train_stdout.exists() else ""
    final_metrics = parse_final_metrics(stdout_text)

    figures = build_charts(rows, summary, args.out_dir)
    report_path = args.out_dir / "final_report.md"
    report_path.write_text(
        render_report(
            summary=summary,
            final_metrics=final_metrics,
            figures=figures,
            hf_repo=args.hf_repo,
            config_path=args.config,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    for figure in figures:
        print(f"Wrote {figure}")


def read_training_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_final_metrics(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for key in ("train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss", "epoch"):
        match = re.search(rf"'{key}': '([^']+)'", text)
        if match:
            metrics[key] = match.group(1)
    eval_match = re.findall(r"\{'eval_loss': '([^']+)'.*?'eval_mean_token_accuracy': '([^']+)'.*?'epoch': '([^']+)'\}", text)
    if eval_match:
        eval_loss, eval_acc, eval_epoch = eval_match[-1]
        metrics["eval_loss"] = eval_loss
        metrics["eval_mean_token_accuracy"] = eval_acc
        metrics["eval_epoch"] = eval_epoch
    return metrics


def build_charts(rows: list[dict[str, float]], summary: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    figures = []
    if rows:
        figures.append(plot_lines(
            rows,
            out_dir / "training_loss_curve.png",
            [("loss", "Train loss"), ("eval_loss", "Eval loss")],
            "Training and Evaluation Loss",
            "Loss",
        ))
        figures.append(plot_lines(
            rows,
            out_dir / "token_accuracy_curve.png",
            [("mean_token_accuracy", "Train token accuracy"), ("eval_mean_token_accuracy", "Eval token accuracy")],
            "Mean Token Accuracy",
            "Accuracy",
        ))
        figures.append(plot_lines(
            rows,
            out_dir / "learning_rate_curve.png",
            [("learning_rate", "Learning rate")],
            "Learning Rate Schedule",
            "Learning rate",
        ))

    by_task = summary.get("by_task_type") or {}
    if by_task:
        figures.append(plot_bar(by_task, out_dir / "dataset_task_distribution.png", "Dataset Task Distribution", "Samples"))

    by_split = summary.get("by_split") or {}
    if by_split:
        figures.append(plot_bar(by_split, out_dir / "dataset_split_distribution.png", "Dataset Split Distribution", "Samples"))

    by_video = summary.get("by_video") or {}
    if by_video:
        figures.append(plot_bar(dict(list(by_video.items())[:12]), out_dir / "dataset_video_distribution.png", "Video-Level Subset Distribution", "Samples"))

    plt.close("all")
    return figures


def plot_lines(rows: list[dict[str, float]], path: Path, series: list[tuple[str, str]], title: str, ylabel: str) -> Path:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for key, label in series:
        points = [(row.get("step", index), row[key]) for index, row in enumerate(rows) if key in row and math.isfinite(row[key])]
        if points:
            plt.plot([x for x, _ in points], [y for _, y in points], marker="o", linewidth=1.8, markersize=3, label=label)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_bar(values: dict[str, Any], path: Path, title: str, ylabel: str) -> Path:
    import matplotlib.pyplot as plt

    labels = list(values.keys())
    counts = [int(values[label]) for label in labels]
    plt.figure(figsize=(max(7, len(labels) * 0.8), 5))
    plt.bar(labels, counts, color="#2f6fed")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def render_report(
    summary: dict[str, Any],
    final_metrics: dict[str, str],
    figures: list[Path],
    hf_repo: str,
    config_path: Path,
) -> str:
    lines = [
        "# Surgical Video Assistant Final Report",
        "",
        "## Objective",
        "",
        "Build a frame-level multimodal surgical assistant for phase, tool, action, and triplet recognition without full video training.",
        "",
        "## Method",
        "",
        "- Base model: Gemma 4 12B multimodal model.",
        "- Fine-tuning: QLoRA/LoRA supervised fine-tuning on surgical frame-question-answer pairs.",
        "- Dataset: CholecT50 frame labels converted into QA templates.",
        "- Split strategy: video-level subset, keeping train and eval procedures separate.",
        "- Inference for label tasks: constrained prediction over valid label candidates.",
        "",
        "## Training Result",
        "",
    ]
    if final_metrics:
        for key, value in final_metrics.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- Final metrics were not found in the stdout log.")

    lines.extend([
        "",
        "## Dataset Subset",
        "",
        f"- Number of samples: `{summary.get('num_samples', 'unknown')}`",
        f"- Missing images: `{summary.get('missing_images', 'unknown')}`",
        "",
        "### By Split",
        "",
        markdown_table(summary.get("by_split") or {}),
        "",
        "### By Task",
        "",
        markdown_table(summary.get("by_task_type") or {}),
        "",
        "### By Video",
        "",
        markdown_table(summary.get("by_video") or {}),
        "",
        "## Figures",
        "",
    ])
    for figure in figures:
        lines.append(f"- `{figure}`")

    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Hugging Face checkpoint: {hf_repo}",
        f"- Training config: `{config_path}`",
        "- Local checkpoint: `checkpoints/gemma4-12b-surgical-frame-lora-video-subset/checkpoint-500`",
        "- Training logs: `reports/training_log.csv` and `reports/train_gemma4_12b_video_subset.log`",
        "",
        "## Limitations",
        "",
        "- This is a research prototype, not clinical decision support.",
        "- The experiment uses a video-level subset, so it does not claim full CholecT50 leaderboard performance.",
        "- The model is frame-level and does not model long temporal surgical dynamics.",
        "- Raw CholecT50 data is not redistributed because of dataset license restrictions.",
        "",
        "## Next Steps",
        "",
        "- Run constrained evaluation on the full official split if more GPU/storage budget is available.",
        "- Add temporal smoothing over adjacent frames for phase recognition.",
        "- Add SSG-VQA alignment after verifying frame coverage from CholecT50/CholecT45.",
    ])
    return "\n".join(lines) + "\n"


def markdown_table(values: dict[str, Any]) -> str:
    if not values:
        return "No data."
    lines = ["| Item | Count |", "| --- | ---: |"]
    for key, value in values.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
