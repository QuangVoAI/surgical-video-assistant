from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from tqdm.auto import tqdm

from scripts.predict_lora import default_choices, score_answer
from src.data.io import iter_json_records
from src.data.schema import SurgicalSample
from src.models.prompts import build_prompt
from src.models.providers import load_multimodal_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run constrained zero-shot predictions over an eval split for benchmark metrics."
    )
    parser.add_argument("--data", type=Path, required=True, help="Processed dataset JSONL.")
    parser.add_argument("--out", type=Path, required=True, help="Prediction JSONL output.")
    parser.add_argument("--model", default="google/gemma-4-12B")
    parser.add_argument("--processor", default="google/gemma-4-12B-it")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Splits to evaluate.")
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=["phase"],
        help="Tasks to evaluate. Start with phase for a fast benchmark.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip rows whose image files are missing instead of failing.",
    )
    args = parser.parse_args()

    samples = [SurgicalSample.from_json(row) for row in iter_json_records(args.data)]
    selected = [
        sample
        for sample in samples
        if sample.split in set(args.splits) and sample.task_type in set(args.task_types)
    ]
    if not selected:
        raise SystemExit("No samples matched the requested splits/task types.")

    candidate_map = build_candidate_map(samples, selected)
    print("benchmark_samples:", len(selected))
    print("tasks:", Counter(sample.task_type for sample in selected))
    print("candidates:", {task: len(values) for task, values in candidate_map.items()})

    model, processor = load_base_model(args.model, args.processor)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    missing_images = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for sample in tqdm(selected, desc="Predicting"):
            image_path = Path(sample.image_path)
            if not image_path.exists():
                missing_images += 1
                if args.skip_missing_images:
                    continue
                raise FileNotFoundError(image_path)

            image = Image.open(image_path).convert("RGB")
            prompt = build_prompt(sample, include_system=False)
            scored = [
                (score_answer(model, processor, image, prompt, candidate), candidate)
                for candidate in candidate_map[sample.task_type]
            ]
            scored.sort(key=lambda item: item[0] if math.isfinite(item[0]) else float("inf"))
            prediction = scored[0][1]
            output = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "image_path": sample.image_path,
                "question": sample.question,
                "ground_truth": sample.answer,
                "prediction": prediction,
                "task_type": sample.task_type,
                "split": sample.split,
                "prompt_type": "zero_shot_constrained",
                "model": args.model,
                "top_scores": [
                    {"answer": answer, "loss": score if math.isfinite(score) else None}
                    for score, answer in scored[:7]
                ],
                "metadata": sample.metadata or {},
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    if missing_images:
        print(f"Skipped missing images: {missing_images}")
    print(f"Wrote predictions to {args.out}")


def build_candidate_map(
    all_samples: list[SurgicalSample],
    selected_samples: list[SurgicalSample],
) -> dict[str, list[str]]:
    task_types = sorted({sample.task_type for sample in selected_samples})
    candidate_map: dict[str, list[str]] = {}
    for task_type in task_types:
        counts = Counter(
            sample.answer.strip()
            for sample in all_samples
            if sample.task_type == task_type and sample.answer.strip()
        )
        candidates = [answer for answer, _ in counts.most_common()] or default_choices(task_type)
        if not candidates:
            raise ValueError(f"No candidates found for task {task_type}.")
        candidate_map[task_type] = candidates
    return candidate_map


def load_base_model(model_name: str, processor_name: str):
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: python -m pip install -r requirements-train.txt") from exc

    processor = AutoProcessor.from_pretrained(processor_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = load_multimodal_model(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
    )
    model.eval()
    return model, processor


if __name__ == "__main__":
    main()
