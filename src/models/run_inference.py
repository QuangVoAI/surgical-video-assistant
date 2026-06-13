from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from src.data.io import read_samples
from src.models.answer_space import attach_answer_spaces, build_answer_spaces, constrain_prediction
from src.models.prompts import build_prompt, select_few_shot_examples
from src.models.providers import build_provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frame-level surgical VQA inference.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_jsonl = Path(config["data"]["input_jsonl"])
    train_jsonl = Path(config["data"].get("train_jsonl") or input_jsonl)
    output_jsonl = Path(config["output"]["predictions_jsonl"])
    max_samples = config["data"].get("max_samples")
    few_shot_k = int(config.get("prompt", {}).get("few_shot_k") or 0)
    prompt_type = config.get("prompt", {}).get("prompt_type") or ("few_shot" if few_shot_k else "zero_shot")
    use_answer_space = bool(config.get("prompt", {}).get("use_answer_space", True))

    samples = read_samples(input_jsonl)
    if max_samples:
        samples = samples[: int(max_samples)]
    train_samples = read_samples(train_jsonl) if few_shot_k else []
    answer_spaces = build_answer_spaces(train_samples or samples) if use_answer_space else {}
    if answer_spaces:
        samples = attach_answer_spaces(samples, answer_spaces)
        if train_samples:
            train_samples = attach_answer_spaces(train_samples, answer_spaces)
    few_shot = select_few_shot_examples(train_samples, samples, few_shot_k) if few_shot_k else {}

    provider = build_provider(config["model"])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for sample in tqdm(samples, desc="inference"):
            examples = few_shot.get(sample.sample_id, [])
            prompt = build_prompt(sample, examples)
            prediction = provider.generate(sample, prompt)
            prediction = constrain_prediction(sample.task_type, prediction, answer_spaces)
            row = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "image_path": sample.image_path,
                "question": sample.question,
                "ground_truth": sample.answer,
                "prediction": prediction,
                "task_type": sample.task_type,
                "prompt_type": prompt_type,
                "few_shot_k": few_shot_k,
                "model": provider.name,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {output_jsonl}")


if __name__ == "__main__":
    main()
