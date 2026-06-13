from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import read_samples


BASE_MODULES = ["PIL", "yaml", "tqdm"]
TRAIN_MODULES = ["torch", "transformers", "datasets", "peft", "trl", "bitsandbytes"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the project is ready for VM training.")
    parser.add_argument("--config", type=Path, default=Path("configs/gemma4_12b_lora_sft.yaml"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings.")
    args = parser.parse_args()

    warnings: list[str] = []
    errors: list[str] = []

    check_modules(BASE_MODULES, errors, required=True)
    check_modules(TRAIN_MODULES, warnings, required=False)
    config = load_config(args.config, errors)
    if config:
        check_training_config(config, errors, warnings)

    print_report(errors, warnings)
    if errors or (warnings and args.strict):
        raise SystemExit(1)


def check_modules(modules: list[str], messages: list[str], required: bool) -> None:
    for module in modules:
        if importlib.util.find_spec(module) is None:
            level = "Missing required module" if required else "Missing training module"
            messages.append(f"{level}: {module}")


def load_config(path: Path, errors: list[str]) -> dict | None:
    if not path.exists():
        errors.append(f"Training config not found: {path}")
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Could not parse config {path}: {exc}")
        return None


def check_training_config(config: dict, errors: list[str], warnings: list[str]) -> None:
    required_paths = [
        ("train_jsonl", Path(config["data"]["train_jsonl"])),
        ("eval_jsonl", Path(config["data"]["eval_jsonl"])),
    ]
    for label, path in required_paths:
        if not path.exists():
            errors.append(f"{label} does not exist: {path}")

    model_name = str(config.get("model", {}).get("name", ""))
    processor_name = str(config.get("model", {}).get("processor_name", ""))
    if not model_name.startswith("google/gemma-4-"):
        warnings.append(f"Model name does not look like Gemma 4: {model_name}")
    if processor_name and not processor_name.endswith("-it"):
        warnings.append(f"Processor should usually use an instruction model tokenizer: {processor_name}")

    train_path = Path(config["data"]["train_jsonl"])
    eval_path = Path(config["data"]["eval_jsonl"])
    if train_path.exists() and eval_path.exists():
        train_samples = read_samples(train_path)
        eval_samples = read_samples(eval_path)
        train_splits = set(config["data"].get("train_splits", ["train"]))
        eval_splits = set(config["data"].get("eval_splits", ["val", "test"]))
        train_filtered = [sample for sample in train_samples if sample.split in train_splits]
        eval_filtered = [sample for sample in eval_samples if sample.split in eval_splits]
        if not train_filtered:
            errors.append(f"No samples found for train_splits={sorted(train_splits)}")
        if not eval_filtered:
            errors.append(f"No samples found for eval_splits={sorted(eval_splits)}")
        overlap = {sample.sample_id for sample in train_filtered} & {sample.sample_id for sample in eval_filtered}
        if overlap:
            errors.append(f"Train/eval sample_id overlap detected: {len(overlap)} samples")
        missing_images = [
            sample.image_path
            for sample in train_filtered[:1000] + eval_filtered[:1000]
            if not Path(sample.image_path).exists()
        ]
        if missing_images:
            errors.append(f"Missing images detected, first path: {missing_images[0]}")


def print_report(errors: list[str], warnings: list[str]) -> None:
    result = {
        "status": "fail" if errors else "pass",
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not errors:
        print("Preflight passed. Remaining risk: real Gemma training still needs a GPU VM, HF access, and the full dataset.")


if __name__ == "__main__":
    main()
