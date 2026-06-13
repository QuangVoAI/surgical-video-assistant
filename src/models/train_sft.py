from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image

from src.data.io import read_samples
from src.models.prompts import build_prompt

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for frame-level surgical VQA.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train(config)


def train(config: dict) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training requires GPU dependencies. Install with: "
            "python -m pip install -r requirements-train.txt"
        ) from exc

    model_name = config["model"]["name"]
    processor = AutoProcessor.from_pretrained(model_name)
    quantization_config = None
    if config["model"].get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if config["training"].get("bf16", True) else torch.float16,
    )
    if config["model"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=list(config["lora"]["target_modules"]),
        task_type="CAUSAL_LM",
    )

    train_samples = read_samples(Path(config["data"]["train_jsonl"]))
    eval_samples = read_samples(Path(config["data"]["eval_jsonl"]))
    max_train = config["data"].get("max_train_samples")
    max_eval = config["data"].get("max_eval_samples")
    if max_train:
        train_samples = train_samples[: int(max_train)]
    if max_eval:
        eval_samples = eval_samples[: int(max_eval)]

    train_dataset = Dataset.from_list([sample_to_record(sample) for sample in train_samples])
    eval_dataset = Dataset.from_list([sample_to_record(sample) for sample in eval_samples])

    args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        logging_dir=config["training"].get("logging_dir", "reports/training_logs"),
        num_train_epochs=float(config["training"]["num_train_epochs"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["training"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        logging_steps=int(config["training"]["logging_steps"]),
        save_steps=int(config["training"]["save_steps"]),
        save_total_limit=int(config["training"].get("save_total_limit", 3)),
        eval_steps=int(config["training"]["eval_steps"]),
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=bool(config["training"].get("bf16", True)),
        fp16=bool(config["training"].get("fp16", False)),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        disable_tqdm=False,
        seed=int(config["training"].get("seed", 13)),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=processor,
        data_collator=make_collator(processor),
        callbacks=[TrainingArtifactCallback()],
    )
    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    processor.save_pretrained(config["training"]["output_dir"])


def sample_to_record(sample) -> dict:
    prompt = build_prompt(sample)
    return {
        "image_path": sample.image_path,
        "prompt": prompt,
        "answer": sample.answer,
        "task_type": sample.task_type,
    }


def make_collator(processor):
    def collate(batch: list[dict]) -> dict:
        messages = []
        for row in batch:
            image = Image.open(row["image_path"]).convert("RGB")
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": row["prompt"]},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": row["answer"]}],
                    },
                ]
            )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return collate


class TrainingArtifactCallback(TrainerCallback):
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.log_path = Path("reports/training_log.csv")
        self.figure_path = Path("reports/figures/training_loss.png")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step}
        row.update({key: value for key, value in logs.items() if isinstance(value, (int, float))})
        self.rows.append(row)
        self.write_csv()
        self.write_chart()

    def write_csv(self) -> None:
        if not self.rows:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in self.rows for key in row})
        with self.log_path.open("w", encoding="utf-8") as handle:
            handle.write(",".join(keys) + "\n")
            for row in self.rows:
                handle.write(",".join(str(row.get(key, "")) for key in keys) + "\n")

    def write_chart(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        train = [(row["step"], row["loss"]) for row in self.rows if "loss" in row]
        evals = [(row["step"], row["eval_loss"]) for row in self.rows if "eval_loss" in row]
        if not train and not evals:
            return

        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        if train:
            plt.plot([x for x, _ in train], [y for _, y in train], label="train loss")
        if evals:
            plt.plot([x for x, _ in evals], [y for _, y in evals], label="eval loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Gemma Surgical Frame LoRA Training")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(self.figure_path, dpi=160, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
