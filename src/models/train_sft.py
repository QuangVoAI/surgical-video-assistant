from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image

from src.data.io import read_samples
from src.models.prompts import build_prompt, system_prompt
from src.models.providers import load_multimodal_model

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
        from transformers import AutoProcessor, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training requires GPU dependencies. Install with: "
            "python -m pip install -r requirements-train.txt"
        ) from exc

    model_name = config["model"]["name"]
    processor_name = config["model"].get("processor_name", model_name)
    processor = AutoProcessor.from_pretrained(processor_name)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    quantization_config = None
    if config["model"].get("load_in_4bit", True):
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
        dtype=torch.bfloat16 if config["training"].get("bf16", True) else torch.float16,
    )
    if config["model"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=config["lora"].get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
        bias="none",
        modules_to_save=config["lora"].get("modules_to_save", ["lm_head", "embed_tokens"]),
        ensure_weight_tying=True,
    )

    train_samples = read_samples(Path(config["data"]["train_jsonl"]))
    eval_samples = read_samples(Path(config["data"]["eval_jsonl"]))
    train_splits = set(config["data"].get("train_splits", ["train"]))
    eval_splits = set(config["data"].get("eval_splits", ["val", "test"]))
    train_samples = [sample for sample in train_samples if sample.split in train_splits]
    eval_samples = [sample for sample in eval_samples if sample.split in eval_splits]
    max_train = config["data"].get("max_train_samples")
    max_eval = config["data"].get("max_eval_samples")
    if max_train:
        train_samples = train_samples[: int(max_train)]
    if max_eval:
        eval_samples = eval_samples[: int(max_eval)]

    if not train_samples:
        raise ValueError(f"No training samples found for splits: {sorted(train_splits)}")
    if not eval_samples:
        raise ValueError(f"No evaluation samples found for splits: {sorted(eval_splits)}")

    train_dataset = Dataset.from_list([sample_to_record(sample) for sample in train_samples])
    eval_dataset = Dataset.from_list([sample_to_record(sample) for sample in eval_samples])

    args = SFTConfig(
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
        eval_strategy="steps",
        save_strategy="steps",
        bf16=bool(config["training"].get("bf16", True)),
        fp16=bool(config["training"].get("fp16", False)),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        disable_tqdm=False,
        seed=int(config["training"].get("seed", 13)),
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
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
    prompt = build_prompt(sample, include_system=False)
    return {
        "image_path": sample.image_path,
        "prompt": prompt,
        "answer": sample.answer,
    }


def make_collator(processor):
    def collate(batch: list[dict]) -> dict:
        texts = []
        prompt_texts = []
        images = []
        for row in batch:
            messages = build_messages(row["image_path"], row["prompt"], row["answer"])
            image_inputs = process_vision_info(messages)
            rendered_prompt = processor.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                tokenize=False,
            )
            rendered_full = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            texts.append(rendered_full.strip())
            prompt_texts.append(rendered_prompt.strip())
            images.append(image_inputs)

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        prompt_inputs = processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, : int(prompt_length)] = -100

        labels[labels == processor.tokenizer.pad_token_id] = -100
        for token_name in ("boi_token_id", "image_token_id", "eoi_token_id"):
            token_id = getattr(processor.tokenizer, token_name, None)
            if token_id is not None:
                labels[labels == token_id] = -100
        inputs["labels"] = labels
        return inputs

    return collate


def build_messages(image_path: str, prompt: str, answer: str) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    return [
        {
            "role": "system",
            "content": system_prompt(),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                image = element.get("image")
                if image is not None:
                    images.append(image.convert("RGB"))
    return images


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
