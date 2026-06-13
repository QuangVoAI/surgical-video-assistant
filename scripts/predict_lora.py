from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.data.schema import SurgicalSample, infer_task_type
from src.models.prompts import PHASE_CHOICES, build_prompt, system_prompt
from src.models.providers import load_multimodal_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a surgical frame QA prediction with a trained LoRA checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="LoRA adapter checkpoint or final adapter folder.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--task-type", default="", help="Optional task type: phase, tool_type, action, triplet, vqa.")
    parser.add_argument("--model", default="google/gemma-4-12B")
    parser.add_argument("--processor", default="google/gemma-4-12B-it")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--mode", choices=["generate", "classify"], default="classify")
    parser.add_argument("--choices", nargs="*", default=None, help="Candidate labels for classify mode.")
    args = parser.parse_args()

    answer = predict(
        checkpoint=args.checkpoint,
        image_path=args.image,
        question=args.question,
        task_type=args.task_type,
        model_name=args.model,
        processor_name=args.processor,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        choices=args.choices,
    )
    print(answer)


def predict(
    checkpoint: Path,
    image_path: Path,
    question: str,
    task_type: str,
    model_name: str,
    processor_name: str,
    max_new_tokens: int,
    mode: str,
    choices: list[str] | None,
) -> str:
    try:
        import torch
        from peft import PeftModel
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
    base_model = load_multimodal_model(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    task_type = task_type or infer_task_type(question)
    sample = SurgicalSample(
        sample_id="demo",
        dataset="demo",
        image_path=str(image_path),
        question=question,
        answer="",
        task_type=task_type,
        split="demo",
        metadata={},
    )
    prompt = build_prompt(sample, include_system=False)
    if mode == "classify":
        candidate_answers = choices or default_choices(task_type)
        if not candidate_answers:
            raise ValueError("Classify mode needs --choices for this task type.")
        scored = [
            (score_answer(model, processor, image, prompt, answer), answer)
            for answer in candidate_answers
        ]
        scored.sort(key=lambda item: item[0])
        best_score, best_answer = scored[0]
        print("scores:")
        for score, answer in scored[:10]:
            print(f"  {score:.4f}\t{answer}")
        print("prediction:")
        return best_answer

    messages = [
        {"role": "system", "content": system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]
    rendered_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=[rendered_prompt],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    generated = output_ids[0][prompt_length:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def default_choices(task_type: str) -> list[str]:
    if task_type == "phase":
        return PHASE_CHOICES
    if task_type == "tool_type":
        return ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen bag"]
    if task_type == "action":
        return [
            "grasp",
            "retract",
            "dissect",
            "coagulate",
            "clip",
            "cut",
            "aspirate",
            "irrigate",
            "pack",
            "null_verb",
        ]
    return []


def score_answer(model, processor, image: Image.Image, prompt: str, answer: str) -> float:
    import torch

    prompt_messages = [
        {"role": "system", "content": system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]
    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        }
    ]
    rendered_prompt = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    rendered_full = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    inputs = processor(
        text=[rendered_full],
        images=[image],
        return_tensors="pt",
    ).to(model.device)
    prompt_inputs = processor(
        text=[rendered_prompt],
        images=[image],
        return_tensors="pt",
    ).to(model.device)
    labels = inputs["input_ids"].clone()
    prompt_length = int(prompt_inputs["attention_mask"].sum(dim=1).item())
    labels[:, :prompt_length] = -100
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    for token_name in ("boi_token_id", "image_token_id", "eoi_token_id"):
        token_id = getattr(processor.tokenizer, token_name, None)
        if token_id is not None:
            labels[labels == token_id] = -100

    with torch.inference_mode():
        output = model(**inputs, labels=labels)
    return float(output.loss.detach().cpu())


if __name__ == "__main__":
    main()
