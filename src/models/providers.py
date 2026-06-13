from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from src.data.schema import SurgicalSample


class VisionLanguageProvider(Protocol):
    name: str

    def generate(self, sample: SurgicalSample, prompt: str) -> str:
        ...


class MockProvider:
    name = "gemma-4-mock"

    def generate(self, sample: SurgicalSample, prompt: str) -> str:
        text = sample.question.lower()
        if sample.task_type == "tool_count":
            return "1"
        if sample.task_type == "tool_type":
            return "grasper"
        if sample.task_type == "phase":
            return "Calot Triangle Dissection"
        if sample.task_type in {"action", "triplet"}:
            return "grasping"
        if "phase" in text:
            return "Calot Triangle Dissection"
        if "instrument" in text or "tool" in text:
            return "grasper"
        return "Unable to determine from the frame."


class TransformersProvider:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 64,
    ) -> None:
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError(
                "Transformers provider requires optional dependencies: "
                "pip install '.[hf]' or install torch/transformers/accelerate."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, device_map=device)

    def generate(self, sample: SurgicalSample, prompt: str) -> str:
        image = Image.open(Path(sample.image_path)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        prompt_length = inputs["input_ids"].shape[-1]
        generated = output_ids[0][prompt_length:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


def build_provider(config: dict) -> VisionLanguageProvider:
    provider = (config.get("provider") or "mock").lower()
    if provider == "mock":
        return MockProvider()
    if provider == "transformers":
        return TransformersProvider(
            model_name=config["name"],
            device=config.get("device", "auto"),
            max_new_tokens=int(config.get("max_new_tokens", 64)),
        )
    raise ValueError(f"Unknown provider: {provider}")
