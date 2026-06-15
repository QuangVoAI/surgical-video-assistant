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
        processor_name: str | None = None,
        adapter_name: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 64,
        trust_remote_code: bool = False,
    ) -> None:
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Transformers provider requires optional dependencies: "
                "pip install '.[hf]' or install torch/transformers/accelerate."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            processor_name or model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = load_multimodal_model(
            model_name,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        if adapter_name:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA adapter loading requires peft. Install requirements-train.txt "
                    "or run: python -m pip install peft"
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_name)
            self.name = f"{model_name} + {adapter_name}"

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
        rendered_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=rendered_prompt,
            images=[image],
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
            processor_name=config.get("processor_name"),
            adapter_name=config.get("adapter_name"),
            device=config.get("device", "auto"),
            max_new_tokens=int(config.get("max_new_tokens", 64)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    raise ValueError(f"Unknown provider: {provider}")


def load_multimodal_model(model_name: str, **kwargs):
    try:
        from transformers import AutoModelForImageTextToText

        return load_with_dtype_compat(AutoModelForImageTextToText, model_name, **kwargs)
    except (ImportError, AttributeError):
        from transformers import AutoModelForMultimodalLM

        return load_with_dtype_compat(AutoModelForMultimodalLM, model_name, **kwargs)


def load_with_dtype_compat(model_cls, model_name: str, **kwargs):
    try:
        return model_cls.from_pretrained(model_name, **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
        try:
            return model_cls.from_pretrained(model_name, **retry_kwargs)
        except TypeError:
            raise exc
