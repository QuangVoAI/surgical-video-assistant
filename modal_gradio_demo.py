from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "surgical-gemma-lora-demo"
APP_DIR = Path("/root/app")
HF_CACHE_DIR = "/root/.cache/huggingface"

BASE_MODEL = "google/gemma-4-12B"
PROCESSOR_MODEL = "google/gemma-4-12B-it"
LORA_ADAPTER = "SpringWang08/surgical-gemma4-12b-lora"

PHASE_CHOICES = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping and Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning and Coagulation",
    "Gallbladder Extraction",
]

TASK_DEFAULTS = {
    "phase": "What surgical phase is shown?",
    "tool_type": "Which surgical instruments are visible?",
    "action": "What action is the instrument performing?",
    "triplet": "What surgical action triplets are visible?",
    "vqa": "What is visible in this surgical frame?",
}

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("surgical-gemma-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "fastapi[standard]>=0.115.0",
        "gradio>=4.44.0",
        "hf-transfer>=0.1.8",
        "huggingface_hub>=0.24.0",
        "peft>=0.11.0",
        "pillow>=10.0.0",
        "protobuf>=4.25.0",
        "sentencepiece>=0.2.0",
        "torch>=2.3.0",
        "transformers>=5.10.1",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": str(APP_DIR),
        }
    )
    .add_local_dir("src", remote_path=str(APP_DIR / "src"))
    .add_local_dir("scripts", remote_path=str(APP_DIR / "scripts"))
    .add_local_dir("demo_samples", remote_path=str(APP_DIR / "demo_samples"))
    .add_local_file("reports/demo_cases.json", remote_path=str(APP_DIR / "reports/demo_cases.json"))
)


@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=900,
    scaledown_window=300,
    max_containers=1,
)
class SurgicalGemma:
    @modal.enter()
    def load(self) -> None:
        import os

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig

        from src.models.providers import load_multimodal_model

        token = os.environ.get("HF_TOKEN")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(
            PROCESSOR_MODEL,
            token=token,
            trust_remote_code=True,
        )
        base_model = load_multimodal_model(
            BASE_MODEL,
            device_map="auto",
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            token=token,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            LORA_ADAPTER,
            token=token,
        )
        self.model.eval()

    @modal.method()
    def predict(self, image_bytes: bytes, task_type: str, question: str, mode: str) -> tuple[str, list[dict[str, str]]]:
        import io
        import math
        import tempfile
        import uuid
        from pathlib import Path

        from PIL import Image

        from scripts.predict_lora import default_choices, score_answer
        from src.data.schema import SurgicalSample
        from src.models.prompts import build_prompt, system_prompt

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        task_type = task_type or "phase"
        question = question.strip() or TASK_DEFAULTS.get(task_type, TASK_DEFAULTS["phase"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / f"{uuid.uuid4()}.png"
            image.save(image_path)
            sample = SurgicalSample(
                sample_id="modal-demo",
                dataset="demo",
                image_path=str(image_path),
                question=question,
                answer="",
                task_type=task_type,
                split="demo",
                metadata={"answer_space": PHASE_CHOICES if task_type == "phase" else []},
            )
            prompt = build_prompt(sample, include_system=False)

        if mode == "Constrained labels" and task_type in {"phase", "tool_type", "action"}:
            choices = PHASE_CHOICES if task_type == "phase" else default_choices(task_type)
            scored = [(score_answer(self.model, self.processor, image, prompt, answer), answer) for answer in choices]
            scored.sort(key=lambda item: item[0] if math.isfinite(item[0]) else float("inf"))
            rows = [{"rank": str(i), "answer": answer, "loss": f"{score:.4f}"} for i, (score, answer) in enumerate(scored[:7], 1)]
            return scored[0][1], rows

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
        rendered_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=[rendered_prompt],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        import torch

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        generated = output_ids[0][prompt_length:]
        answer = self.processor.decode(generated, skip_special_tokens=True).strip()
        return answer, []


def read_sample_examples() -> list[list[str]]:
    manifest_path = APP_DIR / "demo_samples" / "manifest.json"
    if not manifest_path.exists():
        return []
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = []
    for row in rows:
        image_path = APP_DIR / "demo_samples" / row["image"]
        if image_path.exists():
            examples.append(
                [
                    str(image_path),
                    row.get("task_type", "phase"),
                    row.get("question", TASK_DEFAULTS["phase"]),
                    "Constrained labels",
                ]
            )
    return examples


def build_ui():
    import gradio as gr

    def set_default_question(task_type: str) -> str:
        return TASK_DEFAULTS.get(task_type, TASK_DEFAULTS["phase"])

    def run_predict(image_path: str | None, task_type: str, question: str, mode: str):
        if not image_path:
            return "Upload or select a surgical frame first.", []
        with open(image_path, "rb") as handle:
            image_bytes = handle.read()
        return SurgicalGemma().predict.remote(image_bytes, task_type, question, mode)

    with gr.Blocks(title="Surgical Gemma LoRA Demo") as demo:
        gr.Markdown(
            """
            # Surgical Frame QA Demo
            Gemma 4 12B + LoRA for research-only surgical frame understanding.
            Upload an authorized frame, choose a question type, then run prediction on Modal GPU.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Surgical frame", type="filepath", sources=["upload"], height=360)
                task_input = gr.Dropdown(
                    choices=list(TASK_DEFAULTS),
                    value="phase",
                    label="Task",
                )
                question_input = gr.Textbox(
                    value=TASK_DEFAULTS["phase"],
                    label="Question",
                    lines=2,
                )
                mode_input = gr.Radio(
                    choices=["Constrained labels", "Free-form generation"],
                    value="Constrained labels",
                    label="Prediction mode",
                )
                run_button = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                answer_output = gr.Textbox(label="Model answer", lines=3)
                scores_output = gr.Dataframe(
                    headers=["rank", "answer", "loss"],
                    label="Candidate scores",
                    interactive=False,
                )
                gr.Markdown(
                    """
                    **Demo note:** If you use bundled examples, describe them as held-out evaluation examples.
                    For unseen images, upload frames you are authorized to use.
                    """
                )

        task_input.change(set_default_question, inputs=task_input, outputs=question_input)
        run_button.click(
            run_predict,
            inputs=[image_input, task_input, question_input, mode_input],
            outputs=[answer_output, scores_output],
        )

        examples = read_sample_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[image_input, task_input, question_input, mode_input],
                label="Held-out demo examples",
            )

    return demo


@app.function(image=image, timeout=900, scaledown_window=120, max_containers=1)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def ui():
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    return mount_gradio_app(app=FastAPI(), blocks=build_ui(), path="/")
