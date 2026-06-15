import base64
import html
import json
from pathlib import Path

import modal


APP_NAME = "surgical-gemma-lora-demo-v2"
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
        "torchvision>=0.18.0",
        "transformers>=5.10.1",
    )
    .run_commands(
        "python - <<'PY'\n"
        "import torch\n"
        "import torchvision\n"
        "print('torch', torch.__version__)\n"
        "print('torchvision', torchvision.__version__)\n"
        "PY"
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
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
        from src.models.prompts import build_prompt

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
            generation_prompt = build_prompt(sample, include_system=True)

        if mode == "Constrained labels" and task_type in {"phase", "tool_type", "action"}:
            choices = PHASE_CHOICES if task_type == "phase" else default_choices(task_type)
            scored = [(score_answer(self.model, self.processor, image, prompt, answer), answer) for answer in choices]
            scored.sort(key=lambda item: item[0] if math.isfinite(item[0]) else float("inf"))
            rows = [{"rank": str(i), "answer": answer, "loss": f"{score:.4f}"} for i, (score, answer) in enumerate(scored[:7], 1)]
            return scored[0][1], rows

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": generation_prompt},
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
        if is_invalid_generation(answer):
            answer = (
                "Free-form generation returned an invalid short response. "
                "For phase recognition, switch to Constrained labels to get one of the 7 benchmark labels."
            )
        return answer, []


def is_invalid_generation(answer: str) -> bool:
    clean = answer.strip()
    if len(clean) < 4:
        return True
    alnum_count = sum(char.isalnum() for char in clean)
    return alnum_count < 3


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


def render_simple_page(
    answer: str = "",
    rows: list[dict[str, str]] | None = None,
    image_data_url: str = "",
    error: str = "",
) -> str:
    rows = rows or []
    score_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row.get('rank', ''))}</td>"
        f"<td>{html.escape(row.get('answer', ''))}</td>"
        f"<td>{html.escape(row.get('loss', ''))}</td>"
        "</tr>"
        for row in rows
    )
    image_html = (
        f'<img src="{image_data_url}" alt="Uploaded surgical frame" />'
        if image_data_url
        else '<div class="placeholder">Upload a surgical frame to preview it here.</div>'
    )
    answer_html = html.escape(answer or "No prediction yet.")
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Surgical Gemma LoRA Demo</title>
        <style>
          body {{
            margin: 0;
            background: #f7f8fa;
            color: #172033;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          main {{
            max-width: 1180px;
            margin: 0 auto;
            padding: 36px 20px 56px;
          }}
          h1 {{ margin: 0 0 8px; font-size: 34px; }}
          p {{ color: #4b5563; }}
          .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            align-items: start;
          }}
          .panel {{
            background: white;
            border: 1px solid #d9dee7;
            border-radius: 8px;
            padding: 18px;
          }}
          label {{
            display: block;
            font-weight: 700;
            margin: 14px 0 6px;
          }}
          input, select, textarea, button {{
            width: 100%;
            box-sizing: border-box;
            border: 1px solid #cfd6e3;
            border-radius: 7px;
            padding: 11px 12px;
            font-size: 16px;
          }}
          textarea {{ min-height: 92px; resize: vertical; }}
          button {{
            margin-top: 18px;
            border: 0;
            background: #ef7a2e;
            color: white;
            font-weight: 800;
            cursor: pointer;
          }}
          img {{
            display: block;
            width: 100%;
            max-height: 460px;
            object-fit: contain;
            background: #050505;
            border-radius: 6px;
          }}
          .placeholder {{
            min-height: 320px;
            display: grid;
            place-items: center;
            color: #6b7280;
            background: #eef1f5;
            border-radius: 6px;
            text-align: center;
            padding: 20px;
          }}
          .answer {{
            min-height: 80px;
            white-space: pre-wrap;
            background: #f9fafb;
            border: 1px solid #d9dee7;
            border-radius: 7px;
            padding: 14px;
            font-size: 18px;
            font-weight: 700;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
          }}
          th, td {{
            border-bottom: 1px solid #e5e7eb;
            padding: 10px 8px;
            text-align: left;
          }}
          .error {{
            border-left: 5px solid #c2410c;
            background: #fff7ed;
            padding: 12px 14px;
            border-radius: 7px;
            margin: 16px 0;
            color: #7c2d12;
            white-space: pre-wrap;
          }}
          @media (max-width: 860px) {{
            .grid {{ grid-template-columns: 1fr; }}
          }}
        </style>
      </head>
      <body>
        <main>
          <h1>Surgical Frame QA Demo</h1>
          <p>Gemma 4 12B + LoRA on Modal GPU. Use constrained labels for phase benchmark demos.</p>
          {error_html}
          <div class="grid">
            <section class="panel">
              <form action="/predict" method="post" enctype="multipart/form-data">
                <label>Surgical frame</label>
                <input name="file" type="file" accept="image/png,image/jpeg,image/webp" required />
                <label>Task</label>
                <select name="task_type">
                  <option value="phase" selected>phase</option>
                  <option value="tool_type">tool_type</option>
                  <option value="action">action</option>
                  <option value="triplet">triplet</option>
                  <option value="vqa">vqa</option>
                </select>
                <label>Question</label>
                <textarea name="question">What surgical phase is shown?</textarea>
                <label>Prediction mode</label>
                <select name="mode">
                  <option value="Constrained labels" selected>Constrained labels</option>
                  <option value="Free-form generation">Free-form generation</option>
                </select>
                <button type="submit">Predict</button>
              </form>
            </section>
            <section class="panel">
              {image_html}
              <h2>Model answer</h2>
              <div class="answer">{answer_html}</div>
              <h2>Candidate scores</h2>
              <table>
                <thead><tr><th>rank</th><th>answer</th><th>loss</th></tr></thead>
                <tbody>{score_rows}</tbody>
              </table>
            </section>
          </div>
          <p><strong>Demo note:</strong> For phase recognition, constrained labels are the benchmark-style mode.
          Free-form generation is only for assistant-style qualitative examples.</p>
        </main>
      </body>
    </html>
    """


@app.function(image=image, timeout=900, scaledown_window=120, max_containers=1)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def ui():
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    return mount_gradio_app(app=FastAPI(), blocks=build_ui(), path="/")


@app.function(image=image, timeout=1200, scaledown_window=600, max_containers=1)
@modal.asgi_app()
def simple_ui():
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse

    web = FastAPI()

    @web.get("/", response_class=HTMLResponse)
    async def index():
        return render_simple_page()

    @web.post("/predict", response_class=HTMLResponse)
    async def predict(
        file: UploadFile = File(...),
        task_type: str = Form("phase"),
        question: str = Form("What surgical phase is shown?"),
        mode: str = Form("Constrained labels"),
    ):
        image_bytes = await file.read()
        image_type = file.content_type or "image/png"
        image_data_url = f"data:{image_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        try:
            answer, rows = await SurgicalGemma().predict.remote.aio(image_bytes, task_type, question, mode)
        except Exception as exc:
            return render_simple_page(
                image_data_url=image_data_url,
                error=f"{type(exc).__name__}: {exc}",
            )
        return render_simple_page(answer=answer, rows=rows, image_data_url=image_data_url)

    return web
