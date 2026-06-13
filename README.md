# Surgical Video Assistant

Frame-level surgical VQA, phase recognition, and tool/action benchmarking without full video training.

The project is designed around Gemma 4-style multimodal zero-shot and few-shot prompting. It can run end-to-end with a deterministic mock provider first, then switch to a local Hugging Face multimodal model when the environment has the required weights and GPU.

## Project Layout

- `data/raw/`: manually downloaded datasets.
- `data/processed/`: normalized JSONL files.
- `src/data/`: dataset parsers and validation.
- `src/models/`: prompt construction and inference runners.
- `src/eval/`: task metrics.
- `src/demo/`: Streamlit surgical frame assistant.
- `configs/`: inference configs.
- `reports/`: generated metrics and report notes.
- `tests/`: parser and metric tests.

## Quick Start

Install the base dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare SurgMLLMBench raw JSON/JSONL files after downloading them into `data/raw/SurgMLLMBench`:

```bash
python -m src.data.prepare --dataset surgmllmbench --raw data/raw/SurgMLLMBench --out data/processed
```

Run a smoke-test inference with the mock provider:

```bash
python -m src.models.run_inference --config configs/mock_zero_shot.yaml
```

Evaluate predictions:

```bash
python -m src.eval.evaluate --pred reports/predictions_mock.jsonl --out reports/metrics_mock.json
```

Start the demo:

```bash
streamlit run src/demo/app.py
```

## Optional VM Training

If you rent a GPU VM, install the training dependencies:

```bash
python -m pip install -r requirements-train.txt
```

Then run LoRA/QLoRA supervised fine-tuning on normalized frame-level QA records:

```bash
python -m src.models.train_sft --config configs/gemma4_lora_sft.yaml
```

This trains on image-frame QA pairs only. It does not train a full temporal video model.

## Notebook for Presentation

Open [notebooks/Surgical_Video_Assistant_Training.ipynb](notebooks/Surgical_Video_Assistant_Training.ipynb) on Colab or a GPU VM. It includes:

- a single clone cell,
- dependency install,
- GPU check,
- dataset preparation,
- smoke-test inference/evaluation,
- LoRA/QLoRA training,
- checkpoint logging,
- loss and metric charts saved under `reports/figures`.

After pushing this repo to GitHub, replace the notebook variable:

```python
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/surgical-video-assistant.git"
```

with your real repository URL.

## GitHub Push

If GitHub CLI is installed and authenticated:

```bash
gh repo create surgical-video-assistant --public --source . --remote origin --push
```

Without GitHub CLI, create an empty GitHub repository named `surgical-video-assistant`, then run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/surgical-video-assistant.git
git branch -M main
git push -u origin main
```

## Dataset Notes

SurgMLLMBench is the primary dataset because it already includes surgical images/frames and VQA-style annotations. The Hugging Face viewer can fail on mixed JSON schemas, so this project reads raw JSON/JSONL files directly.

CholecT50/CholecT45 can be used as a secondary benchmark after CAMMA access is granted. The included parser supports simple frame-level CSV/JSONL exports and turns labels into QA-style records.

This project is for research and education only. It is not clinical decision support.
