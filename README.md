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

This installs the libraries needed for dataset preparation, Hugging Face inference, reporting charts, the Streamlit demo, and local tests.

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

This trains on image-frame QA pairs only. It does not train a full temporal video model. The training config now expects split-aware JSONL records and filters `train` vs `val/test` before fitting.

On NVIDIA L4/A100/H100 or newer GPUs, FlashAttention can speed up training, but install it separately only after PyTorch is installed:

```bash
python -m pip install flash-attn --no-build-isolation
```

Before starting a long VM run, check the dataset splits and training environment:

```bash
python scripts/preflight_train.py --config configs/gemma4_12b_lora_sft.yaml
```

## Notebook for Presentation

Open [notebooks/Surgical_Video_Assistant_Training.ipynb](notebooks/Surgical_Video_Assistant_Training.ipynb) on Colab or a GPU VM. It includes:

- a single clone cell,
- dependency install,
- GPU check,
- dataset preparation,
- smoke-test inference/evaluation,
- LoRA/QLoRA training,
- Gemma 4 12B and Gemma 4 26B-A4B comparison-ready configs,
- checkpoint logging,
- loss and metric charts saved under `reports/figures`.

After pushing this repo to GitHub, replace the notebook variable:

```python
GITHUB_REPO_URL = "https://github.com/QuangVoAI/surgical-video-assistant.git"
```

with your real repository URL.

## GitHub Push

If GitHub CLI is installed and authenticated:

```bash
gh repo create surgical-video-assistant --public --source . --remote origin --push
```

Without GitHub CLI, create an empty GitHub repository named `surgical-video-assistant`, then run:

```bash
git remote add origin https://github.com/QuangVoAI/surgical-video-assistant.git
git branch -M main
git push -u origin main
```

You can also create and push the repository with the included safe helper. Create a new token, keep it local, then run:

```bash
export GITHUB_TOKEN="paste-your-new-token-here"
python scripts/publish_to_github.py --repo surgical-video-assistant
unset GITHUB_TOKEN
```

If the leaked token was ever pasted into a chat or notebook, revoke it and create a new one before running the command above.

## Presentation Report

After inference/evaluation or training, generate a presentation-ready summary:

```bash
python scripts/make_training_report.py \
  --metrics reports/metrics_mock.json \
  --out reports/training_summary.md
```

To compare your Gemma runs with prior papers in one Markdown table:

```bash
python scripts/compare_benchmarks.py \
  --paper-baselines reports/paper_baselines.template.json \
  --gemma12 reports/metrics_gemma4_12b_zero_shot.json \
  --gemma26 reports/metrics_gemma4_26b_zero_shot.json \
  --out reports/benchmark_comparison.md
```

## Dataset Notes

SurgMLLMBench is the primary dataset because it already includes surgical images/frames and VQA-style annotations. The Hugging Face viewer can fail on mixed JSON schemas, so this project reads raw JSON/JSONL files directly.

CholecT50/CholecT45 can be used as a secondary benchmark after CAMMA access is granted. The included parser supports simple frame-level CSV/JSONL exports and turns labels into QA-style records.

This project is for research and education only. It is not clinical decision support.
