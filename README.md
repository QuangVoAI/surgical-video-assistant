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
python scripts/download_hf_dataset.py \
  --repo-id introvoyz041/SurgMLLMBench \
  --local-dir data/raw/SurgMLLMBench
```

Then normalize the downloaded raw files:

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

Start the presentation results demo:

```bash
streamlit run src/demo/results_app.py
```

This offline demo uses saved zero-shot, LoRA, and calibration metrics under `reports/`.
It does not load Gemma or require a GPU, so it is the recommended demo for a live report.
The same page also has an `Upload & Predict` tab. Use `configs/mock_zero_shot.yaml`
for a local UI demo, or use `configs/gemma4_12b_lora_hf_demo.yaml` on a GPU machine
to load the published LoRA adapter from Hugging Face.

## Modal GPU Demo

For a live GPU demo with your published LoRA adapter, use the Modal/Gradio app:

```bash
python -m pip install -r requirements-modal.txt
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
modal serve modal_gradio_demo.py
```

When it works, deploy a persistent URL:

```bash
modal deploy modal_gradio_demo.py
```

The app uses `google/gemma-4-12B` with the LoRA adapter
`SpringWang08/surgical-gemma4-12b-lora`, loaded in 4-bit on an A100-40GB GPU.
It scales down after inactivity, so stop the serve command or delete the deployment when
the demo is finished to avoid spending credits.

By default, the Modal app supports image upload. If you have authorized CholecT50 frames
locally and want a few preset examples, copy only a small held-out demo subset:

```bash
python scripts/prepare_demo_frames.py --limit 4 --image-root /path/to/surgical-video-assistant
modal serve modal_gradio_demo.py
```

Do not commit raw surgical images. The `demo_samples/` folder is ignored except for
`.gitkeep`. In the report, describe these preset images as held-out evaluation examples,
not as unseen external clinical data.

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

For a fast presentation run, first create a balanced lightweight subset that still points to the original images:

```bash
python scripts/make_fast_subset.py \
  --input data/processed/cholec50_qa.jsonl \
  --out data/processed/processed_dataset_fast.jsonl \
  --train-per-task 300 \
  --eval-per-task 80 \
  --require-images
```

Then choose one of the quick configs:

```bash
# Smoke test, usually the fastest way to verify the full training loop.
python scripts/preflight_train.py --config configs/gemma4_12b_lora_tiny.yaml
python -m src.models.train_sft --config configs/gemma4_12b_lora_tiny.yaml

# Short report-ready run with checkpoints and charts.
python scripts/preflight_train.py --config configs/gemma4_12b_lora_quick.yaml
python -m src.models.train_sft --config configs/gemma4_12b_lora_quick.yaml
```

The quick configs are intended for limited VM credit. They train a LoRA adapter on a subset and save logs, checkpoints, and `reports/figures/training_loss.png`.

For a stronger but still fast experiment, use a video-level subset so train and eval come from different procedures:

```bash
python scripts/make_fast_subset.py \
  --input data/processed/cholec50_qa.jsonl \
  --out data/processed/processed_dataset_video_subset.jsonl \
  --train-videos VID05 VID08 VID10 VID12 VID15 VID18 \
  --eval-videos VID01 VID04 VID13 \
  --train-per-task 400 \
  --eval-per-task 100 \
  --require-images

python scripts/preflight_train.py --config configs/gemma4_12b_lora_video_subset.yaml
python -m src.models.train_sft --config configs/gemma4_12b_lora_video_subset.yaml
```

This is the recommended presentation setup when full CholecT50 training is too expensive. It does not claim full leaderboard performance, but it avoids frame overlap between train and eval videos.

If phase top-1 accuracy is weak, run a focused phase-only balanced experiment instead of training all tasks together:

```bash
python scripts/make_fast_subset.py \
  --input data/processed/cholec50_qa.jsonl \
  --out data/processed/processed_dataset_phase_balanced.jsonl \
  --train-videos VID05 VID08 VID10 VID12 VID15 VID18 \
  --eval-videos VID01 VID04 VID13 \
  --task-types phase \
  --balance-by-answer-tasks phase \
  --train-per-task 700 \
  --eval-per-task 500 \
  --require-images

python scripts/preflight_train.py --config configs/gemma4_12b_lora_phase_balanced.yaml
python -m src.models.train_sft --config configs/gemma4_12b_lora_phase_balanced.yaml
```

This variant fixes three common failure modes at once:

- train on `phase` only instead of mixing phase with tool/action/triplet,
- balance the train subset by phase label,
- use canonical CholecT50 phase names consistently in both prompts and answers.

If you want to push top-1 phase accuracy harder and still stay in a practical VM budget, run the stronger variant:

```bash
python scripts/make_fast_subset.py \
  --input data/processed/cholec50_qa.jsonl \
  --out data/processed/processed_dataset_phase_balanced_strong.jsonl \
  --train-videos VID05 VID08 VID10 VID12 VID15 VID18 \
  --eval-videos VID01 VID04 VID13 \
  --task-types phase \
  --balance-by-answer-tasks phase \
  --train-per-task 980 \
  --eval-per-task 500 \
  --require-images

python scripts/preflight_train.py --config configs/gemma4_12b_lora_phase_balanced_strong.yaml
python -m src.models.train_sft --config configs/gemma4_12b_lora_phase_balanced_strong.yaml
```

Why this stronger config should help:

- more phase-only training samples,
- lower learning rate to reduce label-collapse bias,
- longer training to let the ranking stabilize,
- slightly larger LoRA capacity for phase discrimination.

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

## Benchmark After Training

After a LoRA checkpoint is available, run constrained prediction on the eval split and then compute benchmark-style metrics. Start with `phase` because it is fast, has a small fixed label space, and reports accuracy, macro-F1, and a confusion matrix.

```bash
python scripts/predict_eval_lora.py \
  --checkpoint checkpoints/gemma4-12b-surgical-frame-lora-video-subset/checkpoint-500 \
  --data data/processed/processed_dataset_video_subset.jsonl \
  --out reports/final/predictions_gemma4_12b_lora_phase.jsonl \
  --task-types phase \
  --limit-per-task 100 \
  --skip-missing-images

python -m src.eval.evaluate \
  --pred reports/final/predictions_gemma4_12b_lora_phase.jsonl \
  --out reports/final/metrics_gemma4_12b_lora_phase.json
```

For a larger benchmark, remove `--limit-per-task 100`. Tool/action/triplet can also be evaluated, but they are slower because constrained scoring needs to compare more candidate answers:

```bash
python scripts/predict_eval_lora.py \
  --checkpoint checkpoints/gemma4-12b-surgical-frame-lora-video-subset/checkpoint-500 \
  --data data/processed/processed_dataset_video_subset.jsonl \
  --out reports/final/predictions_gemma4_12b_lora_all.jsonl \
  --task-types phase tool_type action triplet \
  --limit-per-task 50 \
  --skip-missing-images
```

## Dataset Notes

SurgMLLMBench is the primary dataset because it already includes surgical images/frames and VQA-style annotations. The Hugging Face viewer can fail on mixed JSON schemas, so this project reads raw JSON/JSONL files directly.

SSG-VQA is the preferred practical VQA dataset path when SurgMLLMBench source images are unavailable. Download the public QA pairs with:

```bash
python scripts/download_ssg_vqa.py --out data/raw/SSG-VQA
python -m src.data.prepare --dataset ssg-vqa --raw data/raw/SSG-VQA --out data/processed
```

The public SSG-VQA repo provides QA pairs and pre-extracted visual features. Raw frames still come from CholecT45/Cholec80 access, so preflight will continue to fail for Gemma image training until the corresponding images are placed under `data/raw/SSG-VQA/images/VIDxx/000001.jpg` or another path resolvable by the parser.

CholecT50/CholecT45 is now the recommended image-backed training path once CAMMA access is granted. SurgMLLMBench and SSG-VQA are useful for VQA/task design, but their public downloads may not include raw frames. CholecT50 release 2.0 provides the frame images and frame-wise labels needed for Gemma image training.

After downloading the dataset from the access email, copy the archive to the VM and unpack it so the folder looks like this:

```text
data/raw/CholecT50/
  videos/
    VID01/
      000001.png
  labels/
    VID01.json
  label_mapping.txt
  README.md
```

Then prepare the frame-level QA dataset:

```bash
python -m src.data.prepare \
  --dataset cholec \
  --raw data/raw/CholecT50 \
  --out data/processed
```

The native CholecT50 parser reads `labels/VIDxx.json`, maps each frame to `videos/VIDxx/000001.png`, and creates four QA tasks:

- phase recognition: `What surgical phase is shown?`
- tool recognition: `Which surgical instruments are visible?`
- action recognition: `What surgical actions are visible?`
- triplet recognition: `What surgical action triplets <instrument, verb, target> are visible?`

It uses the official CholecT50 cross-validation fold 1 as the default split, with a held-out validation subset from the remaining folds. Run this before training:

```bash
python scripts/preflight_train.py --config configs/gemma4_12b_lora_sft.yaml
```

If the browser download shows `Zero KB` or `stopped`, the archive has not downloaded correctly yet. Restart the download from the one-time access page, wait until the full archive finishes, then copy the completed file to the VM.

This project is for research and education only. It is not clinical decision support.
