# Surgical Video Assistant Final Report

## Objective

Build a frame-level multimodal surgical assistant for phase, tool, action, and triplet recognition without full video training.

## Method

- Base model: Gemma 4 12B multimodal model.
- Fine-tuning: QLoRA/LoRA supervised fine-tuning on surgical frame-question-answer pairs.
- Dataset: CholecT50 frame labels converted into QA templates.
- Split strategy: video-level subset, keeping train and eval procedures separate.
- Inference for label tasks: constrained prediction over valid label candidates.

## Training Result

- `train_runtime`: `5206`
- `train_samples_per_second`: `0.768`
- `train_steps_per_second`: `0.096`
- `train_loss`: `1.603`
- `epoch`: `0.03333`
- `eval_loss`: `nan`
- `eval_mean_token_accuracy`: `0.6905`
- `eval_epoch`: `3.333`

## Dataset Subset

- Number of samples: `1520`
- Missing images: `0`

### By Split

| Item | Count |
| --- | ---: |
| train | 1200 |
| val | 320 |

### By Task

| Item | Count |
| --- | ---: |
| phase | 380 |
| triplet | 380 |
| tool_type | 380 |
| action | 380 |

### By Video

| Item | Count |
| --- | ---: |
| VID18 | 247 |
| VID08 | 160 |
| VID01 | 132 |
| VID15 | 234 |
| VID10 | 202 |
| VID13 | 85 |
| VID04 | 103 |
| VID05 | 253 |
| VID12 | 104 |

## Figures

- `reports/final/training_loss_curve.png`
- `reports/final/token_accuracy_curve.png`
- `reports/final/learning_rate_curve.png`
- `reports/final/dataset_task_distribution.png`
- `reports/final/dataset_split_distribution.png`
- `reports/final/dataset_video_distribution.png`

## Artifacts

- Hugging Face checkpoint: https://huggingface.co/SpringWang08/surgical-gemma4-12b-lora
- Training config: `configs/gemma4_12b_lora_video_subset.yaml`
- Local checkpoint: `checkpoints/gemma4-12b-surgical-frame-lora-video-subset/checkpoint-500`
- Training logs: `reports/training_log.csv` and `reports/train_gemma4_12b_video_subset.log`

## Limitations

- This is a research prototype, not clinical decision support.
- The experiment uses a video-level subset, so it does not claim full CholecT50 leaderboard performance.
- The model is frame-level and does not model long temporal surgical dynamics.
- Raw CholecT50 data is not redistributed because of dataset license restrictions.

## Next Steps

- Run constrained evaluation on the full official split if more GPU/storage budget is available.
- Add temporal smoothing over adjacent frames for phase recognition.
- Add SSG-VQA alignment after verifying frame coverage from CholecT50/CholecT45.
