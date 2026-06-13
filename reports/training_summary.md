# Surgical Video Assistant Training Summary

## Project Goal

Build a feasible surgical frame assistant for VQA, phase recognition, tool recognition, and action recognition without training a full video model.

## Method

- Normalize surgical image/frame annotations into a common JSONL schema.
- Run Gemma-style multimodal zero-shot/few-shot prompting.
- Optionally fine-tune with LoRA/QLoRA on frame-level QA pairs.
- Evaluate each task separately and save charts for presentation.

## Metrics

- Number of evaluated samples: `3`
- Overall exact match: `1.0000`
- Overall token F1: `1.0000`

| Task | Samples | Main score |
| --- | ---: | ---: |
| phase | 1 | 1.0000 |
| tool_type | 1 | 1.0000 |
| tool_count | 1 | 1.0000 |

## Artifacts

- `reports/figures/training_loss.png`: training/eval loss chart.
- `reports/figures/task_scores.png`: task-level score chart.
- `reports/training_log.csv`: step-level training logs.
- `checkpoints/gemma4-surgical-frame-lora`: LoRA checkpoints.

## Limitations

- This is a research prototype, not clinical decision support.
- Frame-level reasoning cannot fully capture temporal surgical context.
- Dataset access and licensing may restrict public release of data and trained weights.
