# Surgical Video Assistant Report Template

## Motivation

This project avoids full video training because frame-level multimodal prompting is more feasible under limited compute and restricted surgical dataset access. The assistant is evaluated as a research prototype, not a clinical decision system.

## Dataset Comparison

| Dataset | Role | Strength | Limitation |
| --- | --- | --- | --- |
| SurgMLLMBench | Primary | Surgical image/frame VQA with workflow, tool, action, and segmentation-style questions | Raw JSON/JSONL parsing is needed because schemas are mixed |
| CholecT50/45 | Secondary benchmark | Phase, instrument, verb, target, triplet labels | Requires CAMMA access request |
| Cholec80 | Backup/extension | Classic phase recognition benchmark | No native VQA format |
| EndoVis2018 | Extension | Segmentation and tool/anatomy grounding | Not the core phase/video assistant dataset |

## Method

- Normalize raw datasets into a shared frame-level JSONL schema.
- Build task-specific zero-shot and few-shot prompts.
- Run Gemma 4-style multimodal inference on individual frames.
- Evaluate phase, tool, count, action/triplet, and free-form VQA separately.

## Results

Add metrics from `reports/metrics_*.json` here.

## Error Analysis

Expected hard cases:

- Phase labels with visually similar contexts.
- Small, occluded, or motion-blurred instruments.
- Instrument action labels that need temporal context.
- Segmentation or anatomy questions when the model lacks pixel grounding.

## Limitations

- No full video temporal modeling.
- No clinical validation.
- No real-time guarantee.
- Dataset licenses are generally non-commercial.

## Future Work

- Temporal smoothing across sampled frames.
- Lightweight supervised classifier for phase/tool labels.
- Segmentation grounding with EndoVis or CholecSeg8k.
- LoRA fine-tuning if compute and license allow.
