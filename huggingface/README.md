---
base_model: google/gemma-4-12B
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - gemma-4
  - peft
  - lora
  - qlora
  - vision-language
  - multimodal
  - surgical-ai
  - surgical-phase-recognition
  - medical-imaging
license: other
language:
  - en
datasets:
  - CAMMA/CholecT50
model-index:
  - name: surgical-gemma4-12b-lora
    results:
      - task:
          type: image-text-to-text
          name: Surgical Frame QA and Recognition
        dataset:
          name: CholecT50 video-level subset
          type: custom
        metrics:
          - type: train_loss
            value: 1.603
            name: Final train loss
          - type: mean_token_accuracy
            value: 0.6905
            name: Eval mean token accuracy
---

# Surgical Gemma 4 12B LoRA

This repository contains a **LoRA adapter** fine-tuned for a research prototype called **Surgical Video Assistant**. The model is designed for **frame-level surgical image understanding**, not full temporal video modeling.

The adapter is loaded on top of **Gemma 4 12B multimodal** and was trained on surgical frame-question-answer pairs converted from CholecT50 labels.

> Research and education only. This model is **not clinical decision support** and must not be used for diagnosis, treatment, or intraoperative decision-making.

## What This Model Does

The project focuses on a practical alternative to full surgical video training:

- surgical phase recognition
- surgical instrument/tool recognition
- surgical action recognition
- surgical triplet recognition: `<instrument, verb, target>`
- frame-level surgical QA using template-style prompts

For label-based tasks such as phase recognition, the recommended inference mode is **constrained prediction** over valid label candidates. This avoids free-form hallucinated text and makes the output easier to evaluate.

## Model Type

This is **not a standalone full model**.

It is a **PEFT/LoRA adapter** that should be used with:

- Base model: `google/gemma-4-12B`
- Processor/tokenizer: `google/gemma-4-12B-it`
- Adapter checkpoint: this repository

Conceptually:

```text
Gemma 4 12B multimodal base model
        +
Surgical Gemma 4 12B LoRA adapter
        =
Surgical frame assistant for image + text QA
```

## Training Data

Training used a **video-level subset of CholecT50**.

CholecT50 contains laparoscopic cholecystectomy frames with frame-level labels for:

- surgical phases
- instruments
- verbs/actions
- targets/anatomies
- action triplets

The raw CholecT50 images/videos are **not redistributed** in this repository. Users must obtain CholecT50 from CAMMA under the dataset license and access terms.

### Subset Strategy

The experiment uses a controlled video-level subset:

- Train videos: `VID05`, `VID08`, `VID10`, `VID12`, `VID15`, `VID18`
- Eval videos: `VID01`, `VID04`, `VID13`
- Max train samples per task: `400`
- Max eval samples per task: `100`

This keeps train/eval procedures separate and avoids direct frame overlap. It is intended as a **fast, presentation-ready feasibility experiment**, not a full CholecT50 leaderboard benchmark.

## Training Setup

- Fine-tuning method: QLoRA / LoRA SFT
- Steps: `500`
- Runtime: `5206s` on 1x A100 80GB
- Final train loss: `1.603`
- Eval mean token accuracy: `0.6905`
- Eval loss: unstable / `nan` in this short mixed-precision subset run
- Checkpoints saved at steps `300`, `400`, and `500`

Uploaded artifacts include:

- `checkpoint-500/adapter_model.safetensors`
- tokenizer/processor files
- training config
- subset summary
- final reports and charts under `reports/final`

## Results Summary

This run demonstrates:

- the full pipeline works end-to-end:
  dataset parsing -> QA conversion -> LoRA fine-tuning -> checkpoint -> constrained prediction demo
- the model can be adapted to surgical frame label tasks
- constrained label prediction gives cleaner outputs for phase/tool/action recognition than free-form generation

The experiment does **not** claim full benchmark performance against surgical video models such as Rendezvous. Full comparison requires the official CholecT50 split and metric protocol.

## How to Use

Clone the project code:

```bash
git clone https://github.com/QuangVoAI/surgical-video-assistant.git
cd surgical-video-assistant
python -m pip install -r requirements-train.txt
```

Download or prepare CholecT50 locally according to its access terms. Then run prediction with the adapter checkpoint:

```bash
python scripts/predict_lora.py \
  --checkpoint checkpoints/gemma4-12b-surgical-frame-lora-video-subset/checkpoint-500 \
  --image data/raw/CholecT50/CholecT50/videos/VID01/000100.png \
  --question "What surgical phase is shown?" \
  --task-type phase
```

For constrained tool recognition:

```bash
python scripts/predict_lora.py \
  --checkpoint checkpoints/gemma4-12b-surgical-frame-lora-video-subset/checkpoint-500 \
  --image data/raw/CholecT50/CholecT50/videos/VID01/000100.png \
  --question "Which surgical instruments are visible?" \
  --task-type tool_type \
  --choices grasper bipolar hook scissors clipper irrigator
```

Example constrained phase output:

```text
scores:
  0.0073    Clipping and Cutting
  0.0090    Calot Triangle Dissection
  0.0100    Cleaning and Coagulation
  ...
prediction:
Clipping and Cutting
```

## Recommended Demo Story

1. Show the motivation: full surgical video training is expensive and hard to reproduce.
2. Explain the scoped solution: frame-level multimodal QA and recognition using Gemma 4.
3. Show dataset conversion: CholecT50 labels -> QA templates.
4. Show training charts: loss curve, token accuracy curve, task/video distribution.
5. Show constrained prediction on a surgical frame.
6. State limitations clearly: subset training, no clinical deployment, no temporal modeling.

## Limitations

- The model is frame-level and does not model surgical temporal dynamics.
- The run uses a video-level subset, not the full official CholecT50 benchmark.
- Free-form generation can be unstable; constrained label prediction is recommended for label tasks.
- This model can make incorrect predictions, especially on occluded tools, ambiguous phases, or frames requiring temporal context.
- This is not a medical device and not suitable for clinical use.

## License and Data Use

This adapter was trained for non-commercial research/education. The underlying dataset, CholecT50, is released by CAMMA under **CC BY-NC-SA 4.0** terms. Raw dataset frames are not redistributed here.

Use of the Gemma base model is subject to Google's Gemma terms. Users are responsible for complying with both the base model license and the dataset license.

## Citation

If you use the CholecT50 dataset, cite:

```bibtex
@article{nwoye2022rendezvous,
  title={Rendezvous: Attention mechanisms for the recognition of surgical action triplets in endoscopic videos},
  author={Nwoye, Chinedu Innocent and Yu, Tong and Gonzalez, Cristians and Seeliger, Benjamin and Mascagni, Pietro and Mutter, Didier and Marescaux, Jacques and Padoy, Nicolas},
  journal={Medical Image Analysis},
  volume={78},
  pages={102433},
  year={2022}
}
```

For official CholecT50 splits and benchmarking protocol:

```bibtex
@article{nwoye2022splits,
  title={Data splits and metrics for method benchmarking on surgical action triplet datasets},
  author={Nwoye, Chinedu Innocent and Padoy, Nicolas},
  journal={arXiv preprint arXiv:2204.05235},
  year={2022}
}
```

Project code:

```text
https://github.com/QuangVoAI/surgical-video-assistant
```
