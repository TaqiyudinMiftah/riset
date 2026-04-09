# Multimodal Emotion Recognition with VLM Pseudo-Labels

Baseline research codebase for the two-stage pipeline described in "VLLMs Provide Better Context for Emotion Understanding Through Common Sense Reasoning".

The implementation is organized around:

1. `Stage 1`: extract context-aware semantic pseudo-labels with a VLM such as LLaVA-1.5 or Moondream2.
2. `Stage 2`: train a multimodal emotion classifier that fuses visual features and pseudo-label text through a Q-Former inspired by InstructBLIP.

## Features

- Red-box prompting for subject-aware VLM context extraction.
- Generic dataset loader for EMOTIC, CAER-S, and BoLD-style annotations.
- CLIP-ViT / ViT visual backbone support through Hugging Face.
- BERT/RoBERTa text encoder with multimodal Q-Former fusion.
- Differential learning rates for vision encoder, Q-Former, and classification head.
- Multi-label and single-label training pipelines.
- Research metrics: mAP, AUC-ROC, ablations, and cross-attention map visualization.
- SOTA comparison utility driven by a JSON baseline file.

## Recommended Environment

Use Python `3.10` to `3.12` for the smoothest PyTorch and Transformers compatibility. The current workspace Python is `3.13`, which may require building some wheels from source depending on your environment.

## Recommended Datasets For This Research

For context-aware emotion recognition, the most appropriate and commonly used benchmarks in this repo are:

- `EMOTIC`: primary benchmark for contextual and multi-label emotion recognition.
- `CAER-S`: standard single-label contextual emotion benchmark.
- `BoLD`: optional multi-label benchmark when you want broader body-language context.

If you later want a face-only comparison baseline, you can extend the same codebase to AffectNet or RAF-DB, but for the paper direction in this project, `EMOTIC + CAER-S` should be the default pair.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Expected Annotation Schema

The training and pseudo-label scripts accept `csv`, `json`, or `jsonl` annotations with the following flexible fields:

- `image_path`: relative or absolute path to the image.
- `labels`: list of class names, pipe-separated labels, or a single class name/index.
- `split`: `train`, `val`, or `test`.
- `bbox`: optional `[x1, y1, x2, y2]` list or dict.
- `sample_id`: optional stable key for merging pseudo-label files.
- `semantic_pseudo_label`: optional text generated in stage 1.

Minimal EMOTIC-style example:

```json
{
  "sample_id": "emotic_0001",
  "image_path": "train/0001.jpg",
  "labels": ["Happiness", "Engagement"],
  "bbox": [41, 22, 189, 276],
  "split": "train"
}
```

Minimal CAER-S example:

```json
{
  "sample_id": "caers_0042",
  "image_path": "images/0042.jpg",
  "labels": "Happy",
  "split": "train"
}
```

## Stage 1: Generate Semantic Pseudo-Labels

```bash
python scripts/generate_pseudo_labels.py \
  --annotation-path data/emotic_annotations.jsonl \
  --image-root data/EMOTIC \
  --output-path outputs/emotic_pseudo_labels.jsonl \
  --dataset-name emotic \
  --vlm-type llava \
  --vlm-model llava-hf/llava-1.5-7b-hf
```

This will:

- draw a red box when a bounding box exists,
- prompt the VLM with the emotion label set,
- save the generated description as `semantic_pseudo_label`.

## Stage 2: Train the Q-Former Classifier

Primary multimodal EMOTIC run:

```bash
python scripts/train.py --config configs/emotic_multimodal.json
```

Optional multimodal BoLD run:

```bash
python scripts/train.py --config configs/bold_multimodal.json
```

Vision-only ablation:

```bash
python scripts/train.py --config configs/emotic_multimodal.json --fusion-mode vision
```

Text-only ablation:

```bash
python scripts/train.py --config configs/emotic_multimodal.json --fusion-mode text
```

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/emotic_multimodal.json \
  --checkpoint outputs/emotic_qformer/best.pt \
  --split test \
  --baseline-file configs/sota_baselines.example.json
```

## Visualize Cross-Attention

```bash
python scripts/visualize_attention.py \
  --config configs/emotic_multimodal.json \
  --checkpoint outputs/emotic_qformer/best.pt \
  --split test \
  --sample-index 0 \
  --output-path outputs/attention_overlay.png
```

## Ablation Suite

```bash
python scripts/run_ablation.py --config configs/emotic_multimodal.json
```

## Project Layout

```text
configs/
scripts/
src/emotion_reasoning/
  datasets/
  evaluation/
  modeling/
  training/
  utils/
  vlm/
tests/
```

## Notes for Publication-Focused Experiments

- Add a second prompt template that asks the VLM to describe facial Action Units.
- Compare pseudo-label quality from LLaVA against Moondream2 and pure captioning baselines.
- Export attention overlays for qualitative figures in the paper.
- Log per-class AP to support error analysis and long-tail discussion.
