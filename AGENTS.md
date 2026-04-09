# Agent Guide

## Purpose

This repository implements a two-stage contextual emotion recognition pipeline inspired by the paper "VLLMs Provide Better Context for Emotion Understanding Through Common Sense Reasoning".

Treat this repo as a research baseline, not a finished product. Favor reproducibility, modular edits, and explicit experiment assumptions.

## Default Research Framing

- Use `EMOTIC` as the default main benchmark.
- Use `CAER-S` as the default single-label contextual benchmark.
- Use `BoLD` only as an optional extension benchmark.
- Preserve the two-stage framing unless the user explicitly asks for a different design:
  1. Stage 1 generates `semantic_pseudo_label` text with a VLM.
  2. Stage 2 fuses image and text with a Q-Former-style multimodal classifier.

## Repo Map

- `README.md`: user-facing overview and example commands.
- `main_notebook.ipynb`: exploratory notebook; useful for quick inspection, but scripts and `src/` are the canonical pipeline.
- `configs/`: experiment presets for EMOTIC, CAER-S, and BoLD.
- `scripts/generate_pseudo_labels.py`: stage-1 CLI for VLM pseudo-label generation.
- `scripts/train.py`: main training entry point.
- `scripts/evaluate.py`: checkpoint evaluation and optional baseline comparison.
- `scripts/run_ablation.py`: `vision`, `text`, and `multimodal` ablations.
- `scripts/visualize_attention.py`: cross-attention overlay export.
- `src/emotion_reasoning/datasets/`: generic annotation loader and collator.
- `src/emotion_reasoning/vlm/`: VLM adapters and prompting logic.
- `src/emotion_reasoning/modeling/`: Q-Former and multimodal classifier.
- `src/emotion_reasoning/training/`: optimizer setup and training loop.
- `src/emotion_reasoning/evaluation/`: metrics, ablations, attention visualization, and SOTA comparison helpers.
- `tests/`: lightweight smoke tests for metrics and Q-Former tensor shapes.

## Current Implementation Assumptions

- Annotation files may be `csv`, `json`, or `jsonl`.
- Expected record fields are `image_path`, `labels`, `split`, optional `bbox`, optional `sample_id`, and optional `semantic_pseudo_label`.
- Multi-label tasks use `BCEWithLogitsLoss`.
- Single-label tasks use `CrossEntropyLoss`.
- Validation early stopping monitors validation `mAP`.
- Stage-1 prompting draws a red box only when `bbox` is present; otherwise it uses a no-box prompt variant.

## Recommended Editing Workflow

When working on this repo:

1. Identify which layer the request touches:
   - dataset schema or splits,
   - VLM pseudo-label generation,
   - multimodal fusion model,
   - training protocol,
   - evaluation and analysis.
2. Read the matching script entry point first, then the underlying module in `src/emotion_reasoning/`.
3. Keep config-driven behavior whenever possible instead of hardcoding dataset or model details.
4. Preserve ablation compatibility for `vision`, `text`, and `multimodal` unless the user asks to remove it.
5. Prefer small patches that keep the stage separation clear.

## Common Task Routing

- Add a new dataset:
  - start at `src/emotion_reasoning/datasets/base.py`
  - update class names in `src/emotion_reasoning/constants.py`
  - add a new config in `configs/`
- Change pseudo-label prompting:
  - edit `src/emotion_reasoning/vlm/pseudo_labeler.py`
  - keep both bbox and no-bbox behavior aligned
- Change fusion architecture:
  - edit `src/emotion_reasoning/modeling/qformer.py`
  - then check `src/emotion_reasoning/modeling/multimodal_model.py`
- Change training rules:
  - edit `src/emotion_reasoning/training/trainer.py`
  - keep optimizer groups in `src/emotion_reasoning/training/optim.py` consistent
- Change metrics or paper tables:
  - edit `src/emotion_reasoning/evaluation/metrics.py`
  - use `src/emotion_reasoning/evaluation/sota.py` for baseline comparisons

## Important Caveats

- The current workspace Python is `3.13`, but the repo recommends `3.10` to `3.12` for smoother PyTorch and Transformers compatibility.
- Runtime validation may fail until `torch`, `transformers`, and related dependencies are installed.
- The `BoLD` config exists as a research extension path; confirm its exact label taxonomy before claiming publication-ready results.
- Attention visualization requires cross-attention outputs, so it is meaningful for multimodal or vision-conditioned runs, not pure text-only analysis.
- If filesystem bytecode writes fail, syntax checks can still be done with `python -B` and `compile(...)` without relying on `__pycache__`.

## What Good Changes Look Like

- New experiments should be reproducible through a config file and an existing script entry point.
- Dataset-specific logic should remain minimal and well isolated.
- Metrics should be exportable for tables and qualitative analysis.
- Explanations and comments should reflect the research intent: contextual emotion understanding with VLM-derived semantic cues.
