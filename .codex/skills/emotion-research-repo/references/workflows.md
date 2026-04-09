# Repo Workflows

## Research Intent

This repository is organized around contextual emotion recognition, not face-only classification.

Keep this default framing:

1. Generate context-aware text with a VLM.
2. Fuse text and visual tokens with a Q-Former-inspired model.
3. Evaluate with metrics that support paper-ready reporting such as `mAP` and `AUC-ROC`.

## Default Benchmarks

- `EMOTIC`: main multi-label contextual benchmark
- `CAER-S`: main single-label contextual benchmark
- `BoLD`: optional extension benchmark

Do not silently switch the main benchmark away from `EMOTIC` unless the user asks.

## Entry Points

- `scripts/generate_pseudo_labels.py`
  - use for stage-1 semantic pseudo-label generation
  - output contract is `semantic_pseudo_label`
- `scripts/train.py`
  - use for end-to-end training from a config preset
- `scripts/evaluate.py`
  - use for checkpoint metrics and optional baseline comparison
- `scripts/run_ablation.py`
  - use for the three core ablations
- `scripts/visualize_attention.py`
  - use for qualitative cross-attention figures

## Architecture Notes

- The generic dataset loader lives in `src/emotion_reasoning/datasets/base.py`.
- The Q-Former implementation lives in `src/emotion_reasoning/modeling/qformer.py`.
- The multimodal wrapper model lives in `src/emotion_reasoning/modeling/multimodal_model.py`.
- The training loop and checkpointing live in `src/emotion_reasoning/training/trainer.py`.
- Metrics and qualitative analysis helpers live in `src/emotion_reasoning/evaluation/`.

## Common Modification Patterns

### Add or adapt a dataset

- Update class names in `src/emotion_reasoning/constants.py`.
- Add or update a preset in `configs/`.
- Keep dataset-specific parsing isolated to the loader.
- Preserve the fields `image_path`, `labels`, `split`, optional `bbox`, optional `sample_id`, and optional `semantic_pseudo_label` unless there is a strong reason to change the contract.

### Change prompting

- Edit `src/emotion_reasoning/vlm/pseudo_labeler.py`.
- Keep bbox-aware and bbox-free prompting in sync.
- Preserve the research goal of context-rich reasoning, not short captioning.

### Change fusion or training

- Keep differential learning rates explicit.
- Keep multi-label vs single-label behavior aligned with the dataset config.
- Do not break `vision`, `text`, and `multimodal` ablation support unless the user requests a redesign.

## Pitfalls

- Python `3.13` may not be the most reliable environment for this stack.
- Missing `torch` or `transformers` means only static validation may be possible.
- `BoLD` should be checked carefully before publication-level claims because label assumptions may need refinement.
- Attention visualization depends on cross-attention outputs and is not informative for text-only runs.
