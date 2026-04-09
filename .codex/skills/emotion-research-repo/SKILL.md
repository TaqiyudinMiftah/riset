---
name: emotion-research-repo
description: Understand, modify, and extend this multimodal emotion recognition repository. Use when working on the two-stage pipeline, including dataset onboarding, VLM pseudo-label generation, Q-Former fusion, training, evaluation, ablations, or repo-specific experiment setup.
---

# Emotion Research Repo

## Overview

Use this skill to get oriented in this repository quickly and make changes that preserve the intended research framing: stage-1 VLM context extraction followed by stage-2 multimodal emotion classification with a Q-Former-style fusion block.

Read [AGENTS.md](../../../AGENTS.md) first for the repo map, defaults, and caveats.

## Quick Start

Use this routing:

- For pseudo-label generation work, read `scripts/generate_pseudo_labels.py` and `src/emotion_reasoning/vlm/pseudo_labeler.py`.
- For dataset or annotation changes, read `src/emotion_reasoning/datasets/base.py` and the target config in `configs/`.
- For model changes, read `src/emotion_reasoning/modeling/qformer.py` and `src/emotion_reasoning/modeling/multimodal_model.py`.
- For training behavior, read `scripts/train.py`, `src/emotion_reasoning/training/trainer.py`, and `src/emotion_reasoning/training/optim.py`.
- For evaluation, read `scripts/evaluate.py` and `src/emotion_reasoning/evaluation/`.

Read [references/workflows.md](references/workflows.md) when the task touches experiment flow, dataset defaults, or validation expectations.

## Working Rules

- Default to `EMOTIC` for the main benchmark unless the user explicitly asks for another dataset.
- Keep `CAER-S` support intact for single-label experiments.
- Preserve the three ablation modes: `vision`, `text`, and `multimodal`.
- Keep changes config-driven when possible.
- Treat `semantic_pseudo_label` as the contract between stage 1 and stage 2.
- If adding a dataset, update both the class list source and the config preset.
- If changing metrics or claims, prefer changes that support reproducible paper tables and qualitative figures.

## Validation

- Run lightweight syntax checks when runtime dependencies are unavailable.
- Run targeted tests in `tests/` when edits touch tensor shapes or metrics.
- Say clearly when training or import validation could not be executed because dependencies are missing.

## References

- [references/workflows.md](references/workflows.md): repo-specific task routing, assumptions, and pitfalls.
