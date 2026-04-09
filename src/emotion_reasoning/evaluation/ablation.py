"""Ablation runners for vision-only, text-only, and multimodal settings."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from emotion_reasoning.config import ExperimentConfig
from emotion_reasoning.training import train_experiment
from emotion_reasoning.utils.io import save_json


def run_ablation_suite(config: ExperimentConfig, modes: list[str] | None = None) -> dict[str, dict]:
    if modes is None:
        modes = ["vision", "text", "multimodal"]
    base_output = Path(config.training.output_dir)
    summary: dict[str, dict] = {}
    for mode in modes:
        mode_config = deepcopy(config)
        mode_config.model.fusion_mode = mode
        mode_config.training.output_dir = str(base_output.parent / f"{base_output.name}_{mode}")
        summary[mode] = train_experiment(mode_config)
    save_json(base_output.parent / f"{base_output.name}_ablation_summary.json", summary)
    return summary
