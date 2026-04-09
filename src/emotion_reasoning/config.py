"""Configuration dataclasses for the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .constants import get_class_names


@dataclass(slots=True)
class DatasetConfig:
    name: str
    annotation_path: str
    image_root: str
    task_type: str = "multilabel"
    image_column: str = "image_path"
    label_column: str = "labels"
    split_column: str = "split"
    sample_id_column: str = "sample_id"
    bbox_column: str = "bbox"
    pseudo_label_column: str = "semantic_pseudo_label"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_workers: int = 4
    max_text_length: int = 128
    class_names: list[str] = field(default_factory=list)

    def resolved_class_names(self) -> list[str]:
        return get_class_names(self.name, self.class_names)


@dataclass(slots=True)
class ModelConfig:
    vision_encoder_name: str = "openai/clip-vit-large-patch14"
    text_encoder_name: str = "roberta-base"
    num_queries: int = 32
    qformer_hidden_size: int = 768
    qformer_num_layers: int = 6
    qformer_num_heads: int = 8
    dropout: float = 0.35
    fusion_mode: str = "multimodal"
    freeze_vision_encoder: bool = False
    freeze_text_encoder: bool = False


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 20
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    weight_decay: float = 0.05
    vision_lr: float = 1e-5
    text_lr: float = 1e-4
    qformer_lr: float = 1e-4
    head_lr: float = 1e-3
    output_dir: str = "outputs/default"


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42

    @property
    def num_classes(self) -> int:
        return len(self.dataset.resolved_class_names())


def _coerce_dataset_config(payload: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(**payload)


def _coerce_model_config(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**payload)


def _coerce_training_config(payload: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**payload)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    dataset = _coerce_dataset_config(payload["dataset"])
    model = _coerce_model_config(payload.get("model", {}))
    training = _coerce_training_config(payload.get("training", {}))
    return ExperimentConfig(
        experiment_name=payload["experiment_name"],
        dataset=dataset,
        model=model,
        training=training,
        seed=int(payload.get("seed", 42))
    )


def save_experiment_config(config: ExperimentConfig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_name": config.experiment_name,
                "seed": config.seed,
                "dataset": asdict(config.dataset),
                "model": asdict(config.model),
                "training": asdict(config.training)
            },
            handle,
            indent=2
        )
