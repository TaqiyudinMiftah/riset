"""Dataset and collator definitions for emotion recognition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset

from emotion_reasoning.config import DatasetConfig
from emotion_reasoning.utils.io import load_records


def _maybe_parse_serialized(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_label_name(label: str) -> str:
    return str(label).strip()


class EmotionDataset(Dataset):
    """Generic dataset that supports multi-label and single-label emotion tasks."""

    def __init__(self, config: DatasetConfig, split: str):
        self.config = config
        self.split = split
        self.image_root = Path(config.image_root)
        self.class_names = config.resolved_class_names()
        self.label_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        self.records = self._load_split_records()

    def _load_split_records(self) -> list[dict[str, Any]]:
        records = load_records(self.config.annotation_path)
        split_column = self.config.split_column
        split_name = getattr(self.config, f"{self.split}_split", self.split)
        if records and split_column in records[0]:
            records = [record for record in records if str(record.get(split_column, "")).lower() == split_name.lower()]
        return records

    def _resolve_image_path(self, record: dict[str, Any]) -> Path:
        image_ref = record.get(self.config.image_column)
        if image_ref is None:
            raise KeyError(f"Missing image column '{self.config.image_column}' in record: {record}")
        candidate = Path(str(image_ref))
        return candidate if candidate.is_absolute() else self.image_root / candidate

    def _extract_text(self, record: dict[str, Any]) -> str:
        text = record.get(self.config.pseudo_label_column, "")
        if text is None:
            return ""
        return str(text)

    def _extract_bbox(self, record: dict[str, Any]) -> Any | None:
        bbox = record.get(self.config.bbox_column)
        if bbox in (None, "", "nan"):
            return None
        return _maybe_parse_serialized(bbox)

    def _encode_multilabel(self, label_value: Any) -> torch.Tensor:
        vector = torch.zeros(len(self.class_names), dtype=torch.float32)
        parsed = _maybe_parse_serialized(label_value)
        if isinstance(parsed, str):
            parsed = [item for item in parsed.split("|") if item.strip()]
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list-like labels for multi-label task, found: {label_value}")
        for label in parsed:
            normalized = _normalize_label_name(label)
            if normalized not in self.label_to_index:
                raise KeyError(f"Unknown label '{normalized}'. Available classes: {self.class_names}")
            vector[self.label_to_index[normalized]] = 1.0
        return vector

    def _encode_single_label(self, label_value: Any) -> torch.Tensor:
        parsed = _maybe_parse_serialized(label_value)
        if isinstance(parsed, int):
            return torch.tensor(parsed, dtype=torch.long)
        if isinstance(parsed, str) and parsed.isdigit():
            return torch.tensor(int(parsed), dtype=torch.long)
        normalized = _normalize_label_name(parsed)
        if normalized not in self.label_to_index:
            raise KeyError(f"Unknown label '{normalized}'. Available classes: {self.class_names}")
        return torch.tensor(self.label_to_index[normalized], dtype=torch.long)

    def _encode_labels(self, record: dict[str, Any]) -> torch.Tensor:
        label_value = record.get(self.config.label_column)
        if label_value is None:
            raise KeyError(f"Missing label column '{self.config.label_column}' in record: {record}")
        if self.config.task_type.lower() == "multilabel":
            return self._encode_multilabel(label_value)
        return self._encode_single_label(label_value)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self._resolve_image_path(record)
        image = Image.open(image_path).convert("RGB")
        return {
            "sample_id": str(record.get(self.config.sample_id_column, index)),
            "image_path": str(image_path),
            "image": image,
            "text": self._extract_text(record),
            "bbox": self._extract_bbox(record),
            "labels": self._encode_labels(record),
            "record": record
        }


class EmotionBatchCollator:
    """Converts raw PIL images and strings into tensors for the model."""

    def __init__(self, vision_processor: Any, tokenizer: Any, max_text_length: int, task_type: str):
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.task_type = task_type.lower()

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item["image"] for item in batch]
        texts = [item["text"] or "" for item in batch]
        image_inputs = self.vision_processor(images=images, return_tensors="pt")
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        if self.task_type == "multilabel":
            labels = torch.stack([item["labels"] for item in batch], dim=0)
        else:
            labels = torch.tensor([int(item["labels"].item()) for item in batch], dtype=torch.long)
        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
            "texts": texts,
            "records": [item["record"] for item in batch]
        }


def build_dataset(config: DatasetConfig, split: str) -> EmotionDataset:
    return EmotionDataset(config=config, split=split)
