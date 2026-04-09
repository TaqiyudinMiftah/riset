"""Dataset label constants and helpers."""

from __future__ import annotations

from typing import Iterable

EMOTIC_CLASSES = [
    "Affection",
    "Anger",
    "Annoyance",
    "Anticipation",
    "Aversion",
    "Confidence",
    "Disapproval",
    "Disconnection",
    "Disquietment",
    "Doubt/Confusion",
    "Embarrassment",
    "Engagement",
    "Esteem",
    "Excitement",
    "Fatigue",
    "Fear",
    "Happiness",
    "Pain",
    "Peace",
    "Pleasure",
    "Sadness",
    "Sensitivity",
    "Suffering",
    "Surprise",
    "Sympathy",
    "Yearning"
]

CAER_S_CLASSES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

KNOWN_DATASET_CLASSES = {
    "emotic": EMOTIC_CLASSES,
    "bold": EMOTIC_CLASSES,
    "caer-s": CAER_S_CLASSES,
    "caers": CAER_S_CLASSES
}


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def get_class_names(dataset_name: str, custom_classes: Iterable[str] | None = None) -> list[str]:
    if custom_classes:
        return [str(item).strip() for item in custom_classes if str(item).strip()]
    normalized = normalize_dataset_name(dataset_name)
    if normalized not in KNOWN_DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Please provide `class_names` explicitly in the config."
        )
    return KNOWN_DATASET_CLASSES[normalized]
