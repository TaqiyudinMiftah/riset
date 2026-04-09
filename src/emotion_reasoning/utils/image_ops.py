"""Image helpers used by datasets and VLM prompting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def draw_red_box(image: Image.Image, bbox: list[float] | tuple[float, ...] | dict[str, Any], width: int = 5) -> Image.Image:
    boxed_image = image.copy()
    draw = ImageDraw.Draw(boxed_image)
    x1, y1, x2, y2 = parse_bbox(bbox)
    draw.rectangle((x1, y1, x2, y2), outline="red", width=width)
    return boxed_image


def parse_bbox(bbox: list[float] | tuple[float, ...] | dict[str, Any]) -> tuple[int, int, int, int]:
    if isinstance(bbox, str):
        text = bbox.strip()
        if text and text[0] in "[{":
            bbox = json.loads(text)
        else:
            raise ValueError(f"Unsupported bbox string format: {bbox}")
    if isinstance(bbox, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(bbox):
            return int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        if {"x", "y", "w", "h"}.issubset(bbox):
            return (
                int(bbox["x"]),
                int(bbox["y"]),
                int(bbox["x"] + bbox["w"]),
                int(bbox["y"] + bbox["h"])
            )
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(x2), int(y2)
    raise ValueError(f"Unsupported bbox format: {bbox}")
