"""Cross-attention visualization utilities."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from emotion_reasoning.utils.io import ensure_dir


def aggregate_cross_attention(cross_attentions: list[torch.Tensor]) -> torch.Tensor:
    if not cross_attentions:
        raise ValueError("No cross-attention maps were returned by the model.")
    attention = cross_attentions[-1].detach().cpu()
    if attention.ndim != 4:
        raise ValueError(f"Expected attention shape [B, H, Q, V], found {attention.shape}")
    attention = attention.mean(dim=1).mean(dim=1)
    token_count = attention.size(-1)
    if token_count > 1 and int(math.sqrt(token_count - 1)) ** 2 == token_count - 1:
        attention = attention[:, 1:]
        token_count -= 1
    grid_size = int(math.sqrt(token_count))
    if grid_size * grid_size != token_count:
        raise ValueError(f"Cannot reshape {token_count} visual tokens into a square grid.")
    return attention.reshape(attention.size(0), grid_size, grid_size)


def overlay_attention_on_image(
    image: Image.Image,
    attention_grid: torch.Tensor,
    output_path: str | Path,
    alpha: float = 0.45
) -> Path:
    ensure_dir(Path(output_path).parent)
    image_array = np.array(image)
    heatmap = F.interpolate(
        attention_grid.unsqueeze(0).unsqueeze(0),
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_array)
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout()
    output = Path(output_path)
    plt.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close()
    return output
