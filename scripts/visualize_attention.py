from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoImageProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_reasoning import load_experiment_config
from emotion_reasoning.datasets import build_dataset
from emotion_reasoning.evaluation.attention_viz import aggregate_cross_attention, overlay_attention_on_image
from emotion_reasoning.modeling import MultimodalEmotionModel
from emotion_reasoning.training.trainer import get_device
from emotion_reasoning.utils.io import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cross-attention maps for one sample.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder_name, use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(config.model.vision_encoder_name)
    dataset = build_dataset(config.dataset, split=args.split)
    sample = dataset[args.sample_index]

    text_inputs = tokenizer(
        [sample["text"] or ""],
        padding=True,
        truncation=True,
        max_length=config.dataset.max_text_length,
        return_tensors="pt"
    )
    image_inputs = image_processor(images=[sample["image"]], return_tensors="pt")

    model = MultimodalEmotionModel(config.model, num_classes=config.num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        outputs = model(
            pixel_values=image_inputs["pixel_values"].to(device),
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=text_inputs["attention_mask"].to(device),
            output_attentions=True
        )

    attention_grid = aggregate_cross_attention(outputs["cross_attentions"])[0]
    saved_path = overlay_attention_on_image(sample["image"], attention_grid, args.output_path)
    print(json.dumps({"output_path": str(saved_path)}, indent=2))


if __name__ == "__main__":
    main()
