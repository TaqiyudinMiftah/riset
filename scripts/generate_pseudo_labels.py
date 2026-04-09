from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_reasoning.vlm import DEFAULT_PROMPT_TEMPLATE, generate_pseudo_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semantic pseudo-labels with a VLM.")
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--vlm-type", choices=["llava", "moondream"], required=True)
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--class-names", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    custom_classes = [item.strip() for item in args.class_names.split(",") if item.strip()] or None
    generate_pseudo_labels(
        annotation_path=args.annotation_path,
        image_root=args.image_root,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        vlm_type=args.vlm_type,
        vlm_model=args.vlm_model,
        prompt_template=args.prompt_template,
        custom_classes=custom_classes,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    print(f"Pseudo-labels saved to {args.output_path}")


if __name__ == "__main__":
    main()
