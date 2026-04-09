from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_reasoning import load_experiment_config
from emotion_reasoning.evaluation import compare_with_baselines
from emotion_reasoning.training import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained multimodal emotion model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--fusion-mode", choices=["vision", "text", "multimodal"], default=None)
    parser.add_argument("--baseline-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    metrics = evaluate_model(
        config=config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        fusion_mode=args.fusion_mode
    )
    payload: dict[str, object] = {"metrics": metrics}
    if args.baseline_file:
        payload["comparison"] = compare_with_baselines(
            our_results=metrics,
            baseline_file=args.baseline_file,
            metric_key="mAP"
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
