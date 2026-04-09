from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_reasoning import load_experiment_config
from emotion_reasoning.training import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Q-Former emotion recognition model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fusion-mode", choices=["vision", "text", "multimodal"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    results = train_experiment(config=config, fusion_mode=args.fusion_mode)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
