from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_reasoning import load_experiment_config
from emotion_reasoning.evaluation.ablation import run_ablation_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vision/text/multimodal ablations.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    summary = run_ablation_suite(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
