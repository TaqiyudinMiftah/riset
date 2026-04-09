"""Comparison utilities against reported baselines."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any


def compare_with_baselines(
    our_results: dict[str, Any],
    baseline_file: str | Path,
    metric_key: str = "mAP",
    method_name: str = "Q-Former Fusion"
) -> list[dict[str, Any]]:
    with Path(baseline_file).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    baselines = list(payload.get("baselines", []))
    our_score = float(our_results[metric_key])
    comparison = [
        {
            "method": item["method"],
            "score": float(item["score"]),
            "delta_vs_ours": our_score - float(item["score"]),
            "notes": item.get("notes", "")
        }
        for item in baselines
    ]
    comparison.append(
        {
            "method": method_name,
            "score": our_score,
            "delta_vs_ours": 0.0,
            "notes": "Current experiment"
        }
    )
    comparison.sort(key=lambda item: item["score"], reverse=True)
    return comparison
