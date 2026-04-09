"""I/O helpers for annotations, metrics, and checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and "records" in payload:
        return [dict(item) for item in payload["records"]]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".json":
        return _read_json(path_obj)
    if suffix == ".jsonl":
        return _read_jsonl(path_obj)
    if suffix == ".csv":
        return _read_csv(path_obj)
    raise ValueError(f"Unsupported annotation file format: {path_obj}")


def save_json(path: str | Path, payload: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    suffix = path_obj.suffix.lower()
    if suffix == ".json":
        save_json(path_obj, records)
        return
    if suffix == ".jsonl":
        save_jsonl(path_obj, records)
        return
    if suffix == ".csv":
        frame = pd.DataFrame.from_records(records)
        frame.to_csv(path_obj, index=False, quoting=csv.QUOTE_MINIMAL)
        return
    raise ValueError(f"Unsupported output file format: {path_obj}")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    torch.save(payload, path_obj)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
