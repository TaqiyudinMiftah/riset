from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_reasoning.config import load_experiment_config
from emotion_reasoning.evaluation.ablation import run_ablation_suite
from emotion_reasoning.training.trainer import evaluate_model, train_experiment
from emotion_reasoning.utils.io import load_records, save_json, save_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare stage-2 training data from existing pseudo labels and run training/evaluation."
    )
    parser.add_argument(
        "--pseudo-label-source",
        default=str(PROJECT_ROOT / "notebook_outputs" / "risetv1_qwen" / "stage1_pseudo_labels_qwen.jsonl"),
    )
    parser.add_argument(
        "--prepared-output",
        default=str(PROJECT_ROOT / "notebook_outputs" / "risetv1_qwen" / "stage1_pseudo_labels_qwen_train10k_ready.jsonl"),
    )
    parser.add_argument(
        "--config-path",
        default=str(PROJECT_ROOT / "notebook_outputs" / "risetv1_qwen" / "configs" / "caers_qwen_qformer.json"),
    )
    parser.add_argument(
        "--summary-path",
        default=str(PROJECT_ROOT / "notebook_outputs" / "risetv1_qwen" / "stage2_train10k_summary.json"),
    )
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_ROOT / "caer_dataset" / "CAER-S"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "notebook_outputs" / "risetv1_qwen" / "stage2_qformer"),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-total-pseudo", type=int, default=10000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument("--text-lr", type=float, default=1e-4)
    parser.add_argument("--qformer-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--run-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-ablation", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--wandb-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="emotion-reasoning")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-tags", default="stage2,caers,pseudo-label")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-api-key", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_label(label: Any) -> str:
    text = str(label).strip()
    if text == "Anger":
        return "Angry"
    return text


def _stable_sample_id_for_stage2(image_path: str, split_name: str) -> str:
    rel_path = str(image_path).replace("\\", "/")
    return f"{split_name}__{rel_path.replace('/', '__').replace('.', '_')}"


def _flatten_metrics(prefix: str, payload: Any) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            nested = _flatten_metrics(f"{prefix}/{key}" if prefix else str(key), value)
            flattened.update(nested)
        return flattened
    if isinstance(payload, (int, float, bool, str)):
        flattened[prefix] = payload
        return flattened
    flattened[prefix] = str(payload)
    return flattened


def prepare_stage2_records(
    pseudo_label_source_path: Path,
    prepared_output_path: Path,
    image_root: Path,
    target_total_pseudo: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    if target_total_pseudo <= 0:
        raise ValueError("target_total_pseudo harus > 0")
    if val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("val_ratio dan test_ratio harus > 0")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio + test_ratio harus < 1")

    if not pseudo_label_source_path.exists():
        raise FileNotFoundError(f"Pseudo-label source tidak ditemukan: {pseudo_label_source_path}")

    source_rows = load_records(pseudo_label_source_path)
    if len(source_rows) == 0:
        raise ValueError("Pseudo-label source kosong.")

    selected_total = min(target_total_pseudo, len(source_rows))
    source_indices = np.arange(len(source_rows))
    source_labels = [_normalize_label(row.get("labels", "")) for row in source_rows]

    if selected_total < len(source_rows):
        try:
            selected_idx, _ = train_test_split(
                source_indices,
                train_size=selected_total,
                random_state=seed,
                stratify=source_labels,
            )
        except ValueError:
            selected_idx, _ = train_test_split(
                source_indices,
                train_size=selected_total,
                random_state=seed,
                stratify=None,
            )
        selected_idx = sorted(int(i) for i in selected_idx)
        selected_rows = [dict(source_rows[i]) for i in selected_idx]
    else:
        selected_rows = [dict(row) for row in source_rows[:selected_total]]

    usable_rows: list[dict[str, Any]] = []
    dropped_empty_text = 0
    dropped_missing_image = 0

    for row in selected_rows:
        image_ref = Path(str(row.get("image_path", "")))
        image_path = image_ref if image_ref.is_absolute() else image_root / image_ref
        if not image_path.exists():
            dropped_missing_image += 1
            continue

        pseudo_text = str(row.get("semantic_pseudo_label", "")).strip()
        if not pseudo_text:
            dropped_empty_text += 1
            continue

        updated = dict(row)
        updated["labels"] = _normalize_label(updated.get("labels", ""))
        usable_rows.append(updated)

    if len(usable_rows) < 20:
        raise ValueError(
            "Jumlah row pseudo-label yang usable terlalu sedikit setelah filtering. "
            f"usable={len(usable_rows)}"
        )

    usable_indices = np.arange(len(usable_rows))
    usable_labels = [_normalize_label(row.get("labels", "")) for row in usable_rows]
    holdout_ratio = val_ratio + test_ratio

    try:
        train_idx, holdout_idx = train_test_split(
            usable_indices,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=usable_labels,
        )
    except ValueError:
        train_idx, holdout_idx = train_test_split(
            usable_indices,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=None,
        )

    holdout_labels = [usable_labels[int(i)] for i in holdout_idx]
    test_share_in_holdout = test_ratio / holdout_ratio

    try:
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=test_share_in_holdout,
            random_state=seed,
            stratify=holdout_labels,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=test_share_in_holdout,
            random_state=seed,
            stratify=None,
        )

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for i in train_idx:
        row = dict(usable_rows[int(i)])
        row["split"] = "train"
        row["sample_id"] = _stable_sample_id_for_stage2(str(row["image_path"]), "train")
        train_records.append(row)

    for i in val_idx:
        row = dict(usable_rows[int(i)])
        row["split"] = "val"
        row["sample_id"] = _stable_sample_id_for_stage2(str(row["image_path"]), "val")
        val_records.append(row)

    for i in test_idx:
        row = dict(usable_rows[int(i)])
        row["split"] = "test"
        row["sample_id"] = _stable_sample_id_for_stage2(str(row["image_path"]), "test")
        test_records.append(row)

    stage2_records = train_records + val_records + test_records
    prepared_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_records(prepared_output_path, stage2_records)

    class_names = sorted({_normalize_label(row.get("labels", "")) for row in stage2_records})
    prep_stats = {
        "source_rows": len(source_rows),
        "selected_rows": len(selected_rows),
        "usable_rows": len(usable_rows),
        "dropped_empty_text": dropped_empty_text,
        "dropped_missing_image": dropped_missing_image,
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "test_rows": len(test_records),
        "total_rows": len(stage2_records),
    }
    return stage2_records, class_names, prep_stats


def build_config_payload(
    prepared_output_path: Path,
    image_root: Path,
    output_dir: Path,
    class_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "experiment_name": "caers_qwen_qformer_from_pseudo_labels",
        "seed": args.seed,
        "dataset": {
            "name": "caer-s",
            "annotation_path": str(prepared_output_path),
            "image_root": str(image_root),
            "task_type": "singlelabel",
            "image_column": "image_path",
            "label_column": "labels",
            "split_column": "split",
            "sample_id_column": "sample_id",
            "bbox_column": "bbox",
            "pseudo_label_column": "semantic_pseudo_label",
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "num_workers": args.num_workers,
            "max_text_length": args.max_text_length,
            "class_names": class_names,
        },
        "model": {
            "vision_encoder_name": "openai/clip-vit-base-patch32",
            "text_encoder_name": "roberta-base",
            "num_queries": 32,
            "qformer_hidden_size": 512,
            "qformer_num_layers": 4,
            "qformer_num_heads": 8,
            "dropout": 0.3,
            "fusion_mode": "multimodal",
            "freeze_vision_encoder": False,
            "freeze_text_encoder": False,
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "gradient_clip_norm": args.gradient_clip_norm,
            "mixed_precision": args.mixed_precision,
            "early_stopping_patience": args.early_stopping_patience,
            "weight_decay": args.weight_decay,
            "vision_lr": args.vision_lr,
            "text_lr": args.text_lr,
            "qformer_lr": args.qformer_lr,
            "head_lr": args.head_lr,
            "output_dir": str(output_dir),
        },
    }


def init_wandb(args: argparse.Namespace, config_payload: dict[str, Any]):
    if not args.wandb_enable or args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except Exception:
        print("W&B tidak tersedia. Install dengan: pip install wandb")
        return None

    api_key = args.wandb_api_key.strip() or os.getenv("WANDB_API_KEY", "").strip()
    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
        except Exception as exc:
            print(f"W&B login gagal, lanjut tanpa logging W&B. detail={exc}")
            return None

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "config": config_payload,
        "mode": args.wandb_mode,
        "tags": tags,
    }
    if args.wandb_entity.strip():
        init_kwargs["entity"] = args.wandb_entity.strip()
    if args.wandb_run_name.strip():
        init_kwargs["name"] = args.wandb_run_name.strip()

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"W&B init gagal, lanjut tanpa logging W&B. detail={exc}")
        return None


def wandb_log(run: Any, payload: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.log(payload)
    except Exception as exc:
        print(f"W&B log gagal: {exc}")


def main() -> None:
    args = parse_args()

    if load_dotenv is not None:
        load_dotenv()

    set_seed(args.seed)

    pseudo_label_source_path = Path(args.pseudo_label_source)
    prepared_output_path = Path(args.prepared_output)
    config_path = Path(args.config_path)
    summary_path = Path(args.summary_path)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    stage2_records, class_names, prep_stats = prepare_stage2_records(
        pseudo_label_source_path=pseudo_label_source_path,
        prepared_output_path=prepared_output_path,
        image_root=image_root,
        target_total_pseudo=args.target_total_pseudo,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    config_payload = build_config_payload(
        prepared_output_path=prepared_output_path,
        image_root=image_root,
        output_dir=output_dir,
        class_names=class_names,
        args=args,
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(config_path, config_payload)
    experiment_config = load_experiment_config(config_path)

    print("Pseudo-label source:", pseudo_label_source_path)
    print("Prepared stage-2 file:", prepared_output_path)
    print("Prepared rows:", len(stage2_records))
    print("Class names:", class_names)
    print("Config path:", config_path)

    run = init_wandb(args=args, config_payload=config_payload)
    wandb_log(run, {f"prep/{k}": v for k, v in prep_stats.items()})

    train_results: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None
    ablation_summary: dict[str, Any] | None = None

    try:
        if args.run_train:
            train_results = train_experiment(experiment_config)
            print("Training finished.")
            print(json.dumps(train_results, indent=2))
            wandb_log(run, _flatten_metrics("train", train_results))
        else:
            print("run_train=False, skip training.")

        best_checkpoint = Path(experiment_config.training.output_dir) / "best.pt"
        print("Best checkpoint:", best_checkpoint, "| exists:", best_checkpoint.exists())

        if args.run_eval:
            if not best_checkpoint.exists():
                raise FileNotFoundError(
                    f"Best checkpoint belum ditemukan: {best_checkpoint}. Jalankan training dulu."
                )
            eval_config = load_experiment_config(config_path)
            test_metrics = evaluate_model(
                config=eval_config,
                checkpoint_path=best_checkpoint,
                split="test",
                fusion_mode="multimodal",
            )
            print("Test metrics (multimodal):")
            print(json.dumps(test_metrics, indent=2))
            wandb_log(run, _flatten_metrics("eval", test_metrics))
        else:
            print("run_eval=False, skip evaluation.")

        if args.run_ablation:
            ablation_config = load_experiment_config(config_path)
            ablation_summary = run_ablation_suite(ablation_config)
            print("Ablation summary:")
            print(json.dumps(ablation_summary, indent=2))
            wandb_log(run, _flatten_metrics("ablation", ablation_summary))
        else:
            print("run_ablation=False, skip ablation.")

    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass

    summary = {
        "pseudo_label_source": str(pseudo_label_source_path),
        "prepared_output": str(prepared_output_path),
        "config_path": str(config_path),
        "prep_stats": prep_stats,
        "train_results": train_results,
        "test_metrics": test_metrics,
        "ablation_summary": ablation_summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary_path, summary)
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()