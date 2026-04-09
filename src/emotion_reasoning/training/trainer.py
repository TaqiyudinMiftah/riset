"""Training and evaluation loop."""

from __future__ import annotations

from dataclasses import asdict
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from emotion_reasoning.config import ExperimentConfig, save_experiment_config
from emotion_reasoning.datasets import EmotionBatchCollator, build_dataset
from emotion_reasoning.evaluation.metrics import compute_classification_metrics
from emotion_reasoning.modeling import MultimodalEmotionModel
from emotion_reasoning.training.optim import build_optimizer
from emotion_reasoning.utils.io import ensure_dir, load_checkpoint, save_checkpoint, save_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dataloader(
    config: ExperimentConfig,
    split: str,
    tokenizer: Any,
    image_processor: Any,
    shuffle: bool
) -> DataLoader:
    dataset = build_dataset(config.dataset, split=split)
    collator = EmotionBatchCollator(
        vision_processor=image_processor,
        tokenizer=tokenizer,
        max_text_length=config.dataset.max_text_length,
        task_type=config.dataset.task_type
    )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.dataset.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )


def _build_loss_fn(task_type: str) -> nn.Module:
    if task_type.lower() == "multilabel":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _compute_loss(loss_fn: nn.Module, logits: torch.Tensor, labels: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type.lower() == "multilabel":
        return loss_fn(logits, labels.float())
    return loss_fn(logits, labels.long())


def _forward_step(model: nn.Module, batch: dict[str, Any], device: torch.device, output_attentions: bool = False) -> dict[str, Any]:
    return model(
        pixel_values=batch["pixel_values"].to(device),
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        output_attentions=output_attentions
    )


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    task_type: str,
    gradient_clip_norm: float = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    losses: list[float] = []
    autocast_enabled = scaler is not None and scaler.is_enabled()
    progress = tqdm(loader, leave=False)
    for batch in progress:
        labels = batch["labels"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = _forward_step(model=model, batch=batch, device=device)
            logits = outputs["logits"]
            loss = _compute_loss(loss_fn, logits, labels, task_type)
        if training:
            assert optimizer is not None
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu().item()))
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        progress.set_description(f"{'train' if training else 'eval'} loss={losses[-1]:.4f}")
    metrics = compute_classification_metrics(
        logits=torch.cat(all_logits, dim=0),
        labels=torch.cat(all_labels, dim=0),
        task_type=task_type,
        class_names=getattr(loader.dataset, "class_names")
    )
    return float(np.mean(losses)) if losses else 0.0, metrics


def evaluate_model(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    split: str = "test",
    fusion_mode: str | None = None
) -> dict[str, Any]:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder_name, use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(config.model.vision_encoder_name)
    if fusion_mode is not None:
        config.model.fusion_mode = fusion_mode
    loader = _make_dataloader(config, split=split, tokenizer=tokenizer, image_processor=image_processor, shuffle=False)
    model = MultimodalEmotionModel(config.model, num_classes=config.num_classes).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = _build_loss_fn(config.dataset.task_type)
    loss, metrics = _epoch_pass(
        model=model,
        loader=loader,
        device=device,
        loss_fn=loss_fn,
        task_type=config.dataset.task_type,
        gradient_clip_norm=config.training.gradient_clip_norm,
        optimizer=None,
        scaler=None
    )
    metrics["loss"] = loss
    return metrics


def train_experiment(config: ExperimentConfig, fusion_mode: str | None = None) -> dict[str, Any]:
    if fusion_mode is not None:
        config.model.fusion_mode = fusion_mode
        output_path = Path(config.training.output_dir)
        if not output_path.name.endswith(f"_{fusion_mode}"):
            config.training.output_dir = str(output_path.parent / f"{output_path.name}_{fusion_mode}")

    set_seed(config.seed)
    device = get_device()
    output_dir = ensure_dir(config.training.output_dir)
    save_experiment_config(config, output_dir / "resolved_config.json")

    tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder_name, use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(config.model.vision_encoder_name)

    train_loader = _make_dataloader(config, "train", tokenizer, image_processor, shuffle=True)
    val_loader = _make_dataloader(config, "val", tokenizer, image_processor, shuffle=False)
    test_loader = _make_dataloader(config, "test", tokenizer, image_processor, shuffle=False)

    model = MultimodalEmotionModel(config.model, num_classes=config.num_classes).to(device)
    optimizer = build_optimizer(model, config.training)
    loss_fn = _build_loss_fn(config.dataset.task_type)
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.mixed_precision and device.type == "cuda")

    best_map = float("-inf")
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    patience_counter = 0

    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_metrics = _epoch_pass(
            model=model,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            task_type=config.dataset.task_type,
            gradient_clip_norm=config.training.gradient_clip_norm,
            optimizer=optimizer,
            scaler=scaler
        )
        val_loss, val_metrics = _epoch_pass(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            task_type=config.dataset.task_type,
            gradient_clip_norm=config.training.gradient_clip_norm,
            optimizer=None,
            scaler=None
        )
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
        history.append(epoch_summary)
        save_json(output_dir / "history.json", history)

        current_map = float(val_metrics.get("mAP", float("-inf")))
        if current_map > best_map:
            best_map = current_map
            patience_counter = 0
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_metrics": val_metrics
            }
            save_checkpoint(
                output_dir / "best.pt",
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "best_metrics": best_metrics
                }
            )
        else:
            patience_counter += 1

        save_checkpoint(
            output_dir / "last.pt",
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "history": history
            }
        )

        if patience_counter >= config.training.early_stopping_patience:
            break

    checkpoint = load_checkpoint(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = _epoch_pass(
        model=model,
        loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        task_type=config.dataset.task_type,
        gradient_clip_norm=config.training.gradient_clip_norm,
        optimizer=None,
        scaler=None
    )
    results = {
        "experiment_name": config.experiment_name,
        "fusion_mode": config.model.fusion_mode,
        "best_validation": best_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics
    }
    save_json(output_dir / "results.json", results)
    with (output_dir / "results.txt").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=2))
    return results
