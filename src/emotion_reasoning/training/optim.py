"""Optimizer builders with differential learning rates."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from emotion_reasoning.config import TrainingConfig
from emotion_reasoning.modeling.multimodal_model import MultimodalEmotionModel


def _split_decay_parameters(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def _extend_groups(
    groups: list[dict[str, object]],
    params: tuple[list[nn.Parameter], list[nn.Parameter]],
    learning_rate: float,
    weight_decay: float
) -> None:
    decay, no_decay = params
    if decay:
        groups.append({"params": decay, "lr": learning_rate, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": learning_rate, "weight_decay": 0.0})


def build_optimizer(model: MultimodalEmotionModel, config: TrainingConfig) -> torch.optim.Optimizer:
    parameter_groups: list[dict[str, object]] = []
    _extend_groups(
        parameter_groups,
        _split_decay_parameters(model.vision_encoder),
        config.vision_lr,
        config.weight_decay
    )
    _extend_groups(
        parameter_groups,
        _split_decay_parameters(model.text_encoder),
        config.text_lr,
        config.weight_decay
    )
    qformer_modules: Iterable[nn.Module] = [
        model.qformer,
        model.visual_projection,
        model.text_projection
    ]
    qformer_decay: list[nn.Parameter] = []
    qformer_no_decay: list[nn.Parameter] = []
    for module in qformer_modules:
        decay, no_decay = _split_decay_parameters(module)
        qformer_decay.extend(decay)
        qformer_no_decay.extend(no_decay)
    _extend_groups(
        parameter_groups,
        (qformer_decay, qformer_no_decay),
        config.qformer_lr,
        config.weight_decay
    )
    head_modules: Iterable[nn.Module] = [model.multimodal_head, model.vision_head, model.text_head]
    head_decay: list[nn.Parameter] = []
    head_no_decay: list[nn.Parameter] = []
    for module in head_modules:
        decay, no_decay = _split_decay_parameters(module)
        head_decay.extend(decay)
        head_no_decay.extend(no_decay)
    _extend_groups(parameter_groups, (head_decay, head_no_decay), config.head_lr, config.weight_decay)
    return torch.optim.AdamW(parameter_groups)
