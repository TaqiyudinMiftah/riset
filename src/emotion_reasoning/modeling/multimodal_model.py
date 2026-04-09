"""Multimodal emotion classifier with Q-Former fusion."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, CLIPVisionModel

from emotion_reasoning.config import ModelConfig
from emotion_reasoning.modeling.qformer import QFormerEncoder


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class MultimodalEmotionModel(nn.Module):
    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.fusion_mode = config.fusion_mode.lower()
        if self.fusion_mode not in {"multimodal", "vision", "text"}:
            raise ValueError(f"Unsupported fusion mode: {config.fusion_mode}")

        self.vision_encoder = self._build_vision_encoder(config.vision_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)

        if config.freeze_vision_encoder:
            _freeze_module(self.vision_encoder)
        if config.freeze_text_encoder:
            _freeze_module(self.text_encoder)

        vision_hidden_size = int(self.vision_encoder.config.hidden_size)
        text_hidden_size = int(self.text_encoder.config.hidden_size)

        self.visual_projection = nn.Linear(vision_hidden_size, config.qformer_hidden_size)
        self.text_projection = nn.Linear(text_hidden_size, config.qformer_hidden_size)

        self.qformer = QFormerEncoder(
            num_queries=config.num_queries,
            hidden_size=config.qformer_hidden_size,
            num_layers=config.qformer_num_layers,
            num_heads=config.qformer_num_heads,
            dropout=config.dropout
        )

        self.multimodal_head = ClassificationHead(config.qformer_hidden_size, num_classes, config.dropout)
        self.vision_head = ClassificationHead(config.qformer_hidden_size, num_classes, config.dropout)
        self.text_head = ClassificationHead(config.qformer_hidden_size, num_classes, config.dropout)

    @staticmethod
    def _build_vision_encoder(model_name: str) -> nn.Module:
        if "clip" in model_name.lower():
            return CLIPVisionModel.from_pretrained(model_name)
        return AutoModel.from_pretrained(model_name)

    def _encode_text(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.fusion_mode == "vision":
            return None, None
        if input_ids is None or attention_mask is None:
            raise ValueError("Text inputs are required for text or multimodal fusion.")
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_states = self.text_projection(text_outputs.last_hidden_state)
        return text_states, attention_mask

    def _encode_vision(self, pixel_values: torch.Tensor | None) -> torch.Tensor | None:
        if self.fusion_mode == "text":
            return None
        if pixel_values is None:
            raise ValueError("Pixel values are required for vision or multimodal fusion.")
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        return self.visual_projection(vision_outputs.last_hidden_state)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False
    ) -> dict[str, Any]:
        text_states, text_mask = self._encode_text(input_ids=input_ids, attention_mask=attention_mask)
        visual_states = self._encode_vision(pixel_values=pixel_values)
        batch_size = 0
        if input_ids is not None:
            batch_size = input_ids.size(0)
        elif pixel_values is not None:
            batch_size = pixel_values.size(0)
        else:
            raise ValueError("At least one modality must be provided.")

        query_states, cross_attentions = self.qformer(
            batch_size=batch_size,
            text_states=text_states,
            visual_states=visual_states,
            text_mask=text_mask,
            output_attentions=output_attentions
        )
        pooled_queries = query_states.mean(dim=1)
        if self.fusion_mode == "multimodal":
            logits = self.multimodal_head(pooled_queries)
        elif self.fusion_mode == "vision":
            logits = self.vision_head(pooled_queries)
        else:
            logits = self.text_head(pooled_queries)

        return {
            "logits": logits,
            "query_states": query_states,
            "cross_attentions": cross_attentions,
            "visual_states": visual_states,
            "text_states": text_states
        }
