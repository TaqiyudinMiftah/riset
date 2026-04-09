"""Q-Former modules inspired by InstructBLIP."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class QFormerLayer(nn.Module):
    """One Q-Former block with text-conditioned self-attention and visual cross-attention."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(hidden_size=hidden_size, dropout=dropout)

    def forward(
        self,
        query_states: torch.Tensor,
        text_states: torch.Tensor | None,
        visual_states: torch.Tensor | None,
        text_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
        output_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized_queries = self.query_norm(query_states)
        if text_states is not None:
            normalized_text = self.text_norm(text_states)
            key_value_states = torch.cat([normalized_queries, normalized_text], dim=1)
            batch_size = query_states.size(0)
            query_mask = torch.ones(batch_size, query_states.size(1), dtype=torch.bool, device=query_states.device)
            if text_mask is None:
                text_mask = torch.ones(
                    batch_size,
                    text_states.size(1),
                    dtype=torch.bool,
                    device=query_states.device
                )
            key_padding_mask = ~torch.cat([query_mask, text_mask.bool()], dim=1)
        else:
            key_value_states = normalized_queries
            key_padding_mask = None

        self_attn_output, _ = self.self_attention(
            query=normalized_queries,
            key=key_value_states,
            value=key_value_states,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        query_states = query_states + self.dropout(self_attn_output)

        cross_attn_weights: torch.Tensor | None = None
        if visual_states is not None:
            normalized_visual = self.visual_norm(visual_states)
            cross_output, cross_attn_weights = self.cross_attention(
                query=self.cross_norm(query_states),
                key=normalized_visual,
                value=normalized_visual,
                key_padding_mask=None if visual_mask is None else ~visual_mask.bool(),
                need_weights=output_attentions,
                average_attn_weights=False
            )
            query_states = query_states + self.dropout(cross_output)

        query_states = query_states + self.feed_forward(self.ffn_norm(query_states))
        return query_states, cross_attn_weights


class QFormerEncoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, hidden_size))
        self.layers = nn.ModuleList(
            [
                QFormerLayer(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(
        self,
        batch_size: int,
        text_states: torch.Tensor | None,
        visual_states: torch.Tensor | None,
        text_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
        output_attentions: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        query_states = self.query_tokens.expand(batch_size, -1, -1)
        cross_attentions: list[torch.Tensor] = []
        for layer in self.layers:
            query_states, cross_attn = layer(
                query_states=query_states,
                text_states=text_states,
                visual_states=visual_states,
                text_mask=text_mask,
                visual_mask=visual_mask,
                output_attentions=output_attentions
            )
            if cross_attn is not None:
                cross_attentions.append(cross_attn)
        return query_states, cross_attentions
