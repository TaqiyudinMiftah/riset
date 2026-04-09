from __future__ import annotations

import torch

from emotion_reasoning.modeling.qformer import QFormerEncoder


def test_qformer_output_shape() -> None:
    encoder = QFormerEncoder(num_queries=32, hidden_size=64, num_layers=2, num_heads=8, dropout=0.1)
    text_states = torch.randn(4, 10, 64)
    visual_states = torch.randn(4, 17, 64)
    query_states, cross_attentions = encoder(
        batch_size=4,
        text_states=text_states,
        visual_states=visual_states,
        text_mask=torch.ones(4, 10, dtype=torch.bool),
        output_attentions=True
    )
    assert query_states.shape == (4, 32, 64)
    assert len(cross_attentions) == 2
