from __future__ import annotations

import torch

from emotion_reasoning.evaluation.metrics import compute_classification_metrics


def test_multilabel_metrics_return_expected_keys() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    metrics = compute_classification_metrics(
        logits=logits,
        labels=labels,
        task_type="multilabel",
        class_names=["Joy", "Sadness"]
    )
    assert "mAP" in metrics
    assert "auc_roc" in metrics
    assert "accuracy" in metrics
