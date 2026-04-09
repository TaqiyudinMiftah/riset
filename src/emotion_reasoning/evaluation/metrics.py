"""Research metrics for emotion recognition."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import torch


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=1, keepdims=True)


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    class_names: list[str]
) -> dict[str, Any]:
    scores = logits.detach().cpu().numpy()
    task_type = task_type.lower()
    if task_type == "multilabel":
        probabilities = _sigmoid(scores)
        targets = labels.detach().cpu().numpy().astype(np.int32)
        predictions = (probabilities >= 0.5).astype(np.int32)
        accuracy = float((predictions == targets).mean())
    else:
        probabilities = _softmax(scores)
        label_indices = labels.detach().cpu().numpy().astype(np.int32)
        targets = np.eye(len(class_names), dtype=np.int32)[label_indices]
        predictions = probabilities.argmax(axis=1)
        accuracy = float(accuracy_score(label_indices, predictions))

    per_class_ap: dict[str, float] = {}
    per_class_auc: dict[str, float] = {}
    ap_values: list[float] = []
    auc_values: list[float] = []

    for class_index, class_name in enumerate(class_names):
        y_true = targets[:, class_index]
        y_score = probabilities[:, class_index]
        if int(y_true.sum()) > 0:
            ap = float(average_precision_score(y_true, y_score))
            per_class_ap[class_name] = ap
            ap_values.append(ap)
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_score))
            per_class_auc[class_name] = auc
            auc_values.append(auc)

    return {
        "mAP": float(np.mean(ap_values)) if ap_values else 0.0,
        "auc_roc": float(np.mean(auc_values)) if auc_values else 0.0,
        "accuracy": accuracy,
        "per_class_ap": per_class_ap,
        "per_class_auc": per_class_auc
    }
