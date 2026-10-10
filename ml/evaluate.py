from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_k_metrics(y_true, scores, capacity: int = 200) -> dict[str, Any]:
    labels = np.asarray(y_true)
    values = np.asarray(scores)
    k = min(capacity, len(labels))
    if k == 0:
        return {"k": 0, "true_positives": 0, "precision_at_k": None, "recall_at_k": None}
    order = np.argsort(-values, kind="stable")[:k]
    true_positives = int(labels[order].sum())
    total_positives = int(labels.sum())
    return {
        "k": k,
        "true_positives": true_positives,
        "precision_at_k": true_positives / k,
        "recall_at_k": true_positives / total_positives if total_positives else None,
    }


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict[str, Any]:
    labels = np.asarray(y_true)
    predictions = (np.asarray(scores) >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix,
        **top_k_metrics(labels, scores),
    }


def dummy_metrics(y_true) -> dict[str, Any]:
    labels = np.asarray(y_true)
    predictions = np.zeros_like(labels)
    return {
        "accuracy": float((predictions == labels).mean()),
        "precision": 0.0,
        "recall": 0.0,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
