"""Evaluation metrics for anomaly detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute binary classification metrics when ground-truth labels are available."""
    labels = np.asarray(y_true).astype(int)
    predictions = np.asarray(y_pred).astype(int)

    if labels.size == 0:
        return {"precision": None, "recall": None, "f1": None, "roc_auc": None}

    metrics: dict[str, float | None] = {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": None,
    }

    if y_scores is not None and len(np.unique(labels)) > 1:
        scores = np.asarray(y_scores, dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))

    return metrics


def metrics_to_dict(metrics: dict[str, float | None]) -> dict[str, Any]:
    """Serialize metrics, omitting keys with no value."""
    return {key: value for key, value in metrics.items() if value is not None}
