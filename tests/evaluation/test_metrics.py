"""Evaluation metrics tests."""

from __future__ import annotations

import numpy as np

from anomaly_detection.evaluation.metrics import compute_metrics


def test_compute_metrics_with_labels() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.4])

    metrics = compute_metrics(y_true, y_pred, y_scores)

    assert metrics["precision"] is not None
    assert metrics["recall"] is not None
    assert metrics["f1"] is not None
    assert metrics["roc_auc"] is not None
