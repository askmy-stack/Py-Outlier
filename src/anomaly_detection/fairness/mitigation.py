"""Fairness mitigation helpers for anomaly detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from anomaly_detection.fairness.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)


def compute_reweighing_weights(
    protected_attribute: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute sample weights that balance joint (label, group) counts.

    Uses the standard reweighing formula: w = P(A) * P(Y) / P(A, Y).
    """
    protected = np.asarray(protected_attribute)
    y = (np.asarray(labels).astype(int).ravel() > 0).astype(int)
    n = len(y)
    if n == 0:
        return np.array([], dtype=float)

    weights = np.empty(n, dtype=float)
    groups = np.unique(protected)
    label_values = np.unique(y)
    for group in groups:
        group_mask = protected == group
        p_a = group_mask.mean()
        for label in label_values:
            joint_mask = group_mask & (y == label)
            p_y = (y == label).mean()
            p_ay = joint_mask.mean()
            cell_weight = 0.0 if p_ay == 0 else (p_a * p_y) / p_ay
            weights[joint_mask] = cell_weight

    if weights.sum() > 0:
        weights *= n / weights.sum()
    return weights


def threshold_post_processing(
    scores: np.ndarray,
    protected_attribute: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    target: str = "demographic_parity",
) -> np.ndarray:
    """Adjust per-group score thresholds to reduce disparity."""
    scores = np.asarray(scores, dtype=float).ravel()
    protected = np.asarray(protected_attribute)
    groups = np.unique(protected)
    if groups.size < 2:
        return (scores >= np.median(scores)).astype(int)

    best_predictions = (scores >= np.median(scores)).astype(int)
    if target == "equalized_odds" and labels is not None:
        best_metric = equalized_odds_difference(labels, best_predictions, protected)
    else:
        best_metric = demographic_parity_difference(best_predictions, protected)

    percentiles = np.linspace(50, 99, 25)
    for threshold_grid in np.array(np.meshgrid(*[percentiles] * groups.size)).T.reshape(-1, groups.size):
        predictions = np.zeros(len(scores), dtype=int)
        for group, pct in zip(groups, threshold_grid, strict=True):
            mask = protected == group
            group_scores = scores[mask]
            if group_scores.size == 0:
                continue
            cutoff = np.percentile(group_scores, pct)
            predictions[mask] = (group_scores >= cutoff).astype(int)

        if target == "equalized_odds" and labels is not None:
            metric = equalized_odds_difference(labels, predictions, protected)
        else:
            metric = demographic_parity_difference(predictions, protected)

        if metric < best_metric:
            best_metric = metric
            best_predictions = predictions.copy()

    return best_predictions


def apply_mitigation(
    mitigation: str | None,
    *,
    scores: np.ndarray,
    predictions: np.ndarray,
    protected_attributes: dict[str, np.ndarray],
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Apply configured mitigation and return adjusted outputs plus metadata."""
    if not mitigation:
        return {
            "mitigation": None,
            "predictions": np.asarray(predictions).astype(int),
            "weights": None,
        }

    primary_attr = next(iter(protected_attributes))
    protected = protected_attributes[primary_attr]

    if mitigation == "reweighing":
        if labels is None:
            raise ValueError("Reweighing mitigation requires ground-truth labels")
        weights = compute_reweighing_weights(protected, labels)
        adjusted = threshold_post_processing(
            scores,
            protected,
            labels=labels,
            target="demographic_parity",
        )
        return {
            "mitigation": "reweighing",
            "predictions": adjusted,
            "weights": weights.tolist(),
            "primary_attribute": primary_attr,
        }

    if mitigation == "threshold_post_processing":
        adjusted = threshold_post_processing(
            scores,
            protected,
            labels=labels,
            target="equalized_odds" if labels is not None else "demographic_parity",
        )
        return {
            "mitigation": "threshold_post_processing",
            "predictions": adjusted,
            "weights": None,
            "primary_attribute": primary_attr,
        }

    raise ValueError(
        f"Unknown mitigation '{mitigation}'. Supported: reweighing, threshold_post_processing"
    )
