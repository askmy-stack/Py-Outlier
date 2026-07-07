"""Tests for fairness metrics."""

from __future__ import annotations

import numpy as np

from anomaly_detection.fairness.metrics import (
    compute_fairness_metrics,
    demographic_parity_difference,
    equalized_odds_difference,
)


def _biased_predictions(n: int = 200, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    protected = rng.integers(0, 2, size=n)
    labels = rng.integers(0, 2, size=n)
    predictions = np.zeros(n, dtype=int)
    for idx in range(n):
        base_rate = 0.55 if protected[idx] == 1 else 0.15
        predictions[idx] = int(rng.random() < base_rate)
    return labels, predictions, protected


def test_demographic_parity_difference_detects_bias() -> None:
    _, predictions, protected = _biased_predictions()
    dpd = demographic_parity_difference(predictions, protected)
    assert dpd > 0.2


def test_demographic_parity_difference_balanced_groups() -> None:
    predictions = np.array([0, 1, 0, 1])
    protected = np.array([0, 0, 1, 1])
    dpd = demographic_parity_difference(predictions, protected)
    assert dpd == 0.0


def test_equalized_odds_difference_with_labels() -> None:
    labels, predictions, protected = _biased_predictions()
    eod = equalized_odds_difference(labels, predictions, protected)
    assert eod >= 0.0
    assert eod > 0.1


def test_compute_fairness_metrics_multiple_attributes() -> None:
    labels, predictions, protected = _biased_predictions()
    age_group = (protected + np.arange(len(protected))) % 2
    report = compute_fairness_metrics(
        predictions,
        {
            "gender": protected,
            "age_group": age_group,
        },
        labels=labels,
    )
    assert report["backend"] in {"numpy", "aif360"}
    assert "gender" in report["attributes"]
    assert "age_group" in report["attributes"]
    assert report["attributes"]["gender"]["demographic_parity_difference"] > 0.2
    assert report["attributes"]["gender"]["equalized_odds_difference"] is not None
