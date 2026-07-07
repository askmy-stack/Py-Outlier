"""Tests for fairness mitigation helpers."""

from __future__ import annotations

import numpy as np

from anomaly_detection.fairness.metrics import demographic_parity_difference
from anomaly_detection.fairness.mitigation import (
    apply_mitigation,
    compute_reweighing_weights,
    threshold_post_processing,
)


def _biased_fixture(seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 160
    protected = rng.integers(0, 2, size=n)
    labels = rng.integers(0, 2, size=n)
    scores = rng.normal(0, 1, size=n)
    scores[protected == 1] += 1.8
    predictions = (scores >= np.median(scores)).astype(int)
    return scores, predictions, protected, labels


def test_reweighing_weights_balance_joint_counts() -> None:
    protected = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([0, 0, 1, 0, 1, 1])
    weights = compute_reweighing_weights(protected, labels)
    assert weights.shape == labels.shape
    assert np.all(weights > 0)
    assert np.isclose(weights.sum(), len(labels), rtol=1e-6)


def test_threshold_post_processing_reduces_disparity() -> None:
    scores, predictions, protected, labels = _biased_fixture()
    before = demographic_parity_difference(predictions, protected)
    adjusted = threshold_post_processing(scores, protected, labels=labels)
    after = demographic_parity_difference(adjusted, protected)
    assert after <= before


def test_apply_mitigation_reweighing_reduces_disparity() -> None:
    scores, predictions, protected, labels = _biased_fixture()
    before = demographic_parity_difference(predictions, protected)
    result = apply_mitigation(
        "reweighing",
        scores=scores,
        predictions=predictions,
        protected_attributes={"gender": protected},
        labels=labels,
    )
    after = demographic_parity_difference(result["predictions"], protected)
    assert result["mitigation"] == "reweighing"
    assert result["weights"] is not None
    assert after <= before
