"""Tests for detector performance profiler."""

from __future__ import annotations

import numpy as np

from anomaly_detection.evaluation.profiler import profile_detector


def test_profile_detector_returns_timing_keys() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))

    def fit_predict_fn(data: np.ndarray) -> None:
        _ = data.mean(axis=0)

    result = profile_detector("smoke", X, fit_predict_fn)

    assert result["name"] == "smoke"
    assert result["n_samples"] == 50
    assert "wall_time_sec" in result
    assert result["wall_time_sec"] >= 0.0
    assert "peak_memory_mb" in result
