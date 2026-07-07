"""Performance profiler for anomaly detectors."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_PSUTIL = False


def profile_detector(
    name: str,
    X: np.ndarray,
    fit_predict_fn: Callable[[np.ndarray], Any],
) -> dict[str, Any]:
    """Profile wall time and peak memory for a detector fit+predict cycle.

    Parameters
    ----------
    name:
        Detector identifier (for reporting).
    X:
        Feature matrix passed to ``fit_predict_fn``.
    fit_predict_fn:
        Callable that fits and predicts on ``X`` (e.g. detector.fit + predict).

    Returns
    -------
    dict
        Keys: ``name``, ``n_samples``, ``wall_time_sec``, ``peak_memory_mb``
        (``peak_memory_mb`` is ``None`` when psutil is unavailable).
    """
    result: dict[str, Any] = {
        "name": name,
        "n_samples": int(len(X)),
    }

    mem_before = 0
    if _HAS_PSUTIL:
        process = psutil.Process()
        mem_before = process.memory_info().rss

    start = time.perf_counter()
    fit_predict_fn(X)
    wall_time = time.perf_counter() - start

    result["wall_time_sec"] = wall_time

    if _HAS_PSUTIL:
        mem_after = psutil.Process().memory_info().rss
        peak_bytes = max(mem_before, mem_after)
        result["peak_memory_mb"] = peak_bytes / (1024 * 1024)
    else:
        result["peak_memory_mb"] = None

    return result
