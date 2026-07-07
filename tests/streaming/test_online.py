"""Online streaming anomaly detection tests."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from anomaly_detection.models.streaming.zscore_window import ZScoreWindowDetector
from anomaly_detection.monitoring.metrics import StreamMetrics
from anomaly_detection.preprocessing.window import WindowPreprocessor

REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_CONFIG = REPO_ROOT / "configs" / "examples" / "streaming.yaml"
MAX_WINDOW = 50


def _stream_points(
    detector: ZScoreWindowDetector,
    points: np.ndarray,
    metrics: StreamMetrics | None = None,
) -> list[float]:
    scores: list[float] = []
    for point in points:
        start = time.perf_counter()
        score = detector.score_one(point)
        detector.fit_partial(point)
        if metrics is not None:
            metrics.record(time.perf_counter() - start)
        scores.append(score)
        assert detector.window_size <= MAX_WINDOW
    return scores


def test_stream_10k_points_bounded_memory() -> None:
    rng = np.random.default_rng(42)
    points = rng.normal(loc=0.0, scale=0.5, size=(10_000, 3))
    detector = ZScoreWindowDetector(max_window=MAX_WINDOW, min_window=5)
    scores = _stream_points(detector, points)
    assert len(scores) == 10_000
    assert detector.window_size == MAX_WINDOW


def test_window_preprocessor_does_not_grow_beyond_max() -> None:
    window = WindowPreprocessor(max_window=MAX_WINDOW)
    for index in range(10_000):
        window.update(np.array([float(index), float(index) + 1.0]))
    assert window.size == MAX_WINDOW


def test_zscore_window_detects_injected_spike() -> None:
    rng = np.random.default_rng(7)
    normal = rng.normal(loc=0.0, scale=0.2, size=(200, 2))
    spike_index = 150
    points = normal.copy()
    points[spike_index] = np.array([20.0, 20.0])

    detector = ZScoreWindowDetector(max_window=MAX_WINDOW, min_window=10)
    scores = _stream_points(detector, points)

    baseline = np.median(scores[:spike_index])
    spike_score = scores[spike_index]
    assert spike_score > baseline + 2.0
    assert spike_score > 3.0


def test_latency_p99_documented_not_enforced() -> None:
    """Document latency on current hardware; do not fail CI on slow machines."""
    rng = np.random.default_rng(99)
    points = rng.normal(size=(2_000, 3))
    detector = ZScoreWindowDetector(max_window=MAX_WINDOW, min_window=5)
    metrics = StreamMetrics()
    _stream_points(detector, points, metrics=metrics)

    p99_seconds = metrics.latency_percentile(99)
    assert p99_seconds is not None
    p99_ms = p99_seconds * 1000
    hardware = f"{platform.system()} {platform.machine()} Python {sys.version.split()[0]}"
    if p99_ms > 10.0:
        pytest.skip(f"p99 latency {p99_ms:.2f}ms > 10ms on {hardware} (documented only)")


def test_cli_stream_runs_on_sample_csv() -> None:
    assert STREAMING_CONFIG.exists()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "anomaly_detection.cli.stream",
            "--config",
            str(STREAMING_CONFIG),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines
    last_line = json.loads(lines[-1])
    assert "summary" in last_line
    assert last_line["summary"]["n_points"] > 0
