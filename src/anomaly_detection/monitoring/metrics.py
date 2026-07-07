"""Streaming processing metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def _memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


@dataclass
class StreamMetrics:
    """Track processing rate, per-point latency, and optional memory usage."""

    track_memory: bool = False
    _count: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.perf_counter, init=False)
    _latencies: list[float] = field(default_factory=list, init=False)

    def record(self, latency_seconds: float) -> None:
        self._count += 1
        self._latencies.append(latency_seconds)

    @property
    def count(self) -> int:
        return self._count

    @property
    def processing_rate(self) -> float:
        elapsed = time.perf_counter() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._count / elapsed

    def latency_percentile(self, percentile: float) -> float | None:
        if not self._latencies:
            return None
        return float(np_percentile(self._latencies, percentile))

    def memory_mb(self) -> float | None:
        if not self.track_memory:
            return None
        return _memory_mb()

    def summary(self) -> dict[str, float | int | None]:
        return {
            "count": self._count,
            "processing_rate": self.processing_rate,
            "latency_p50_ms": _ms(self.latency_percentile(50)),
            "latency_p99_ms": _ms(self.latency_percentile(99)),
            "memory_mb": self.memory_mb(),
        }


def _ms(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return seconds * 1000.0


def np_percentile(values: list[float], percentile: float) -> float:
    import numpy as np

    return float(np.percentile(values, percentile))
