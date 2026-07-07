"""Bounded sliding-window statistics for streaming preprocessing."""

from __future__ import annotations

from collections import deque

import numpy as np


class WindowPreprocessor:
    """Rolling mean, variance, and z-score over a fixed-size window."""

    def __init__(self, max_window: int) -> None:
        if max_window < 1:
            raise ValueError(f"max_window must be >= 1, got {max_window}")
        self.max_window = max_window
        self._buffer: deque[np.ndarray] = deque(maxlen=max_window)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def update(self, x: np.ndarray) -> None:
        """Append a single observation; oldest points are dropped at max_window."""
        self._buffer.append(np.asarray(x, dtype=float).reshape(-1))

    def _stacked(self) -> np.ndarray | None:
        if not self._buffer:
            return None
        return np.stack(list(self._buffer), axis=0)

    def rolling_mean(self) -> np.ndarray | None:
        stacked = self._stacked()
        if stacked is None:
            return None
        return np.mean(stacked, axis=0)

    def rolling_variance(self, ddof: int = 1) -> np.ndarray | None:
        if len(self._buffer) < 2:
            return None
        stacked = self._stacked()
        assert stacked is not None
        return np.var(stacked, axis=0, ddof=ddof)

    def rolling_zscore(self, x: np.ndarray) -> np.ndarray | None:
        """Z-score of x relative to the current window (excluding x)."""
        mean = self.rolling_mean()
        variance = self.rolling_variance()
        if mean is None or variance is None:
            return None
        point = np.asarray(x, dtype=float).reshape(-1)
        std = np.sqrt(variance)
        std[std == 0] = 1.0
        return (point - mean) / std
