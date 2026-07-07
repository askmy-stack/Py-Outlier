"""Rolling z-score streaming anomaly detector."""

from __future__ import annotations

import numpy as np

from anomaly_detection.models.streaming.base import BaseStreamingDetector
from anomaly_detection.preprocessing.window import WindowPreprocessor


class ZScoreWindowDetector(BaseStreamingDetector):
    """Flag points whose z-score relative to a sliding window exceeds typical range."""

    def __init__(self, max_window: int = 100, min_window: int = 2) -> None:
        self.max_window = max_window
        self.min_window = min_window
        self._window = WindowPreprocessor(max_window=max_window)
        self._n_seen = 0

    @property
    def window_size(self) -> int:
        return self._window.size

    def fit_partial(self, x: np.ndarray) -> None:
        self._window.update(x)
        self._n_seen += 1

    def score_one(self, x: np.ndarray) -> float:
        if self._window.size < self.min_window:
            return 0.0
        z_scores = self._window.rolling_zscore(x)
        if z_scores is None:
            return 0.0
        return float(np.max(np.abs(z_scores)))
