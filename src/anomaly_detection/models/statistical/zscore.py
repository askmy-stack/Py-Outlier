"""Z-score based anomaly detector."""

from __future__ import annotations

import numpy as np

from anomaly_detection.models.base import BaseDetector


class ZScoreDetector(BaseDetector):
    """Flag samples whose per-feature z-scores exceed a learned threshold."""

    def __init__(self, contamination: float = 0.05) -> None:
        self.contamination = contamination
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, X: np.ndarray) -> ZScoreDetector:
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        train_scores = self._raw_scores(X)
        self.threshold_ = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")
        return self._raw_scores(np.asarray(X, dtype=float))

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        z_scores = np.abs((X - self.mean_) / self.std_)
        return np.max(z_scores, axis=1)
