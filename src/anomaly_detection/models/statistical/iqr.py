"""IQR based anomaly detector."""

from __future__ import annotations

import numpy as np

from anomaly_detection.models.base import BaseDetector


class IQRDetector(BaseDetector):
    """Flag samples outside the interquartile range on any feature."""

    def __init__(self, contamination: float = 0.05, iqr_multiplier: float = 1.5) -> None:
        self.contamination = contamination
        self.iqr_multiplier = iqr_multiplier
        self.q1_: np.ndarray | None = None
        self.q3_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, X: np.ndarray) -> IQRDetector:
        X = np.asarray(X, dtype=float)
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.iqr_[self.iqr_ == 0] = 1.0
        train_scores = self._raw_scores(X)
        self.threshold_ = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.q1_ is None or self.q3_ is None or self.iqr_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")
        return self._raw_scores(np.asarray(X, dtype=float))

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        lower = self.q1_ - self.iqr_multiplier * self.iqr_
        upper = self.q3_ + self.iqr_multiplier * self.iqr_
        below = np.maximum(lower - X, 0) / self.iqr_
        above = np.maximum(X - upper, 0) / self.iqr_
        return np.max(np.maximum(below, above), axis=1)
