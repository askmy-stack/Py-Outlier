"""Isolation Forest anomaly detector."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from anomaly_detection.models.base import BaseDetector


class IsolationForestDetector(BaseDetector):
    """Wrap scikit-learn IsolationForest with the unified detector API."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42, **kwargs) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.model_kwargs = kwargs
        self.model_: IsolationForest | None = None
        self.threshold_: float | None = None

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        X = np.asarray(X, dtype=float)
        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self.model_kwargs,
        )
        self.model_.fit(X)
        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")
        # sklearn score_samples: lower values are more anomalous.
        return -self.model_.score_samples(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted before calling predict().")
        predictions = self.model_.predict(np.asarray(X, dtype=float))
        return (predictions == -1).astype(int)
