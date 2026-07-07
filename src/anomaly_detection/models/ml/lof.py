"""Local Outlier Factor anomaly detector."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from anomaly_detection.models.base import BaseDetector


class LOFDetector(BaseDetector):
    """Wrap scikit-learn LocalOutlierFactor with the unified detector API."""

    def __init__(self, contamination: float = 0.05, n_neighbors: int = 20, **kwargs) -> None:
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.model_kwargs = kwargs
        self.model_: LocalOutlierFactor | None = None
        self.threshold_: float | None = None

    def fit(self, X: np.ndarray) -> LOFDetector:
        X = np.asarray(X, dtype=float)
        self.model_ = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            novelty=True,
            **self.model_kwargs,
        )
        self.model_.fit(X)
        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")
        # decision_function: higher is more normal; invert for anomaly score.
        return -self.model_.decision_function(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted before calling predict().")
        predictions = self.model_.predict(np.asarray(X, dtype=float))
        return (predictions == -1).astype(int)
