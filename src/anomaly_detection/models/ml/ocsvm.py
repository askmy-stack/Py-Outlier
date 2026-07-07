"""One-Class SVM anomaly detector."""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from anomaly_detection.models.base import BaseDetector


class OneClassSVMDetector(BaseDetector):
    """Wrap scikit-learn OneClassSVM with the unified detector API."""

    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale", **kwargs) -> None:
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model_kwargs = kwargs
        self.model_: OneClassSVM | None = None
        self.threshold_: float | None = None
        self.contamination = nu

    def fit(self, X: np.ndarray) -> OneClassSVMDetector:
        X = np.asarray(X, dtype=float)
        self.model_ = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            **self.model_kwargs,
        )
        self.model_.fit(X)
        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")
        return -self.model_.decision_function(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted before calling predict().")
        predictions = self.model_.predict(np.asarray(X, dtype=float))
        return (predictions == -1).astype(int)
