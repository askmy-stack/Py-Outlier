"""Lightweight autoencoder anomaly detector using sklearn MLP."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from anomaly_detection.models.base import BaseDetector


class AutoencoderDetector(BaseDetector):
    """Detect anomalies via reconstruction error from a shallow MLP autoencoder."""

    def __init__(
        self,
        contamination: float = 0.05,
        hidden_layer_sizes: tuple[int, ...] = (8, 4, 8),
        max_iter: int = 500,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.contamination = contamination
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_kwargs = kwargs
        self.scaler_: StandardScaler | None = None
        self.model_: MLPRegressor | None = None
        self.threshold_: float | None = None

    def fit(self, X: np.ndarray) -> AutoencoderDetector:
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.model_kwargs,
        )
        self.model_.fit(X_scaled, X_scaled)
        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.model_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        reconstructions = self.model_.predict(X_scaled)
        return np.mean((X_scaled - reconstructions) ** 2, axis=1)
