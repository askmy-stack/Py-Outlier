"""Diffusion-style anomaly detector via MLP reconstruction error (stub)."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from anomaly_detection.models.base import BaseDetector


class DiffusionDetector(BaseDetector):
    """Score anomalies using reconstruction error from a shallow MLP denoiser stub.

    This is a lightweight stand-in for diffusion-based reconstruction scoring that
    avoids GPU dependencies. Higher scores indicate larger reconstruction error.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        hidden_layer_sizes: tuple[int, ...] = (16, 8, 16),
        noise_scale: float = 0.1,
        max_iter: int = 500,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.contamination = contamination
        self.hidden_layer_sizes = hidden_layer_sizes
        self.noise_scale = noise_scale
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_kwargs = kwargs
        self.scaler_: StandardScaler | None = None
        self.model_: MLPRegressor | None = None
        self.threshold_: float | None = None
        self._rng = np.random.default_rng(random_state)

    def _add_noise(self, X: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0.0, self.noise_scale, size=X.shape)
        return X + noise

    def fit(self, X: np.ndarray) -> DiffusionDetector:
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        noisy = self._add_noise(X_scaled)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.model_kwargs,
        )
        self.model_.fit(noisy, X_scaled)
        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.model_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        noisy = self._add_noise(X_scaled)
        reconstructions = self.model_.predict(noisy)
        return np.mean((X_scaled - reconstructions) ** 2, axis=1)
