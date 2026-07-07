"""Base detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Unified anomaly detector API aligned with scikit-learn conventions."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> BaseDetector:
        """Fit the detector on training data."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores; higher values indicate more anomalous samples."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels: 0 for normal, 1 for anomaly."""
        scores = self.score(X)
        threshold = getattr(self, "threshold_", None)
        if threshold is None:
            raise RuntimeError("Detector must be fitted before calling predict().")
        return (scores > threshold).astype(int)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the detector and return predictions for the training data."""
        self.fit(X)
        return self.predict(X)
