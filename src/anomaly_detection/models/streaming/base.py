"""Base streaming detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseStreamingDetector(ABC):
    """Online anomaly detector API for point-by-point scoring."""

    @abstractmethod
    def fit_partial(self, x: np.ndarray) -> None:
        """Update detector state with a single observation."""

    @abstractmethod
    def score_one(self, x: np.ndarray) -> float:
        """Return anomaly score for a single observation; higher is more anomalous."""
