"""Optional PySAD streaming detector wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from anomaly_detection.models.streaming.base import BaseStreamingDetector

PYSAD_AVAILABLE = False
_PYSAD_IMPORT_ERROR: Exception | None = None

try:
    from pysad.models import HalfSpaceTrees

    PYSAD_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised when pysad not installed
    HalfSpaceTrees = None  # type: ignore[misc, assignment]
    _PYSAD_IMPORT_ERROR = exc


def require_pysad() -> None:
    """Raise ImportError with install hint when PySAD is unavailable."""
    if not PYSAD_AVAILABLE:
        message = (
            "PySAD is not installed. Install with: pip install anomaly-detection[streaming]"
        )
        if _PYSAD_IMPORT_ERROR is not None:
            raise ImportError(message) from _PYSAD_IMPORT_ERROR
        raise ImportError(message)


class PySADStreamingDetector(BaseStreamingDetector):
    """Wrap a PySAD streaming estimator behind BaseStreamingDetector."""

    def __init__(
        self,
        estimator: str = "half_space_trees",
        params: dict[str, Any] | None = None,
    ) -> None:
        require_pysad()
        self.estimator_name = estimator
        self.params = params or {}
        self._model = self._create_estimator(estimator, self.params)

    def _create_estimator(self, name: str, params: dict[str, Any]) -> Any:
        if name == "half_space_trees":
            return HalfSpaceTrees(**params)
        raise ValueError(f"Unsupported PySAD estimator: {name}")

    def fit_partial(self, x: np.ndarray) -> None:
        self._model.fit_partial(np.asarray(x, dtype=float).reshape(1, -1))

    def score_one(self, x: np.ndarray) -> float:
        return float(self._model.score_one(np.asarray(x, dtype=float).reshape(1, -1)))
