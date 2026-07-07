"""Detector factory registry."""

from __future__ import annotations

from typing import Any

from anomaly_detection.models.base import BaseDetector
from anomaly_detection.models.deep.autoencoder import AutoencoderDetector
from anomaly_detection.models.ml.isolation_forest import IsolationForestDetector
from anomaly_detection.models.ml.lof import LOFDetector
from anomaly_detection.models.ml.ocsvm import OneClassSVMDetector
from anomaly_detection.models.statistical.iqr import IQRDetector
from anomaly_detection.models.statistical.zscore import ZScoreDetector

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "zscore": ZScoreDetector,
    "iqr": IQRDetector,
    "isolation_forest": IsolationForestDetector,
    "lof": LOFDetector,
    "one_class_svm": OneClassSVMDetector,
    "autoencoder": AutoencoderDetector,
}


def get_detector(name: str, params: dict[str, Any] | None = None) -> BaseDetector:
    """Instantiate a detector by registry name."""
    if name not in DETECTOR_REGISTRY:
        available = ", ".join(sorted(DETECTOR_REGISTRY))
        raise KeyError(f"Unknown detector '{name}'. Available detectors: {available}")
    return DETECTOR_REGISTRY[name](**(params or {}))


def create_detector_from_config(config: dict[str, Any]) -> BaseDetector:
    """Create a detector from a loaded YAML config."""
    model_config = config.get("model", {})
    name = model_config.get("name")
    if not name:
        raise ValueError("Config must include model.name")
    params = model_config.get("params", {})
    return get_detector(name, params)
