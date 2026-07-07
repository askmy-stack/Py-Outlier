"""Streaming detector factory."""

from __future__ import annotations

from typing import Any

from anomaly_detection.models.streaming.base import BaseStreamingDetector
from anomaly_detection.models.streaming.zscore_window import ZScoreWindowDetector

STREAMING_REGISTRY: dict[str, type[BaseStreamingDetector]] = {
    "zscore_window": ZScoreWindowDetector,
}


def get_streaming_detector(name: str, params: dict[str, Any] | None = None) -> BaseStreamingDetector:
    """Instantiate a streaming detector by registry name."""
    if name not in STREAMING_REGISTRY:
        available = ", ".join(sorted(STREAMING_REGISTRY))
        raise KeyError(f"Unknown streaming detector '{name}'. Available: {available}")
    return STREAMING_REGISTRY[name](**(params or {}))


def create_streaming_detector_from_config(config: dict[str, Any]) -> BaseStreamingDetector:
    """Create a streaming detector from a loaded YAML config."""
    model_config = config.get("model", {})
    name = model_config.get("name")
    if not name:
        raise ValueError("Config must include model.name")
    params = dict(model_config.get("params", {}))

    stream_config = config.get("stream", {})
    max_window = stream_config.get("max_window")
    if max_window is not None and "max_window" not in params:
        params["max_window"] = max_window

    if name == "pysad":
        from anomaly_detection.models.streaming.pysad_wrapper import PySADStreamingDetector

        return PySADStreamingDetector(
            estimator=params.pop("estimator", "half_space_trees"),
            params=params,
        )

    return get_streaming_detector(name, params)
