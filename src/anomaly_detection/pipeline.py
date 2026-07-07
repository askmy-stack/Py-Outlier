"""Shared anomaly detection pipeline for CLI and API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.config.loader import load_config
from anomaly_detection.data_ingestion.loader import (
    _load_frame_from_config,
    load_dataset_from_config,
    load_protected_attributes,
)
from anomaly_detection.evaluation.metrics import compute_metrics, metrics_to_dict
from anomaly_detection.fairness.metrics import compute_fairness_metrics
from anomaly_detection.fairness.mitigation import apply_mitigation
from anomaly_detection.models.registry import create_detector_from_config
from anomaly_detection.preprocessing.scaler import apply_scaler


def run_detection(
    config: dict[str, Any],
    *,
    data: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Fit a detector and return predictions, scores, and optional metrics."""
    dataset_config = config.get("dataset", {})
    fairness_config = config.get("fairness", {})
    protected_attribute_names: list[str] = list(
        fairness_config.get("protected_attributes") or []
    )
    mitigation = fairness_config.get("mitigation")
    protected_attributes: dict[str, np.ndarray] = {}

    if data is None:
        if not dataset_config.get("path") and not dataset_config.get("id"):
            raise ValueError(
                "Config must include dataset.path or dataset.id when data is not provided"
            )
        if protected_attribute_names:
            frame = _load_frame_from_config(dataset_config, quick=quick)
            protected_attributes = load_protected_attributes(
                frame,
                protected_attribute_names,
            )
        features, loaded_labels, loaded_feature_names = load_dataset_from_config(
            dataset_config,
            quick=quick,
            exclude_columns=protected_attribute_names or None,
        )
        if labels is None:
            labels = loaded_labels
        feature_names = loaded_feature_names
    else:
        features = np.asarray(data, dtype=float)
        if feature_names is None:
            feature_names = [f"feature_{index}" for index in range(features.shape[1])]

    preprocessing_config = config.get("preprocessing", {})
    scaler_name = preprocessing_config.get("scaler")
    scaled_features = apply_scaler(features, scaler_name)

    detector = create_detector_from_config(config)
    detector.fit(scaled_features)
    scores = detector.score(scaled_features)
    predictions = detector.predict(scaled_features)

    model_config = config.get("model", {})
    report: dict[str, Any] = {
        "model": model_config.get("name"),
        "n_samples": int(len(features)),
        "n_anomalies": int(predictions.sum()),
        "feature_names": feature_names,
        "scores": scores.tolist(),
        "predictions": predictions.astype(int).tolist(),
    }

    if labels is not None:
        metrics = compute_metrics(labels, predictions, scores)
        report["metrics"] = metrics_to_dict(metrics)

    if protected_attributes:
        working_predictions = predictions
        if mitigation:
            mitigation_result = apply_mitigation(
                mitigation,
                scores=scores,
                predictions=predictions,
                protected_attributes=protected_attributes,
                labels=labels,
            )
            working_predictions = np.asarray(mitigation_result["predictions"], dtype=int)
            report["mitigation"] = {
                "strategy": mitigation_result["mitigation"],
                "primary_attribute": mitigation_result.get("primary_attribute"),
            }
            if mitigation_result.get("weights") is not None:
                report["mitigation"]["weights_sample"] = mitigation_result["weights"][:5]

        report["fairness_metrics"] = compute_fairness_metrics(
            working_predictions,
            protected_attributes,
            labels=labels,
        )

    return report


def load_config_and_run(
    config_path: str | Path,
    *,
    data: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML config, apply optional overrides, and run detection."""
    config = load_config(config_path)
    if config_override:
        config = _deep_merge(config, config_override)
    return run_detection(config, data=data, labels=labels)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
