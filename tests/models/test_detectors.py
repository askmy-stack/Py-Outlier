"""Unit tests for anomaly detectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.config.loader import load_config
from anomaly_detection.models.base import BaseDetector
from anomaly_detection.models.deep.autoencoder import AutoencoderDetector
from anomaly_detection.models.ml.isolation_forest import IsolationForestDetector
from anomaly_detection.models.ml.lof import LOFDetector
from anomaly_detection.models.ml.ocsvm import OneClassSVMDetector
from anomaly_detection.models.registry import create_detector_from_config, get_detector
from anomaly_detection.models.statistical.iqr import IQRDetector
from anomaly_detection.models.statistical.zscore import ZScoreDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
SAMPLE_DATA = REPO_ROOT / "data" / "sample.csv"
CONTAMINATION = 0.05
RANDOM_STATE = 42


def _make_synthetic_data(
    n_samples: int = 200,
    n_features: int = 3,
    contamination: float = CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_outliers = max(1, int(n_samples * contamination))
    n_normal = n_samples - n_outliers
    normal = rng.normal(loc=0.0, scale=0.5, size=(n_normal, n_features))
    outliers = rng.normal(loc=8.0, scale=0.2, size=(n_outliers, n_features))
    X = np.vstack([normal, outliers])
    y = np.array([0] * n_normal + [1] * n_outliers)
    indices = rng.permutation(len(y))
    return X[indices], y[indices]


def _assert_detector_round_trip(detector: BaseDetector, X: np.ndarray) -> None:
    fitted = detector.fit(X)
    assert fitted is detector

    scores = detector.score(X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()

    predictions = detector.predict(X)
    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)).issubset({0, 1})

    fit_predict = detector.fit_predict(X)
    np.testing.assert_array_equal(fit_predict, predictions)


@pytest.mark.parametrize(
    ("name", "params", "detector_cls"),
    [
        ("zscore", {"contamination": CONTAMINATION}, ZScoreDetector),
        ("iqr", {"contamination": CONTAMINATION}, IQRDetector),
        (
            "isolation_forest",
            {"contamination": CONTAMINATION, "random_state": RANDOM_STATE},
            IsolationForestDetector,
        ),
        ("lof", {"contamination": CONTAMINATION, "n_neighbors": 10}, LOFDetector),
        ("one_class_svm", {"nu": CONTAMINATION, "gamma": "scale"}, OneClassSVMDetector),
        (
            "autoencoder",
            {
                "contamination": CONTAMINATION,
                "hidden_layer_sizes": (6, 3, 6),
                "max_iter": 300,
                "random_state": RANDOM_STATE,
            },
            AutoencoderDetector,
        ),
    ],
)
def test_detector_round_trip(name: str, params: dict, detector_cls: type[BaseDetector]) -> None:
    X, y = _make_synthetic_data()
    detector = get_detector(name, params)
    assert isinstance(detector, detector_cls)
    _assert_detector_round_trip(detector, X)

    fitted = detector.fit(X)
    predictions = fitted.predict(X)
    detected = int(predictions.sum())
    assert detected >= 1  # At least some outliers should be flagged on synthetic data.
    assert detected <= len(y)


def test_sample_csv_fixture_exists_with_labels() -> None:
    assert SAMPLE_DATA.exists()
    frame = pd.read_csv(SAMPLE_DATA)
    assert {"feature_1", "feature_2", "feature_3", "label"}.issubset(frame.columns)
    assert frame["label"].isin([0, 1]).all()
    assert int(frame["label"].sum()) >= 1


def test_config_loads_isolation_forest_detector() -> None:
    config = load_config(DEFAULT_CONFIG)
    detector = create_detector_from_config(config)
    assert isinstance(detector, IsolationForestDetector)
    assert config["model"]["name"] == "isolation_forest"
    assert config["dataset"]["path"] == "data/sample.csv"


def test_config_can_switch_to_zscore(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config["model"] = {"name": "zscore", "params": {"contamination": CONTAMINATION}}
    alt_config = tmp_path / "zscore.yaml"
    import yaml

    alt_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(alt_config)
    detector = create_detector_from_config(loaded)
    assert isinstance(detector, ZScoreDetector)

    X, _ = _make_synthetic_data()
    _assert_detector_round_trip(detector, X)
