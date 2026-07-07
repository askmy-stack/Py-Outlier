"""Tests for generative anomaly detectors."""

from __future__ import annotations

import numpy as np

from anomaly_detection.models.generative.diffusion_detector import DiffusionDetector
from anomaly_detection.models.generative.gan_augmentation import GANAugmentation
from anomaly_detection.models.registry import get_detector

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


def test_diffusion_detector_round_trip() -> None:
    X, _ = _make_synthetic_data()
    detector = DiffusionDetector(
        contamination=CONTAMINATION,
        hidden_layer_sizes=(6, 3, 6),
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    detector.fit(X)
    scores = detector.score(X)
    predictions = detector.predict(X)

    assert scores.shape == (len(X),)
    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert int(predictions.sum()) >= 1


def test_diffusion_detector_scores_outliers_higher() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    normal = rng.normal(0.0, 0.2, size=(80, 3))
    noisy = rng.normal(6.0, 0.5, size=(20, 3))
    X = np.vstack([normal, noisy])

    detector = DiffusionDetector(
        contamination=0.1,
        hidden_layer_sizes=(6, 3, 6),
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    detector.fit(normal)
    scores = detector.score(X)
    assert float(scores[-5:].mean()) > float(scores[:5].mean())


def test_diffusion_registered_in_registry() -> None:
    detector = get_detector(
        "diffusion",
        {
            "contamination": CONTAMINATION,
            "hidden_layer_sizes": (6, 3, 6),
            "max_iter": 200,
            "random_state": RANDOM_STATE,
        },
    )
    assert isinstance(detector, DiffusionDetector)


def test_gan_augmentation_appends_synthetic_anomalies() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(0.0, 1.0, size=(50, 3))
    labels = np.zeros(50, dtype=int)

    augmenter = GANAugmentation(n_synthetic=5, random_state=RANDOM_STATE)
    augmented_X, augmented_labels = augmenter.augment(X, labels=labels)

    assert augmented_X.shape == (55, 3)
    assert augmented_labels is not None
    assert augmented_labels.shape == (55,)
    assert int(augmented_labels[-5:].sum()) == 5
