"""Tests for multimodal tabular+text fusion."""

from __future__ import annotations

import numpy as np

from anomaly_detection.multimodal.fusion import MultimodalFusion

RANDOM_STATE = 42


def _make_paired_fixture() -> tuple[np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    normal_tabular = rng.normal(0.0, 0.3, size=(40, 3))
    anomaly_tabular = rng.normal(5.0, 0.4, size=(10, 3))
    tabular = np.vstack([normal_tabular, anomaly_tabular])

    texts = ["normal system metrics within range"] * 40 + ["critical failure spike detected"] * 10
    labels = np.array([0] * 40 + [1] * 10)
    indices = rng.permutation(len(labels))
    return tabular[indices], [texts[i] for i in indices], labels[indices]


def test_multimodal_fusion_fit_score_predict() -> None:
    tabular, texts, _ = _make_paired_fixture()
    fusion = MultimodalFusion(contamination=0.1, latent_dim=3, random_state=RANDOM_STATE)
    fusion.fit(tabular, texts=texts)

    scores = fusion.score(tabular, texts=texts)
    predictions = fusion.predict(tabular, texts=texts)

    assert scores.shape == (len(tabular),)
    assert predictions.shape == (len(tabular),)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert int(predictions.sum()) >= 1


def test_multimodal_fusion_requires_matching_lengths() -> None:
    tabular, texts, _ = _make_paired_fixture()
    fusion = MultimodalFusion(contamination=0.1, latent_dim=3, random_state=RANDOM_STATE)

    try:
        fusion.fit(tabular[:5], texts=texts)
        raised = False
    except ValueError:
        raised = True
    assert raised
