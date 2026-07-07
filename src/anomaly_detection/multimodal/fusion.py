"""Experimental tabular + text multimodal fusion stub."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from anomaly_detection.models.base import BaseDetector


def _text_to_features(texts: list[str], n_components: int) -> np.ndarray:
    """Encode text rows into a fixed-size numeric vector via hashed token counts."""
    if not texts:
        raise ValueError("texts must not be empty")

    vocab_size = max(n_components, 8)
    vectors = np.zeros((len(texts), vocab_size), dtype=float)
    for row_index, text in enumerate(texts):
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            column = int(digest, 16) % vocab_size
            vectors[row_index, column] += 1.0
    return vectors


class MultimodalFusion(BaseDetector):
    """Fuse tabular features with hashed text embeddings in a shared latent space."""

    def __init__(
        self,
        contamination: float = 0.05,
        latent_dim: int = 4,
        text_weight: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.latent_dim = latent_dim
        self.text_weight = text_weight
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.threshold_: float | None = None
        self._train_latent: np.ndarray | None = None

    def _combine_features(self, tabular: np.ndarray, texts: list[str]) -> np.ndarray:
        tabular = np.asarray(tabular, dtype=float)
        text_features = _text_to_features(texts, self.latent_dim)
        if len(text_features) != len(tabular):
            raise ValueError("tabular rows and texts must have the same length")
        tabular_scaled = self.scaler_.transform(tabular)
        text_scaled = StandardScaler().fit_transform(text_features)
        alpha = self.text_weight
        return np.hstack([(1 - alpha) * tabular_scaled, alpha * text_scaled])

    def fit(self, X: np.ndarray, texts: list[str] | None = None) -> MultimodalFusion:
        if texts is None:
            raise ValueError("MultimodalFusion.fit requires a texts argument")
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        combined = self._combine_features(X, texts)
        n_components = min(self.latent_dim, combined.shape[1], combined.shape[0])
        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        self._train_latent = self.pca_.fit_transform(combined)
        distances = np.linalg.norm(self._train_latent, axis=1)
        self.threshold_ = float(np.percentile(distances, 100 * (1 - self.contamination)))
        return self

    def score(self, X: np.ndarray, texts: list[str] | None = None) -> np.ndarray:
        if self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("Detector must be fitted before calling score().")
        if texts is None:
            raise ValueError("MultimodalFusion.score requires a texts argument")
        combined = self._combine_features(np.asarray(X, dtype=float), texts)
        latent = self.pca_.transform(combined)
        return np.linalg.norm(latent, axis=1)

    def predict(self, X: np.ndarray, texts: list[str] | None = None) -> np.ndarray:
        scores = self.score(X, texts=texts)
        if self.threshold_ is None:
            raise RuntimeError("Detector must be fitted before calling predict().")
        return (scores > self.threshold_).astype(int)
