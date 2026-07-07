"""GAN-style synthetic anomaly augmentation stub using numpy noise."""

from __future__ import annotations

import numpy as np


class GANAugmentation:
    """Generate synthetic rare anomalies by perturbing normal samples with noise.

    This is a lightweight stand-in for GAN-based augmentation that requires no
    GPU or deep-learning dependencies.
    """

    def __init__(
        self,
        noise_scale: float = 2.0,
        n_synthetic: int = 10,
        random_state: int = 42,
    ) -> None:
        self.noise_scale = noise_scale
        self.n_synthetic = n_synthetic
        self._rng = np.random.default_rng(random_state)

    def augment(self, X: np.ndarray, *, labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Return augmented data with synthetic anomaly rows appended."""
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            raise ValueError("Cannot augment empty input array")

        n_synthetic = min(self.n_synthetic, len(X))
        source_indices = self._rng.choice(len(X), size=n_synthetic, replace=False)
        synthetic = X[source_indices] + self._rng.normal(
            0.0, self.noise_scale, size=(n_synthetic, X.shape[1])
        )
        augmented = np.vstack([X, synthetic])

        if labels is None:
            return augmented, None

        labels = np.asarray(labels)
        synthetic_labels = np.ones(n_synthetic, dtype=labels.dtype)
        return augmented, np.concatenate([labels, synthetic_labels])
