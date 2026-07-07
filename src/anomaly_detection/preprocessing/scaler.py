"""Feature scaling for tabular anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def apply_scaler(X: np.ndarray, scaler_name: str | None) -> np.ndarray:
    """Scale features using the configured scaler."""
    if scaler_name is None or scaler_name in {"", "none", "identity"}:
        return np.asarray(X, dtype=float)

    normalized = scaler_name.lower()
    if normalized == "standard":
        return StandardScaler().fit_transform(np.asarray(X, dtype=float))
    if normalized == "minmax":
        return MinMaxScaler().fit_transform(np.asarray(X, dtype=float))

    raise ValueError(f"Unsupported scaler: {scaler_name}")
