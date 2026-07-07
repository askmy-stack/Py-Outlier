"""Shared helpers for converting loaded frames to model inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def frame_to_arrays(
    frame: pd.DataFrame,
    target_column: str | None = None,
    exclude_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Split a DataFrame into features, optional labels, and feature names."""
    if frame.empty:
        raise ValueError("Dataset frame is empty")

    drop_columns = set(exclude_columns or [])
    labels: np.ndarray | None = None
    feature_frame = frame
    if target_column:
        if target_column not in frame.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        labels = frame[target_column].to_numpy()
        drop_columns.add(target_column)

    if drop_columns:
        missing = sorted(col for col in drop_columns if col not in frame.columns)
        if missing:
            raise ValueError(f"Columns not found in dataset: {', '.join(missing)}")
        feature_frame = frame.drop(columns=list(drop_columns))

    feature_names = list(feature_frame.columns)
    features = feature_frame.to_numpy(dtype=float)
    return features, labels, feature_names
