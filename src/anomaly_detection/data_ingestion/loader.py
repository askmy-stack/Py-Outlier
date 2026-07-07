"""Dataset loading facade for pipeline and CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.data_ingestion.loaders.csv_loader import CSVLoader
from anomaly_detection.data_ingestion.loaders.registry_loader import RegistryLoader
from anomaly_detection.data_ingestion.utils import frame_to_arrays


def _load_frame_from_config(
    dataset_config: dict[str, Any],
    *,
    quick: bool = False,
):
    """Load the raw dataset frame from config."""
    from pandas import DataFrame

    target_column = dataset_config.get("target_column")

    if dataset_config.get("id"):
        loader = RegistryLoader(dataset_config["id"], quick=quick)
        return loader.load()

    dataset_path = dataset_config.get("path")
    if not dataset_path:
        raise ValueError("Config must include dataset.path or dataset.id")

    loader = CSVLoader(dataset_path, target_column=target_column)
    frame = loader.load()
    if not isinstance(frame, DataFrame):
        raise TypeError("Dataset loader must return a pandas DataFrame")
    return frame


def load_protected_attributes(
    frame,
    protected_attributes: list[str],
) -> dict[str, np.ndarray]:
    """Extract configured protected attribute columns from a dataset frame."""
    missing = [name for name in protected_attributes if name not in frame.columns]
    if missing:
        raise ValueError(
            "Protected attributes missing from dataset: "
            + ", ".join(missing)
            + ". Configure protected_attributes explicitly; columns are never inferred."
        )
    return {name: frame[name].to_numpy() for name in protected_attributes}


def load_dataset_from_config(
    dataset_config: dict[str, Any],
    *,
    quick: bool = False,
    exclude_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Load features and labels from dataset.path or dataset.id."""
    target_column = dataset_config.get("target_column")
    frame = _load_frame_from_config(dataset_config, quick=quick)

    if dataset_config.get("id"):
        entry_target = RegistryLoader(dataset_config["id"], quick=quick).entry.target_column
        effective_target = target_column or entry_target
    else:
        effective_target = target_column

    return frame_to_arrays(
        frame,
        target_column=effective_target,
        exclude_columns=exclude_columns,
    )


def load_csv(
    path: str | Path,
    target_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Load feature matrix and optional labels from a CSV file."""
    loader = CSVLoader(path, target_column=target_column)
    frame = loader.load()
    return frame_to_arrays(frame, target_column=target_column)
