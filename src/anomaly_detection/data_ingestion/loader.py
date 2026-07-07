"""Dataset loading facade for pipeline and CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.data_ingestion.loaders.csv_loader import CSVLoader
from anomaly_detection.data_ingestion.loaders.registry_loader import RegistryLoader
from anomaly_detection.data_ingestion.utils import frame_to_arrays


def load_dataset_from_config(
    dataset_config: dict[str, Any],
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Load features and labels from dataset.path or dataset.id."""
    target_column = dataset_config.get("target_column")

    if dataset_config.get("id"):
        loader = RegistryLoader(dataset_config["id"], quick=quick)
        entry_target = loader.entry.target_column
        effective_target = target_column or entry_target
        frame = loader.load()
        return frame_to_arrays(frame, target_column=effective_target)

    dataset_path = dataset_config.get("path")
    if not dataset_path:
        raise ValueError("Config must include dataset.path or dataset.id")

    loader = CSVLoader(dataset_path, target_column=target_column)
    frame = loader.load()
    return frame_to_arrays(frame, target_column=target_column)


def load_csv(
    path: str | Path,
    target_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Load feature matrix and optional labels from a CSV file."""
    loader = CSVLoader(path, target_column=target_column)
    frame = loader.load()
    return frame_to_arrays(frame, target_column=target_column)
