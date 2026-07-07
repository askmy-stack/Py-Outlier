"""Registry-backed dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_detection.data_ingestion.base import BaseLoader
from anomaly_detection.data_ingestion.loaders.csv_loader import CSVLoader
from anomaly_detection.data_ingestion.loaders.openml_loader import OpenMLLoader
from anomaly_detection.data_ingestion.loaders.ucf_loader import UCFLoader
from anomaly_detection.data_ingestion.registry import DatasetEntry, get_dataset_entry

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


class RegistryLoader(BaseLoader):
    """Resolve and load a dataset by registry id."""

    def __init__(
        self,
        dataset_id: str,
        *,
        quick: bool = False,
        registry_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.quick = quick
        self.registry_path = registry_path
        self.cache_dir = cache_dir
        self.entry = get_dataset_entry(dataset_id, registry_path)
        self._delegate = self._create_delegate()

    def _create_delegate(self) -> BaseLoader:
        if self.entry.source == "openml":
            return OpenMLLoader(self.entry, quick=self.quick, cache_dir=self.cache_dir)
        if self.entry.id == "ucf_crime" or self.entry.domain == "vision":
            return UCFLoader(self.entry, quick=self.quick)
        if self.entry.source == "numenta" or self.entry.domain == "timeseries":
            return _NABLoader(self.entry, quick=self.quick)
        if self.entry.path:
            return CSVLoader(self.entry.path, target_column=self.entry.target_column)
        raise ValueError(
            f"No loader available for dataset '{self.entry.id}' "
            f"(source={self.entry.source}, domain={self.entry.domain})"
        )

    def load(self) -> pd.DataFrame:
        return self._delegate.load()

    def metadata(self) -> dict[str, Any]:
        data = self.entry.to_dict()
        data["quick"] = self.quick
        return data


class _NABLoader(BaseLoader):
    """Load NAB time-series samples from fixtures (quick) or a local CSV path."""

    def __init__(self, entry: DatasetEntry, *, quick: bool = False) -> None:
        self.entry = entry
        self.quick = quick

    def load(self) -> pd.DataFrame:
        if self.quick or not self.entry.path:
            fixture_path = FIXTURES_DIR / f"{self.entry.id}_sample.csv"
            if not fixture_path.exists():
                raise FileNotFoundError(f"NAB fixture not found: {fixture_path}")
            return pd.read_csv(fixture_path)

        return CSVLoader(self.entry.path, target_column=self.entry.target_column).load()

    def metadata(self) -> dict[str, Any]:
        data = self.entry.to_dict()
        data["quick"] = self.quick
        return data
