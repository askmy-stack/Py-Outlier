"""OpenML dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

from anomaly_detection.data_ingestion.base import BaseLoader
from anomaly_detection.data_ingestion.registry import DatasetEntry

FIXTURES_DIR = Path(__file__).resolve().parents[4] / "tests" / "fixtures"


class OpenMLLoader(BaseLoader):
    """Load tabular datasets from OpenML (or local fixtures in quick mode)."""

    def __init__(
        self,
        entry: DatasetEntry,
        *,
        quick: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.entry = entry
        self.quick = quick
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def load(self) -> pd.DataFrame:
        if self.quick:
            fixture_path = FIXTURES_DIR / f"{self.entry.id}_sample.csv"
            if not fixture_path.exists():
                raise FileNotFoundError(f"Quick-mode fixture not found: {fixture_path}")
            return pd.read_csv(fixture_path)

        if not self.entry.openml_name:
            raise ValueError(f"OpenML dataset '{self.entry.id}' is missing openml_name")

        kwargs: dict[str, Any] = {
            "name": self.entry.openml_name,
            "as_frame": True,
            "parser": "pandas",
        }
        if self.entry.openml_version is not None:
            kwargs["version"] = self.entry.openml_version
        if self.cache_dir is not None:
            kwargs["data_home"] = str(self.cache_dir)

        bundle = fetch_openml(**kwargs)
        if not isinstance(bundle.data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame from OpenML for '{self.entry.id}'")

        frame = bundle.data.copy()
        if bundle.target is not None:
            target_name = self.entry.target_column or "target"
            frame[target_name] = bundle.target
        return frame

    def metadata(self) -> dict[str, Any]:
        data = self.entry.to_dict()
        data["quick"] = self.quick
        return data
