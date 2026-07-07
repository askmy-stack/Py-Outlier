"""UCF Crime vision dataset loader (local fixture / notebook reference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_detection.data_ingestion.base import BaseLoader
from anomaly_detection.data_ingestion.registry import DatasetEntry

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


class UCFLoader(BaseLoader):
    """Load UCF Crime features from a local sample (no Kaggle download in CI)."""

    def __init__(self, entry: DatasetEntry, *, quick: bool = True) -> None:
        self.entry = entry
        self.quick = quick

    def load(self) -> pd.DataFrame:
        fixture_path = FIXTURES_DIR / f"{self.entry.id}_sample.csv"
        if fixture_path.exists():
            return pd.read_csv(fixture_path)

        notebook_path = REPO_ROOT / (self.entry.local_notebook or "")
        if notebook_path.exists():
            raise FileNotFoundError(
                f"UCF Crime full dataset is not bundled. "
                f"See notebook for manual setup: {notebook_path}. "
                f"For CI/tests use the fixture at {fixture_path}."
            )
        raise FileNotFoundError(
            f"UCF Crime sample not found at {fixture_path} "
            f"and notebook reference is missing."
        )

    def metadata(self) -> dict[str, Any]:
        data = self.entry.to_dict()
        data["quick"] = self.quick
        data["fixture_path"] = str(FIXTURES_DIR / f"{self.entry.id}_sample.csv")
        return data
