"""CSV file loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_detection.data_ingestion.base import BaseLoader


class CSVLoader(BaseLoader):
    """Load tabular data from a local CSV file."""

    def __init__(self, path: str | Path, *, target_column: str | None = None) -> None:
        self.path = Path(path)
        self.target_column = target_column

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        frame = pd.read_csv(self.path)
        if frame.empty:
            raise ValueError(f"Dataset is empty: {self.path}")
        return frame

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.path.stem,
            "domain": "tabular",
            "source": "csv",
            "path": str(self.path),
            "target_column": self.target_column,
            "labeled": self.target_column is not None,
        }
