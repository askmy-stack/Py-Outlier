"""Base loader interface for dataset ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseLoader(ABC):
    """Abstract dataset loader returning tabular data and metadata."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load dataset as a pandas DataFrame."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return dataset metadata (id, domain, source, license, etc.)."""
