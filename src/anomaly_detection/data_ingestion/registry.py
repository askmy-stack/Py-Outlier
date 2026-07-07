"""Dataset registry loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "datasets" / "registry.yaml"

VALID_DOMAINS = {"tabular", "timeseries", "vision"}
VALID_SOURCES = {"openml", "numenta", "local", "csv", "kaggle"}


@dataclass(frozen=True)
class DatasetEntry:
    """Validated dataset metadata from the registry."""

    id: str
    domain: str
    source: str
    labeled: bool
    description: str | None = None
    url: str | None = None
    license: str | None = None
    target_column: str | None = None
    openml_name: str | None = None
    openml_version: int | None = None
    local_notebook: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "id": self.id,
                "domain": self.domain,
                "source": self.source,
                "labeled": self.labeled,
                "description": self.description,
                "url": self.url,
                "license": self.license,
                "target_column": self.target_column,
                "openml_name": self.openml_name,
                "openml_version": self.openml_version,
                "local_notebook": self.local_notebook,
                "path": self.path,
            }.items()
            if value is not None
        }


def _validate_entry(raw: dict[str, Any]) -> DatasetEntry:
    dataset_id = raw.get("id")
    if not dataset_id or not isinstance(dataset_id, str):
        raise ValueError("Each registry entry must include a string 'id'")

    domain = raw.get("domain")
    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"Dataset '{dataset_id}' has invalid domain '{domain}'. "
            f"Expected one of: {', '.join(sorted(VALID_DOMAINS))}"
        )

    source = raw.get("source")
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Dataset '{dataset_id}' has invalid source '{source}'. "
            f"Expected one of: {', '.join(sorted(VALID_SOURCES))}"
        )

    labeled = raw.get("labeled")
    if not isinstance(labeled, bool):
        raise ValueError(f"Dataset '{dataset_id}' must include boolean 'labeled'")

    openml_version = raw.get("openml_version")
    if openml_version is not None and not isinstance(openml_version, int):
        raise ValueError(f"Dataset '{dataset_id}' openml_version must be an integer")

    return DatasetEntry(
        id=dataset_id,
        domain=domain,
        source=source,
        labeled=labeled,
        description=raw.get("description"),
        url=raw.get("url"),
        license=raw.get("license"),
        target_column=raw.get("target_column"),
        openml_name=raw.get("openml_name"),
        openml_version=openml_version,
        local_notebook=raw.get("local_notebook"),
        path=raw.get("path"),
    )


def load_registry(path: str | Path | None = None) -> dict[str, DatasetEntry]:
    """Load and validate the dataset registry YAML."""
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    with registry_path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    if not isinstance(document, dict):
        raise ValueError(f"Registry root must be a mapping: {registry_path}")

    entries = document.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Registry must include a non-empty 'datasets' list: {registry_path}")

    registry: dict[str, DatasetEntry] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("Each registry entry must be a mapping")
        entry = _validate_entry(raw)
        if entry.id in registry:
            raise ValueError(f"Duplicate dataset id in registry: {entry.id}")
        registry[entry.id] = entry

    return registry


def get_dataset_entry(dataset_id: str, path: str | Path | None = None) -> DatasetEntry:
    """Return a single validated registry entry by id."""
    registry = load_registry(path)
    if dataset_id not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown dataset '{dataset_id}'. Available: {available}")
    return registry[dataset_id]
