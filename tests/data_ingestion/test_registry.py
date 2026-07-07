"""Tests for dataset registry and loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anomaly_detection.data_ingestion.loader import load_csv, load_dataset_from_config
from anomaly_detection.data_ingestion.loaders.csv_loader import CSVLoader
from anomaly_detection.data_ingestion.loaders.openml_loader import OpenMLLoader
from anomaly_detection.data_ingestion.loaders.registry_loader import RegistryLoader
from anomaly_detection.data_ingestion.loaders.ucf_loader import UCFLoader
from anomaly_detection.data_ingestion.registry import get_dataset_entry, load_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
SAMPLE_CSV = REPO_ROOT / "data" / "sample.csv"


def test_registry_loads_all_datasets() -> None:
    registry = load_registry()
    assert set(registry) == {"credit_card_fraud", "nab", "ucf_crime"}


def test_registry_entries_include_license() -> None:
    registry = load_registry()
    for entry in registry.values():
        assert entry.license
        assert entry.description or entry.local_notebook


def test_get_dataset_entry_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset_entry("missing_dataset")


def test_csv_loader_reads_sample() -> None:
    loader = CSVLoader(SAMPLE_CSV, target_column="label")
    frame = loader.load()
    assert len(frame) > 0
    meta = loader.metadata()
    assert meta["source"] == "csv"
    assert meta["labeled"] is True


def test_load_csv_backward_compatible() -> None:
    features, labels, feature_names = load_csv(SAMPLE_CSV, target_column="label")
    assert features.shape[0] == labels.shape[0]
    assert len(feature_names) == features.shape[1]


def test_registry_loader_quick_credit_card() -> None:
    loader = RegistryLoader("credit_card_fraud", quick=True)
    frame = loader.load()
    assert isinstance(frame, pd.DataFrame)
    assert "Class" in frame.columns
    assert len(frame) >= 10


def test_openml_loader_quick_uses_fixture() -> None:
    entry = get_dataset_entry("credit_card_fraud")
    loader = OpenMLLoader(entry, quick=True)
    frame = loader.load()
    assert "Class" in frame.columns


def test_ucf_loader_reads_local_fixture() -> None:
    entry = get_dataset_entry("ucf_crime")
    loader = UCFLoader(entry, quick=True)
    frame = loader.load()
    assert "label" in frame.columns
    meta = loader.metadata()
    assert "image-anomaly-detection-2.ipynb" in (entry.local_notebook or "")
    assert meta["fixture_path"].endswith("ucf_crime_sample.csv")


def test_load_dataset_from_config_by_id_quick() -> None:
    features, labels, feature_names = load_dataset_from_config(
        {"id": "credit_card_fraud", "target_column": "Class"},
        quick=True,
    )
    assert features.shape[0] == labels.shape[0]
    assert len(feature_names) > 0


def test_load_dataset_from_config_by_path() -> None:
    features, labels, feature_names = load_dataset_from_config(
        {"path": str(SAMPLE_CSV), "target_column": "label"},
    )
    assert features.shape[0] > 0
    assert labels is not None
    assert len(feature_names) == 3


def test_nab_quick_fixture_loads() -> None:
    loader = RegistryLoader("nab", quick=True)
    frame = loader.load()
    assert {"timestamp", "value", "label"}.issubset(frame.columns)
