"""Pipeline tests for registry-backed datasets."""

from __future__ import annotations

from pathlib import Path

from anomaly_detection.config.loader import load_config
from anomaly_detection.pipeline import run_detection

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAUD_CONFIG = REPO_ROOT / "configs" / "examples" / "fraud.yaml"


def test_pipeline_with_dataset_id_quick() -> None:
    config = load_config(FRAUD_CONFIG)
    report = run_detection(config, quick=True)
    assert report["n_samples"] > 0
    assert "metrics" in report


def test_pipeline_with_dataset_path_still_works() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    report = run_detection(config)
    assert report["n_samples"] > 0
