"""API tests for anomaly detection endpoints."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from anomaly_detection.api.app import app

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA = REPO_ROOT / "data" / "sample.csv"

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint_lists_detectors() -> None:
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert "isolation_forest" in models
    assert "zscore" in models


def test_detect_endpoint_with_sample_payload() -> None:
    frame = pd.read_csv(SAMPLE_DATA).drop(columns=["label"])
    payload = {"data": frame.head(20).values.tolist()}
    response = client.post("/detect", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "isolation_forest"
    assert body["n_samples"] == 20
    assert len(body["scores"]) == 20
    assert len(body["predictions"]) == 20


def test_detect_batch_csv_upload() -> None:
    csv_bytes = SAMPLE_DATA.read_bytes()
    response = client.post(
        "/detect/batch",
        files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["n_samples"] == len(pd.read_csv(SAMPLE_DATA))
    assert "metrics" in body
    assert body["feature_names"] == ["feature_1", "feature_2", "feature_3"]
