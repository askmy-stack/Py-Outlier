"""API tests for anomaly explanation endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from anomaly_detection.api.app import app

client = TestClient(app)


def test_explain_endpoint_mock_when_llm_disabled() -> None:
    payload = {
        "scores": [0.1, 0.95, 0.2],
        "predictions": [0, 1, 0],
        "feature_names": ["latency", "errors", "cpu"],
        "model_name": "isolation_forest",
        "config": {"llm": {"enabled": False}},
    }
    response = client.post("/explain", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "template"
    assert "anomal" in body["summary"].lower()
    assert body["llm_response"] == body["summary"]
    assert len(body["remediation_steps"]) >= 1


def test_explain_endpoint_rejects_mismatched_lengths() -> None:
    payload = {
        "scores": [0.1, 0.9],
        "predictions": [0, 1, 0],
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 400
