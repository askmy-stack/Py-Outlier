"""Tests for optional API key authentication."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from anomaly_detection.api.app import app

client = TestClient(app)


def test_open_mode_when_no_key_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANOMALY_API_KEY", raising=False)
    response = client.get("/models")
    assert response.status_code == 200


def test_rejects_missing_key_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANOMALY_API_KEY", "sk_test")
    response = client.get("/models")
    assert response.status_code == 401


def test_rejects_invalid_key_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANOMALY_API_KEY", "sk_test")
    response = client.get("/models", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_accepts_valid_key_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANOMALY_API_KEY", "sk_test")
    response = client.get("/models", headers={"X-API-Key": "sk_test"})
    assert response.status_code == 200


def test_health_is_always_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANOMALY_API_KEY", "sk_test")
    response = client.get("/health")
    assert response.status_code == 200


def test_cors_preflight_bypasses_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANOMALY_API_KEY", "sk_test")
    response = client.options(
        "/detect",
        headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://example.com"
