"""API tests for root cause analysis endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from anomaly_detection.api.app import app
from anomaly_detection.api.routes import root_cause as root_cause_routes

client = TestClient(app)


def _chain_metrics_payload(n: int = 80, spike: float = 20.0) -> dict:
    rng = np.random.default_rng(7)
    a = np.cumsum(rng.normal(0, 0.1, n)) + 10.0
    b = np.empty(n)
    c = np.empty(n)
    b[0] = a[0]
    c[0] = b[0]
    for t in range(1, n):
        b[t] = 0.7 * a[t - 1] + 0.3 * a[t] + rng.normal(0, 0.05)
        c[t] = 0.7 * b[t - 1] + 0.3 * b[t] + rng.normal(0, 0.05)
    a[-1] += spike
    frame = pd.DataFrame({"A": a, "B": b, "C": c})
    return {"A": frame["A"].tolist(), "B": frame["B"].tolist(), "C": frame["C"].tolist()}


def test_post_root_cause_returns_top_causes() -> None:
    root_cause_routes._rca_cache.clear()
    payload = {
        "anomaly_id": "anomaly-chain-1",
        "metrics": _chain_metrics_payload(),
        "timestamp": -1,
        "top_k": 3,
    }
    response = client.post("/root_cause", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["anomaly_id"] == "anomaly-chain-1"
    assert len(body["causes"]) == 3
    assert body["causes"][0]["metric"] == "A"
    assert body["causes"][0]["rank"] == 1
    assert "disclaimer" in body
    assert body["graph"]["nodes"]
    assert body["method"] in {"correlation_fallback", "pyrca"}


def test_get_root_cause_cached_result() -> None:
    root_cause_routes._rca_cache.clear()
    anomaly_id = "cached-anomaly-42"
    payload = {
        "anomaly_id": anomaly_id,
        "metrics": _chain_metrics_payload(spike=15.0),
    }
    post = client.post("/root_cause", json=payload)
    assert post.status_code == 200

    get_resp = client.get(f"/root_cause/{anomaly_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["anomaly_id"] == anomaly_id
    assert get_resp.json()["causes"][0]["metric"] == "A"
    assert "disclaimer" in get_resp.json()


def test_get_root_cause_missing_returns_404() -> None:
    response = client.get("/root_cause/nonexistent-id")
    assert response.status_code == 404


def test_post_root_cause_invalid_metrics_returns_400() -> None:
    response = client.post(
        "/root_cause",
        json={"anomaly_id": "bad", "metrics": {"only_one": [1.0, 2.0]}},
    )
    assert response.status_code == 400
