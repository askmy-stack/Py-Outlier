"""
API endpoint tests.

Uses httpx AsyncClient against the FastAPI app with the ML model mocked
so tests run without TensorFlow or GPU.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_model_manager():
    """Patch model_manager so tests don't require real TF models."""
    probs = np.zeros(14, dtype=np.float32)
    probs[6] = 0.91  # "Fighting" wins with 91% confidence

    mock = MagicMock()
    mock.is_ready = True
    mock.predict_image.return_value = probs
    mock._image_model = None  # no Keras model → Grad-CAM skipped gracefully

    with patch("app.api.analyze.model_manager", mock):
        yield mock


@pytest.fixture
async def client(mock_model_manager):
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.anyio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


@pytest.mark.anyio
async def test_analyze_image_no_file(client):
    """POST without a file should return 422 Unprocessable Entity."""
    response = await client.post("/analyze/image")
    assert response.status_code == 422


@pytest.mark.anyio
async def test_analyze_image_wrong_mime(client):
    """POST with a text file should return 400 Bad Request."""
    response = await client.post(
        "/analyze/image",
        files={"file": ("test.txt", b"hello world", "text/plain")},
        data={"message": "suspicious activity"},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.anyio
async def test_analyze_image_success(client, mock_model_manager):
    """POST a valid JPEG should return AnalysisResponse with predicted class."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = await client.post(
        "/analyze/image",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        data={"message": "Two people fighting near the alley"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["media_type"] == "image"
    assert data["classification"]["predicted_class"] == "Fighting"
    assert data["classification"]["confidence"] == pytest.approx(0.91)
    assert len(data["classification"]["all_scores"]) == 14
    assert "submission_id" in data
    assert data["processing_time_ms"] > 0


@pytest.mark.anyio
async def test_job_status_not_found(client):
    """GET /analyze/job/<fake-id> should return 404 when Celery backend unavailable."""
    with patch("app.api.analyze.AsyncResult") as mock_result_cls:
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result_cls.return_value = mock_result

        with patch("app.api.analyze.celery_app", MagicMock()):
            response = await client.get("/analyze/job/nonexistent-job-id")

    assert response.status_code == 404
