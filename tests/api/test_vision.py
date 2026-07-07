"""API tests for vision classification endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_detection.api.app import app
from anomaly_detection.domains.vision.densenet import CLASS_LABELS, NUM_CLASSES

pytest.importorskip("tensorflow", reason="tensorflow not installed")

client = TestClient(app)


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (100, 100), (42, 84, 126))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _mock_probs(class_idx: int = 7, confidence: float = 0.92) -> np.ndarray:
    probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    probs[class_idx] = confidence
    remaining = 1.0 - confidence
    for i in range(NUM_CLASSES):
        if i != class_idx:
            probs[i] = remaining / (NUM_CLASSES - 1)
    return probs


@patch("anomaly_detection.api.routes.vision._get_model_manager")
def test_analyze_image_returns_classification(mock_get_manager: MagicMock) -> None:
    mock_manager = MagicMock()
    mock_manager.predict_image.return_value = _mock_probs(class_idx=7)
    mock_manager.format_prediction.return_value = {
        "predicted_class": CLASS_LABELS[7],
        "confidence": 0.92,
        "class_index": 7,
        "all_scores": {label: 0.07 for label in CLASS_LABELS},
        "disclaimer": "UCF vision module performs supervised multi-class classification, not unsupervised anomaly detection.",
    }
    mock_get_manager.return_value = mock_manager

    response = client.post(
        "/vision/analyze/image",
        files={"file": ("test.png", _make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "image"
    assert body["predicted_class"] == "Normal"
    assert body["confidence"] == pytest.approx(0.92)
    assert "disclaimer" in body
    assert "not unsupervised" in body["disclaimer"].lower() or "supervised" in body["disclaimer"].lower()


@patch("anomaly_detection.api.routes.vision._get_model_manager")
def test_analyze_image_rejects_unsupported_type(mock_get_manager: MagicMock) -> None:
    response = client.post(
        "/vision/analyze/image",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    mock_get_manager.assert_not_called()


@patch("anomaly_detection.api.routes.vision._get_model_manager")
@patch("anomaly_detection.domains.vision.video.extract_frames")
def test_analyze_video_returns_classification(
    mock_extract: MagicMock,
    mock_get_manager: MagicMock,
) -> None:
    mock_extract.return_value = np.zeros((4, 64, 64, 3), dtype=np.float32)
    mock_manager = MagicMock()
    mock_manager.predict_video_frames.return_value = _mock_probs(class_idx=3)
    mock_manager.format_prediction.return_value = {
        "predicted_class": CLASS_LABELS[3],
        "confidence": 0.88,
        "class_index": 3,
        "all_scores": {label: 0.08 for label in CLASS_LABELS},
        "disclaimer": "UCF vision module performs supervised multi-class classification, not unsupervised anomaly detection.",
    }
    mock_get_manager.return_value = mock_manager

    response = client.post(
        "/vision/analyze/video",
        files={"file": ("test.mp4", b"fake-video-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "video"
    assert body["predicted_class"] == "Assault"


def test_model_manager_format_prediction() -> None:
    from anomaly_detection.domains.vision.model_manager import ModelManager

    probs = _mock_probs(class_idx=0, confidence=0.75)
    result = ModelManager.format_prediction(probs)
    assert result["predicted_class"] == "Abuse"
    assert result["confidence"] == pytest.approx(0.75)
    assert len(result["all_scores"]) == NUM_CLASSES
    assert "supervised" in result["disclaimer"].lower()
