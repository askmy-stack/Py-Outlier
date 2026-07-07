"""Tests for vision preprocessing (no TensorFlow model required)."""

from __future__ import annotations

import importlib.util
import io

import numpy as np
import pytest
from PIL import Image

from anomaly_detection.domains.vision.densenet import IMG_HEIGHT, IMG_WIDTH
from anomaly_detection.domains.vision.preprocess import preprocess_image


def _make_test_image_bytes(width: int = 128, height: int = 96, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_image_shape_and_dtype() -> None:
    arr = preprocess_image(_make_test_image_bytes())
    assert arr.shape == (1, IMG_HEIGHT, IMG_WIDTH, 3)
    assert arr.dtype == np.float32


def test_preprocess_image_normalizes_to_unit_range() -> None:
    arr = preprocess_image(_make_test_image_bytes(color=(255, 128, 0)))
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_preprocess_image_resizes_to_target() -> None:
    arr = preprocess_image(_make_test_image_bytes(width=320, height=240))
    assert arr.shape == (1, 64, 64, 3)


def test_preprocess_image_converts_rgba_to_rgb() -> None:
    image = Image.new("RGBA", (80, 80), (10, 20, 30, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    arr = preprocess_image(buf.getvalue())
    assert arr.shape == (1, IMG_HEIGHT, IMG_WIDTH, 3)


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="opencv not installed",
)
def test_extract_frames_requires_opencv() -> None:
    from anomaly_detection.domains.vision.preprocess import extract_frames

    # Minimal valid MP4 header is complex; just verify import and signature exist
    assert callable(extract_frames)
