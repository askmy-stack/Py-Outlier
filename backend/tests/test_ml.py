"""
Unit tests for ML preprocessing utilities.
These tests run without TensorFlow — they only test NumPy/PIL operations.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image


def make_jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_shape():
    from app.ml.preprocess import preprocess_image

    image_bytes = make_jpeg_bytes(100, 100)
    result = preprocess_image(image_bytes)
    assert result.shape == (1, 64, 64, 3), f"Expected (1, 64, 64, 3), got {result.shape}"


def test_preprocess_image_normalized():
    from app.ml.preprocess import preprocess_image

    image_bytes = make_jpeg_bytes(200, 300)
    result = preprocess_image(image_bytes)
    assert result.min() >= 0.0, "Values should be >= 0"
    assert result.max() <= 1.0, "Values should be <= 1"


def test_preprocess_image_dtype():
    from app.ml.preprocess import preprocess_image

    image_bytes = make_jpeg_bytes()
    result = preprocess_image(image_bytes)
    assert result.dtype == np.float32


def test_classes_count():
    from app.core.config import settings

    assert len(settings.classes) == 14, f"Expected 14 classes, got {len(settings.classes)}"


def test_classes_names():
    from app.core.config import settings

    expected = {
        "Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "Normal", "RoadAccidents",
        "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
    }
    assert set(settings.classes) == expected
