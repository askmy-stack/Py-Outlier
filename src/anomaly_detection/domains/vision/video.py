"""Video classification for UCF crime detection.

Note: This is **supervised classification**, not unsupervised anomaly detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from anomaly_detection.domains.vision.preprocess import extract_frames

if TYPE_CHECKING:
    from anomaly_detection.domains.vision.model_manager import ModelManager


def predict_video(
    video_bytes: bytes,
    model_manager: ModelManager,
    max_frames: int = 16,
) -> dict[str, Any]:
    """Extract frames from video bytes and run classification.

    Args:
        video_bytes: Raw video file bytes (MP4, etc.)
        model_manager: Loaded ModelManager with video model
        max_frames: Maximum frames to sample evenly across the video

    Returns:
        Classification result dict with predicted_class, confidence, all_scores
    """
    frames = extract_frames(video_bytes, max_frames=max_frames)
    probs = model_manager.predict_video_frames(frames)
    result = model_manager.format_prediction(probs)
    result["n_frames"] = int(frames.shape[0])
    return result


def predict_image(
    image_bytes: bytes,
    model_manager: ModelManager,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess image bytes and run classification.

    Returns:
        Tuple of (preprocessed array, classification result dict)
    """
    from anomaly_detection.domains.vision.preprocess import preprocess_image

    image_array = preprocess_image(image_bytes)
    probs = model_manager.predict_image(image_array)
    return image_array, model_manager.format_prediction(probs)
