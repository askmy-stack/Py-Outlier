"""Singleton model manager for TensorFlow SavedModel loading and inference."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.domains.vision.densenet import CLASS_LABELS


class ModelManager:
    """Load and serve UCF image/video SavedModels from configurable paths."""

    def __init__(
        self,
        image_model_path: str | Path | None = None,
        video_model_path: str | Path | None = None,
    ) -> None:
        self._image_model_path = Path(image_model_path) if image_model_path else None
        self._video_model_path = Path(video_model_path) if video_model_path else None
        self._image_model: Any = None
        self._video_model: Any = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        """Load both SavedModels from disk."""
        import tensorflow as tf

        with self._lock:
            if self._loaded:
                return

            if self._image_model_path and self._image_model_path.exists():
                self._image_model = tf.saved_model.load(str(self._image_model_path))

            if self._video_model_path and self._video_model_path.exists():
                self._video_model = tf.saved_model.load(str(self._video_model_path))

            self._loaded = True

    def _infer(self, model: Any, array: np.ndarray) -> np.ndarray:
        """Run inference using the SavedModel's default serving signature."""
        import tensorflow as tf

        infer_fn = model.signatures.get("serving_default")
        if infer_fn is None:
            output = model(tf.constant(array, dtype=tf.float32), training=False)
            return output.numpy()

        tensor = tf.constant(array, dtype=tf.float32)
        output = infer_fn(tensor)
        output_key = next(iter(output.keys()))
        return output[output_key].numpy()

    def predict_image(self, image_array: np.ndarray) -> np.ndarray:
        """Run image inference.

        Args:
            image_array: shape (1, 64, 64, 3), values in [0, 1]

        Returns:
            np.ndarray of shape (14,) — softmax probabilities
        """
        if self._image_model is None:
            raise RuntimeError("Image model is not loaded.")
        probs = self._infer(self._image_model, image_array)
        return probs.flatten()

    def predict_video_frames(self, frames: np.ndarray) -> np.ndarray:
        """Run video inference by averaging per-frame predictions.

        Args:
            frames: shape (N, 64, 64, 3), values in [0, 1]

        Returns:
            np.ndarray of shape (14,) — averaged softmax probabilities
        """
        if self._video_model is None:
            raise RuntimeError("Video model is not loaded.")
        probs = self._infer(self._video_model, frames)
        if probs.ndim == 1:
            return probs
        return probs.mean(axis=0)

    @property
    def image_model(self) -> Any:
        return self._image_model

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @staticmethod
    def format_prediction(probs: np.ndarray) -> dict[str, Any]:
        """Format softmax probabilities into a classification result dict."""
        top_idx = int(probs.argmax())
        return {
            "predicted_class": CLASS_LABELS[top_idx],
            "confidence": float(probs[top_idx]),
            "class_index": top_idx,
            "all_scores": {label: float(probs[i]) for i, label in enumerate(CLASS_LABELS)},
            "disclaimer": (
                "UCF vision module performs supervised multi-class classification, "
                "not unsupervised anomaly detection."
            ),
        }
