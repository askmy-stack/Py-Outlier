"""
Singleton model manager for TensorFlow SavedModel loading and inference.

Models are loaded once at application startup via the FastAPI lifespan
and kept in memory for the process lifetime. Thread-safety is ensured
via a threading.Lock during the load phase.
"""

import threading
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self) -> None:
        self._image_model: Any = None
        self._video_model: Any = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        """Load both SavedModels from disk. Called once at startup."""
        import tensorflow as tf

        with self._lock:
            if self._loaded:
                return

            logger.info("loading_models", image_path=settings.model_image_path)
            try:
                self._image_model = tf.saved_model.load(settings.model_image_path)
                logger.info("image_model_loaded", path=settings.model_image_path)
            except Exception as exc:
                logger.warning("image_model_load_failed", error=str(exc))
                self._image_model = None

            try:
                self._video_model = tf.saved_model.load(settings.model_video_path)
                logger.info("video_model_loaded", path=settings.model_video_path)
            except Exception as exc:
                logger.warning("video_model_load_failed", error=str(exc))
                self._video_model = None

            self._loaded = True

    def _infer(self, model: Any, array: np.ndarray) -> np.ndarray:
        """Run inference using the SavedModel's default serving signature."""
        import tensorflow as tf

        infer_fn = model.signatures.get("serving_default")
        if infer_fn is None:
            # Fall back to calling the model directly (Keras SavedModel)
            output = model(tf.constant(array, dtype=tf.float32), training=False)
            return output.numpy()

        tensor = tf.constant(array, dtype=tf.float32)
        output = infer_fn(tensor)
        # The output key name varies; take the first value
        output_key = list(output.keys())[0]
        return output[output_key].numpy()

    def predict_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Run image inference.

        Args:
            image_array: shape (1, 64, 64, 3), values in [0, 1]

        Returns:
            np.ndarray of shape (14,) — softmax probabilities
        """
        if self._image_model is None:
            raise RuntimeError("Image model is not loaded.")
        probs = self._infer(self._image_model, image_array)
        return probs.flatten()  # (14,)

    def predict_video(self, frames: np.ndarray) -> np.ndarray:
        """
        Run video inference by averaging per-frame predictions.

        Args:
            frames: shape (N, 64, 64, 3), values in [0, 1]

        Returns:
            np.ndarray of shape (14,) — averaged softmax probabilities
        """
        if self._video_model is None:
            raise RuntimeError("Video model is not loaded.")
        probs = self._infer(self._video_model, frames)  # (N, 14)
        return probs.mean(axis=0)  # (14,)

    @property
    def is_ready(self) -> bool:
        return self._loaded


model_manager = ModelManager()
