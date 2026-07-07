"""UCF vision domain: supervised image/video crime classification."""

from anomaly_detection.domains.vision.densenet import (
    CLASS_LABELS,
    NUM_CLASSES,
    build_densenet_classifier,
)
from anomaly_detection.domains.vision.model_manager import ModelManager
from anomaly_detection.domains.vision.preprocess import extract_frames, preprocess_image
from anomaly_detection.domains.vision.video import predict_image, predict_video

__all__ = [
    "CLASS_LABELS",
    "NUM_CLASSES",
    "ModelManager",
    "build_densenet_classifier",
    "extract_frames",
    "predict_image",
    "predict_video",
    "preprocess_image",
]
