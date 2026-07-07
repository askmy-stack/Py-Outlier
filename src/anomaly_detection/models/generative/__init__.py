"""Generative-model-based anomaly detectors (optional [generative] extra)."""

from anomaly_detection.models.generative.diffusion_detector import DiffusionDetector
from anomaly_detection.models.generative.gan_augmentation import GANAugmentation

__all__ = ["DiffusionDetector", "GANAugmentation"]
