"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Grad-CAM produces a heatmap highlighting the regions of the input image
that most influenced the model's prediction. This makes AI decisions
interpretable and auditable — critical for a law-enforcement-adjacent platform.

Note on SavedModel compatibility:
    Grad-CAM requires access to intermediate layer activations and their
    gradients. Full support requires the model to be loaded as a Keras
    functional model (model.get_layer()). With a TensorFlow SavedModel
    loaded via tf.saved_model.load(), intermediate layer access is limited.

    This module provides:
    1. Full Grad-CAM for Keras Model objects
    2. A graceful fallback (returns None) for opaque SavedModel objects

    To enable full Grad-CAM, load the model with tf.keras.models.load_model()
    instead of tf.saved_model.load() where possible.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np


def compute_gradcam(
    model: Any,
    image_array: np.ndarray,
    class_idx: int,
    last_conv_layer_name: str = "conv5_block16_2_conv",  # DenseNet121 last conv layer
) -> np.ndarray | None:
    """
    Compute Grad-CAM heatmap for a given class index.

    Args:
        model: A tf.keras.Model instance (not a raw SavedModel).
               Returns None if model doesn't support get_layer().
        image_array: shape (1, 64, 64, 3), values in [0, 1]
        class_idx: Index of the target class (0-13)
        last_conv_layer_name: Name of the last convolutional layer in the model.

    Returns:
        np.ndarray of shape (64, 64) with values in [0, 1], or None on failure.
    """
    try:
        import tensorflow as tf

        # Verify the model exposes get_layer (Keras Model, not raw SavedModel)
        if not hasattr(model, "get_layer"):
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output,
            ],
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image_array, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            class_score = predictions[:, class_idx]

        grads = tape.gradient(class_score, conv_outputs)  # (1, H, W, C)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        conv_outputs = conv_outputs[0]  # (H, W, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
        heatmap = tf.squeeze(heatmap)  # (H, W)
        heatmap = tf.nn.relu(heatmap)  # apply ReLU

        # Normalize to [0, 1]
        heatmap = heatmap.numpy()
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

        # Resize to input image size (64x64)
        import cv2
        heatmap_resized = cv2.resize(heatmap, (64, 64))
        return heatmap_resized.astype(np.float32)

    except Exception:
        return None


def overlay_heatmap(original_image_bytes: bytes, heatmap: np.ndarray) -> bytes:
    """
    Overlay Grad-CAM heatmap onto the original image.

    Args:
        original_image_bytes: Raw image bytes (JPEG or PNG)
        heatmap: np.ndarray of shape (H, W), values in [0, 1]

    Returns:
        PNG bytes of the overlaid image.
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Decode original image
    original = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    orig_size = original.size  # (W, H)
    orig_arr = np.array(original, dtype=np.float32) / 255.0

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, orig_size)

    # Apply 'jet' colormap: heatmap values [0,1] → RGB colors
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Blend: alpha=0.4 for heatmap, 0.6 for original
    overlaid = 0.6 * orig_arr + 0.4 * colored_rgb
    overlaid = np.clip(overlaid, 0, 1)
    overlaid_uint8 = (overlaid * 255).astype(np.uint8)

    # Encode as PNG bytes
    out_image = Image.fromarray(overlaid_uint8)
    buf = io.BytesIO()
    out_image.save(buf, format="PNG")
    return buf.getvalue()
