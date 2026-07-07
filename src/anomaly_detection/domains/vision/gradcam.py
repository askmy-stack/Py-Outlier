"""Grad-CAM (Gradient-weighted Class Activation Mapping) for vision models.

Grad-CAM requires a Keras Model with layer access. SavedModels loaded via
``tf.saved_model.load()`` may not support full Grad-CAM; this module returns
None gracefully in that case.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np


def compute_gradcam(
    model: Any,
    image_array: np.ndarray,
    class_idx: int,
    last_conv_layer_name: str = "conv5_block16_2_conv",
) -> np.ndarray | None:
    """Compute Grad-CAM heatmap for a given class index.

    Args:
        model: A tf.keras.Model instance (not a raw SavedModel).
        image_array: shape (1, 64, 64, 3), values in [0, 1]
        class_idx: Index of the target class (0-13)
        last_conv_layer_name: Name of the last convolutional layer in DenseNet121.

    Returns:
        np.ndarray of shape (64, 64) with values in [0, 1], or None on failure.
    """
    try:
        import cv2
        import tensorflow as tf

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

        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap)

        heatmap = heatmap.numpy()
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

        heatmap_resized = cv2.resize(heatmap, (64, 64))
        return heatmap_resized.astype(np.float32)

    except Exception:
        return None


def overlay_heatmap(original_image_bytes: bytes, heatmap: np.ndarray) -> bytes:
    """Overlay Grad-CAM heatmap onto the original image.

    Returns:
        PNG bytes of the overlaid image.
    """
    import cv2
    from PIL import Image

    original = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    orig_size = original.size
    orig_arr = np.array(original, dtype=np.float32) / 255.0

    heatmap_resized = cv2.resize(heatmap, orig_size)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlaid = 0.6 * orig_arr + 0.4 * colored_rgb
    overlaid = np.clip(overlaid, 0, 1)
    overlaid_uint8 = (overlaid * 255).astype(np.uint8)

    out_image = Image.fromarray(overlaid_uint8)
    buf = io.BytesIO()
    out_image.save(buf, format="PNG")
    return buf.getvalue()
