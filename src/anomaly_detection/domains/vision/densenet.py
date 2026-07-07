"""DenseNet121-based UCF crime image classifier architecture and labels.

Note: This module implements **supervised multi-class classification** for the
UCF-Crime dataset. It is not unsupervised anomaly detection.
"""

from __future__ import annotations

from typing import Any

IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CLASSES = 14

CLASS_LABELS: list[str] = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "Normal",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


def feature_extractor(inputs: Any) -> Any:
    """DenseNet121 backbone (ImageNet weights, no top)."""
    import tensorflow as tf

    return tf.keras.applications.DenseNet121(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )(inputs)


def classifier(inputs: Any) -> Any:
    """Dense head for 14-class softmax classification."""
    import tensorflow as tf

    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)
    return x


def build_densenet_classifier() -> Any:
    """Build and compile the DenseNet121 UCF crime classifier (training scaffold)."""
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    backbone = feature_extractor(inputs)
    outputs = classifier(backbone)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )
    return model
