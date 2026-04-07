"""
Model evaluation script — used as the CI quality gate.

Loads the saved model, runs inference on a held-out test set,
and exits non-zero if accuracy falls below the threshold.

Usage:
    python scripts/evaluate.py \
        --model-path "Image Anomaly Detection-2" \
        --data-path /path/to/test-data \
        --threshold 0.70

In CI, if EVAL_DATA_PATH env var is not set, the script exits 0 (skip).
This allows CI to pass when evaluation data is unavailable (Kaggle-hosted).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "Normal", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_IMAGE_PATH", "Image Anomaly Detection-2"),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.getenv("EVAL_DATA_PATH", ""),
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_path:
        print("EVAL_DATA_PATH not set — skipping model quality gate (no evaluation data in CI).")
        sys.exit(0)

    if not Path(args.data_path).exists():
        print(f"Data path does not exist: {args.data_path} — skipping.")
        sys.exit(0)

    if not Path(args.model_path).exists():
        print(f"Model path does not exist: {args.model_path}")
        sys.exit(1)

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"Loading model from: {args.model_path}")
    model = tf.saved_model.load(args.model_path)
    infer = model.signatures["serving_default"]

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        args.data_path,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    print(f"Evaluating on {gen.samples} samples...")
    all_preds = []
    all_true = []

    for batch_imgs, batch_labels in gen:
        output = infer(tf.constant(batch_imgs, dtype=tf.float32))
        # Extract the output tensor (key may vary by model)
        output_key = list(output.keys())[0]
        preds = output[output_key].numpy()
        all_preds.extend(np.argmax(preds, axis=1))
        all_true.extend(np.argmax(batch_labels, axis=1))
        if len(all_preds) >= gen.samples:
            break

    all_preds = np.array(all_preds[: gen.samples])
    all_true = np.array(all_true[: gen.samples])

    accuracy = np.mean(all_preds == all_true)
    print(f"\nOverall accuracy: {accuracy:.4f}")
    print(f"Quality gate threshold: {args.threshold:.4f}")
    print("\nPer-class report:")
    print(classification_report(all_true, all_preds, target_names=CLASSES))

    if accuracy < args.threshold:
        print(
            f"\nFAIL: accuracy {accuracy:.4f} < threshold {args.threshold:.4f}. "
            "Blocking deploy."
        )
        sys.exit(1)

    print(f"\nPASS: accuracy {accuracy:.4f} >= threshold {args.threshold:.4f}.")
    sys.exit(0)


if __name__ == "__main__":
    main()
