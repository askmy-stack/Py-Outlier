"""
Training script for the DenseNet121-based anomaly detection model.

Replaces the Jupyter notebook workflow with a reproducible, CI-runnable script.
Logs all experiments to MLflow for tracking and comparison.

Usage:
    python scripts/train.py \
        --data-dir /path/to/ucf-crime \
        --output-dir ./trained_models/run_001 \
        --epochs 50 \
        --batch-size 64 \
        --image-size 64 \
        --mlflow-uri http://localhost:5000

Dataset directory structure expected:
    data-dir/
        train/
            Abuse/
            Arrest/
            ...
        test/
            Abuse/
            Arrest/
            ...
"""

import argparse
import os
import time
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "Normal", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]
NUM_CLASSES = len(CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--data-dir", type=str, required=True, help="Root dataset directory")
    parser.add_argument("--output-dir", type=str, default="./trained_models/latest")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--experiment-name", type=str, default="anonmaly-detection")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(image_size: int, num_classes: int, dropout: float) -> Model:
    """DenseNet121 with custom classification head."""
    base = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=Input(shape=(image_size, image_size, 3)),
    )
    base.trainable = False  # freeze backbone initially

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base.input, outputs=x)


def make_generators(
    data_dir: str, image_size: int, batch_size: int
):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASSES,
        seed=42,
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )
    return train_gen, test_gen


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "backbone": "DenseNet121",
            "num_classes": NUM_CLASSES,
        })

        # ── Data ──────────────────────────────────────────────────────────
        train_gen, test_gen = make_generators(args.data_dir, args.image_size, args.batch_size)

        # ── Model ─────────────────────────────────────────────────────────
        model = build_model(args.image_size, NUM_CLASSES, args.dropout)
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        # ── Callbacks ─────────────────────────────────────────────────────
        checkpoint_path = str(output_dir / "best_model.keras")
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy"),
            EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
            ReduceLROnPlateau(factor=0.5, patience=4, monitor="val_loss", min_lr=1e-7),
        ]

        # ── Phase 1: Train head only ───────────────────────────────────────
        print("\n── Phase 1: Training classification head (backbone frozen) ──")
        start = time.time()
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=test_gen,
            callbacks=callbacks,
        )
        phase1_duration = time.time() - start

        # ── Phase 2: Fine-tune top layers ─────────────────────────────────
        print("\n── Phase 2: Fine-tuning top 30 layers of DenseNet121 ──")
        for layer in model.layers[-30:]:
            layer.trainable = True
        model.compile(
            optimizer=Adam(learning_rate=args.learning_rate / 10),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_gen,
            epochs=20,
            validation_data=test_gen,
            callbacks=callbacks,
        )

        # ── Evaluation ────────────────────────────────────────────────────
        print("\n── Evaluating on test set ──")
        loss, accuracy = model.evaluate(test_gen)
        print(f"Test accuracy: {accuracy:.4f} | Test loss: {loss:.4f}")

        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_loss": loss,
            "phase1_duration_s": phase1_duration,
            "best_val_accuracy": max(history.history["val_accuracy"]),
        })

        # ── Save model ────────────────────────────────────────────────────
        save_path = str(output_dir / "saved_model")
        model.save(save_path)
        mlflow.keras.log_model(model, "model")
        print(f"Model saved to {save_path}")
        print(f"MLflow model logged. Run ID: {run.info.run_id}")

        if accuracy < 0.70:
            print(f"WARNING: accuracy {accuracy:.4f} is below the 0.70 quality gate threshold.")


if __name__ == "__main__":
    main()
