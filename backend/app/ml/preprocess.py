import io
import os
import tempfile

import cv2
import numpy as np
from PIL import Image


TARGET_SIZE = (64, 64)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes, resize to 64x64, normalize to [0, 1].

    Returns:
        np.ndarray of shape (1, 64, 64, 3), dtype float32
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 64, 64, 3)


def extract_frames(video_bytes: bytes, max_frames: int = 16) -> np.ndarray:
    """
    Extract evenly-spaced frames from video bytes.

    Writes to a temp file (required by OpenCV), samples max_frames evenly
    across the video duration, preprocesses each frame to 64x64.

    Returns:
        np.ndarray of shape (N, 64, 64, 3), dtype float32, where N <= max_frames
    """
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise ValueError("Could not read frame count from video.")

        # Evenly sample frame indices
        indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            # OpenCV uses BGR — convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
            frames.append(frame_resized.astype(np.float32) / 255.0)

        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    return np.stack(frames, axis=0)  # (N, 64, 64, 3)
