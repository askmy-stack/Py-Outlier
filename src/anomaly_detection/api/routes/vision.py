"""Vision API routes — separate from tabular /detect.

UCF vision endpoints perform supervised multi-class classification.
They do not produce generic unsupervised anomaly scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from anomaly_detection.api.schemas import VisionClassificationResponse
from anomaly_detection.config.loader import load_config

router = APIRouter(prefix="/vision/analyze", tags=["vision"])

DEFAULT_CONFIG_PATH = "configs/examples/vision.yaml"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}

_model_manager: Any = None


def _get_vision_config() -> dict[str, Any]:
    config = load_config(DEFAULT_CONFIG_PATH)
    vision = config.get("vision", {})
    if not isinstance(vision, dict):
        raise ValueError("vision config must be a mapping")
    return vision


def _get_model_manager() -> Any:
    global _model_manager
    if _model_manager is not None:
        return _model_manager

    try:
        from anomaly_detection.domains.vision.model_manager import ModelManager
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision extras not installed. Run: pip install -e '.[vision]'",
        ) from exc

    vision_cfg = _get_vision_config()
    repo_root = Path(__file__).resolve().parents[4]

    image_path = repo_root / vision_cfg.get("image_model_path", "Image Anomaly Detection-2")
    video_path = repo_root / vision_cfg.get("video_model_path", "Video Anomaly Detection")

    _model_manager = ModelManager(image_model_path=image_path, video_model_path=video_path)
    _model_manager.load()
    return _model_manager


def _max_upload_bytes() -> int:
    vision_cfg = _get_vision_config()
    max_mb = int(vision_cfg.get("max_upload_mb", 50))
    return max_mb * 1024 * 1024


@router.post("/image", response_model=VisionClassificationResponse)
async def analyze_image(
    file: Annotated[UploadFile, File(...)],
) -> VisionClassificationResponse:
    """Classify an uploaded image into one of 14 UCF crime categories."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {ALLOWED_IMAGE_TYPES}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(image_bytes) > _max_upload_bytes():
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds upload size limit.",
        )

    try:
        from anomaly_detection.domains.vision.video import predict_image

        manager = _get_model_manager()
        _, result = predict_image(image_bytes, manager)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return VisionClassificationResponse(media_type="image", **result)


@router.post("/video", response_model=VisionClassificationResponse)
async def analyze_video(
    file: Annotated[UploadFile, File(...)],
) -> VisionClassificationResponse:
    """Classify an uploaded video (sync stub: samples frames and averages predictions)."""
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {ALLOWED_VIDEO_TYPES}",
        )

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(video_bytes) > _max_upload_bytes():
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds upload size limit.",
        )

    try:
        from anomaly_detection.domains.vision.video import predict_video

        manager = _get_model_manager()
        vision_cfg = _get_vision_config()
        max_frames = int(vision_cfg.get("max_frames", 16))
        result = predict_video(video_bytes, manager, max_frames=max_frames)
    except HTTPException:
        raise
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return VisionClassificationResponse(media_type="video", **result)
