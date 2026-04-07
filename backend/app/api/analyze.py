"""
/analyze endpoints — image and video anomaly detection.
"""

from __future__ import annotations

import base64
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.synthesizer import synthesizer
from app.ml.gradcam import compute_gradcam, overlay_heatmap
from app.ml.model import model_manager
from app.ml.preprocess import extract_frames, preprocess_image
from app.models.schemas import AnalysisResponse, ClassificationResult, JobStatus

logger = get_logger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}


def _build_classification(probs, image_bytes: bytes | None = None) -> ClassificationResult:
    """Build a ClassificationResult from a probability array, optionally with Grad-CAM."""
    top_idx = int(probs.argmax())
    top_class = settings.classes[top_idx]
    confidence = float(probs[top_idx])
    all_scores = {cls: float(probs[i]) for i, cls in enumerate(settings.classes)}

    heatmap_url = None
    # Grad-CAM requires a Keras Model with get_layer; gracefully skip for SavedModels
    if image_bytes is not None:
        try:
            import tensorflow as tf
            from app.ml.model import model_manager as mm

            if hasattr(mm._image_model, "get_layer"):
                heatmap = compute_gradcam(mm._image_model, probs.reshape(1, -1), top_idx)
                if heatmap is not None:
                    overlay_bytes = overlay_heatmap(image_bytes, heatmap)
                    # In production, upload overlay_bytes to S3/R2 and return URL.
                    # For local dev, return as base64 data URI.
                    b64 = base64.b64encode(overlay_bytes).decode()
                    heatmap_url = f"data:image/png;base64,{b64}"
        except Exception as exc:
            logger.warning("gradcam_skipped", reason=str(exc))

    return ClassificationResult(
        predicted_class=top_class,
        confidence=confidence,
        all_scores=all_scores,
        heatmap_url=heatmap_url,
    )


@router.post("/image", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_image(
    file: UploadFile,
    message: Annotated[str, Form(max_length=1000)] = "",
) -> AnalysisResponse:
    """
    Classify an uploaded image for anomaly/crime type.

    - Validates MIME type and file size
    - Runs DenseNet121 inference
    - Attempts Grad-CAM heatmap generation
    - If ANTHROPIC_API_KEY is set, synthesizes an IncidentReport via Claude
    """
    start = time.monotonic()
    submission_id = str(uuid.uuid4())

    # Validate MIME type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {ALLOWED_IMAGE_TYPES}",
        )

    # Read and validate size
    image_bytes = await file.read()
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
        )

    # Preprocess and infer
    try:
        image_array = preprocess_image(image_bytes)
        probs = model_manager.predict_image(image_array)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.error("image_inference_error", submission_id=submission_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Please try again.",
        )

    classification = _build_classification(probs, image_bytes)

    # LLM incident report synthesis (optional)
    incident_report = None
    if message and settings.anthropic_api_key:
        incident_report = await synthesizer.synthesize(
            text=message,
            ml_class=classification.predicted_class,
            confidence=classification.confidence,
            classes=settings.classes,
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "image_analyzed",
        submission_id=submission_id,
        predicted_class=classification.predicted_class,
        confidence=classification.confidence,
        duration_ms=elapsed_ms,
    )

    return AnalysisResponse(
        submission_id=submission_id,
        media_type="image",
        classification=classification,
        incident_report=incident_report,
        processing_time_ms=elapsed_ms,
    )


@router.post("/video", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
async def analyze_video(file: UploadFile) -> JobStatus:
    """
    Enqueue a video for async analysis via Celery.

    Returns a job_id immediately. Poll GET /analyze/job/{job_id} for results.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {ALLOWED_VIDEO_TYPES}",
        )

    video_bytes = await file.read()
    if len(video_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
        )

    submission_id = str(uuid.uuid4())

    # Enqueue Celery task (video bytes serialized as base64 for JSON transport)
    try:
        from app.workers.tasks import analyze_video_task

        video_b64 = base64.b64encode(video_bytes).decode()
        task = analyze_video_task.delay(video_b64, submission_id)
        job_id = task.id
    except Exception as exc:
        logger.error("video_enqueue_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue unavailable. Please try again later.",
        )

    logger.info("video_enqueued", submission_id=submission_id, job_id=job_id)
    return JobStatus(job_id=job_id, status="queued")


@router.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Poll the status of a video analysis job."""
    try:
        from celery.result import AsyncResult

        from app.workers.tasks import celery_app

        result = AsyncResult(job_id, app=celery_app)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    if result.state == "PENDING":
        # PENDING in Celery means either truly queued OR unknown ID
        # We treat unknown IDs as not-found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    if result.state == "STARTED":
        return JobStatus(job_id=job_id, status="processing")

    if result.state == "SUCCESS":
        data = result.result
        analysis = AnalysisResponse(**data)
        return JobStatus(job_id=job_id, status="complete", result=analysis)

    if result.state == "FAILURE":
        return JobStatus(job_id=job_id, status="failed", error=str(result.result))

    return JobStatus(job_id=job_id, status="processing")
