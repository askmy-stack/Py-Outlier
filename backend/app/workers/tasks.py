"""
Celery tasks for async video processing.

Video inference is significantly slower than image inference (frame extraction +
per-frame model calls). Offloading to a Celery worker queue keeps the HTTP
response time fast and prevents API server timeouts.
"""

from __future__ import annotations

import base64
import uuid

from celery import Celery

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

celery_app = Celery(
    "anonmaly_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker (video is heavy)
    result_expires=3600,  # results expire after 1 hour
)


@celery_app.task(bind=True, name="analyze_video", max_retries=2, queue="video")
def analyze_video_task(self, video_bytes_b64: str, submission_id: str) -> dict:
    """
    Async video analysis task.

    Args:
        video_bytes_b64: Base64-encoded video bytes (JSON-serializable)
        submission_id: UUID string for correlation logging

    Returns:
        Serializable dict matching the AnalysisResponse structure.
    """
    import time

    logger.info("video_task_started", submission_id=submission_id, task_id=self.request.id)
    start = time.monotonic()

    try:
        # Lazy imports — avoid loading TF at module import time
        from app.ml.model import model_manager
        from app.ml.preprocess import extract_frames
        from app.core.config import settings as cfg

        video_bytes = base64.b64decode(video_bytes_b64)
        frames = extract_frames(video_bytes, max_frames=16)
        probs = model_manager.predict_video(frames)

        top_idx = int(probs.argmax())
        top_class = cfg.classes[top_idx]
        confidence = float(probs[top_idx])
        all_scores = {cls: float(probs[i]) for i, cls in enumerate(cfg.classes)}

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "video_task_complete",
            submission_id=submission_id,
            predicted_class=top_class,
            confidence=confidence,
            duration_ms=elapsed_ms,
        )

        return {
            "submission_id": submission_id,
            "media_type": "video",
            "classification": {
                "predicted_class": top_class,
                "confidence": confidence,
                "all_scores": all_scores,
                "heatmap_url": None,
            },
            "incident_report": None,
            "processing_time_ms": elapsed_ms,
        }

    except Exception as exc:
        logger.error("video_task_failed", submission_id=submission_id, error=str(exc))
        raise self.retry(exc=exc, countdown=5)
