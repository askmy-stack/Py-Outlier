from fastapi import APIRouter

from app.core.config import settings
from app.ml.model import model_manager

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": settings.version,
        "models_loaded": model_manager.is_ready,
    }
