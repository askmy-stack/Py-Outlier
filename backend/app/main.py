"""
FastAPI application entry point.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, clean up on shutdown."""
    logger.info("startup", app=settings.app_name, version=settings.version, debug=settings.debug)

    # Initialise Sentry before anything else if configured
    if settings.sentry_dsn:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[StarletteIntegration(), FastApiIntegration()],
            traces_sample_rate=0.2,
        )
        logger.info("sentry_initialised")

    # Load TensorFlow models
    from app.ml.model import model_manager

    model_manager.load()
    logger.info("models_ready", image_loaded=model_manager._image_model is not None)

    yield

    logger.info("shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Anonymous tip-off platform with AI-powered anomaly detection. "
        "Classifies images and videos into 14 crime categories using DenseNet121."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
        client=request.client.host if request.client else "unknown",
    )
    return response


# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.analyze import router as analyze_router  # noqa: E402
from app.api.health import router as health_router  # noqa: E402

app.include_router(health_router)
app.include_router(analyze_router)


@app.get("/", tags=["root"])
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/health",
    }
