"""FastAPI application entry point."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from anomaly_detection.api.routes.detect import router as detect_router
from anomaly_detection.api.routes.explain import router as explain_router
from anomaly_detection.api.routes.root_cause import router as root_cause_router
from anomaly_detection.api.schemas import HealthResponse

try:
    import tensorflow  # noqa: F401

    from anomaly_detection.api.routes.vision import router as vision_router

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

APP_NAME = "anomaly-detection"
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Tabular anomaly detection service with pluggable detectors.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.monotonic() - start) * 1000:.2f}"
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


app.include_router(detect_router)
app.include_router(explain_router)
app.include_router(root_cause_router)
if _VISION_AVAILABLE:
    app.include_router(vision_router)


def run() -> None:
    """Run the API with uvicorn (sync fit only; no background jobs in Phase 3)."""
    import uvicorn

    uvicorn.run("anomaly_detection.api.app:app", host="0.0.0.0", port=8000, reload=False)
