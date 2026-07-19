"""Detection API routes."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from anomaly_detection.api.schemas import BatchDetectResponse, DetectRequest, DetectResponse
from anomaly_detection.config.loader import load_config
from anomaly_detection.models.registry import DETECTOR_REGISTRY
from anomaly_detection.pipeline import _deep_merge, run_detection

router = APIRouter(tags=["detect"])

# Resolved relative to this file (repo_root/src/anomaly_detection/api/routes/detect.py)
# rather than the process CWD, so the API works no matter where it's launched from.
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_MAX_DETECT_ROWS = 100_000


def _default_config() -> dict[str, Any]:
    return load_config(DEFAULT_CONFIG_PATH)


def _max_detect_rows() -> int:
    """Row cap for /detect and /detect/batch, configurable via MAX_DETECT_ROWS."""
    raw = os.environ.get("MAX_DETECT_ROWS", "").strip()
    if not raw:
        return DEFAULT_MAX_DETECT_ROWS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_DETECT_ROWS
    return value if value > 0 else DEFAULT_MAX_DETECT_ROWS


def _check_row_limit(n_rows: int) -> None:
    max_rows = _max_detect_rows()
    if n_rows > max_rows:
        # 413 Content Too Large (status.HTTP_413_REQUEST_ENTITY_TOO_LARGE is
        # deprecated in newer Starlette; use the literal for broad compatibility).
        raise HTTPException(
            status_code=413,
            detail=f"data has {n_rows} rows, which exceeds MAX_DETECT_ROWS ({max_rows})",
        )


@router.get("/models")
def list_models() -> dict[str, list[str]]:
    return {"models": sorted(DETECTOR_REGISTRY)}


@router.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    config = _default_config()
    if request.config:
        config = _deep_merge(config, request.config)

    try:
        data = np.asarray(request.data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        _check_row_limit(data.shape[0])
        report = run_detection(config, data=data)
    except HTTPException:
        raise
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return DetectResponse(
        model=report.get("model"),
        n_samples=report["n_samples"],
        n_anomalies=report["n_anomalies"],
        scores=report["scores"],
        predictions=report["predictions"],
        metrics=report.get("metrics"),
    )


@router.post("/detect/batch", response_model=BatchDetectResponse)
async def detect_batch(
    file: Annotated[UploadFile, File(...)],
    config_yaml: str | None = None,
) -> BatchDetectResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV uploads are supported.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    config = _default_config()
    if config_yaml:
        try:
            override = yaml.safe_load(config_yaml)
            if not isinstance(override, dict):
                raise ValueError("config_yaml must deserialize to a mapping")
            config = _deep_merge(config, override)
        except (yaml.YAMLError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        frame = pd.read_csv(io.BytesIO(raw_bytes))
        if frame.empty:
            raise ValueError("CSV contains no rows")
        _check_row_limit(len(frame))

        target_column = config.get("dataset", {}).get("target_column")
        labels = None
        feature_frame = frame
        if target_column and target_column in frame.columns:
            labels = frame[target_column].to_numpy()
            feature_frame = frame.drop(columns=[target_column])

        data = feature_frame.to_numpy(dtype=float)
        report = run_detection(
            config,
            data=data,
            labels=labels,
            feature_names=list(feature_frame.columns),
        )
    except HTTPException:
        raise
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return BatchDetectResponse(
        model=report.get("model"),
        n_samples=report["n_samples"],
        n_anomalies=report["n_anomalies"],
        scores=report["scores"],
        predictions=report["predictions"],
        metrics=report.get("metrics"),
        feature_names=report.get("feature_names", []),
    )
