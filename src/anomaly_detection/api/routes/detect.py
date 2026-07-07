"""Detection API routes."""

from __future__ import annotations

import io
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

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def _default_config() -> dict[str, Any]:
    return load_config(DEFAULT_CONFIG_PATH)


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
        report = run_detection(config, data=data)
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
