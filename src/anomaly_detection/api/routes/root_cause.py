"""Root cause analysis API routes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from anomaly_detection.api.schemas import RootCauseItem, RootCauseRequest, RootCauseResponse
from anomaly_detection.rca.scorer import analyze_root_cause

router = APIRouter(tags=["root_cause"])

# In-memory session cache: anomaly_id -> RCA result.
# Not persisted across restarts or shared across workers.
_rca_cache: dict[str, dict[str, Any]] = {}


def _metrics_to_dataframe(metrics: dict[str, list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics)
    if frame.empty:
        raise ValueError("metrics must contain at least one row")
    if frame.shape[1] < 2:
        raise ValueError("metrics must contain at least two metric columns")
    return frame


@router.post("/root_cause", response_model=RootCauseResponse)
def post_root_cause(request: RootCauseRequest) -> RootCauseResponse:
    try:
        frame = _metrics_to_dataframe(request.metrics)
        result = analyze_root_cause(frame, timestamp=request.timestamp, top_k=request.top_k)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    response = RootCauseResponse(
        anomaly_id=request.anomaly_id,
        causes=[RootCauseItem(**item) for item in result["causes"]],
        graph=result["graph"],
        method=result["method"],
        disclaimer=result["disclaimer"],
    )
    _rca_cache[request.anomaly_id] = response.model_dump()
    return response


@router.get("/root_cause/{anomaly_id}", response_model=RootCauseResponse)
def get_root_cause(anomaly_id: str) -> RootCauseResponse:
    cached = _rca_cache.get(anomaly_id)
    if cached is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No cached RCA result for anomaly_id '{anomaly_id}'.",
        )
    return RootCauseResponse(**cached)
