"""Pydantic schemas for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ModelsResponse(BaseModel):
    models: list[str]


class DetectRequest(BaseModel):
    data: list[list[float]] = Field(..., min_length=1)
    config: dict[str, Any] | None = None


class DetectResponse(BaseModel):
    model: str | None
    n_samples: int
    n_anomalies: int
    scores: list[float]
    predictions: list[int]
    metrics: dict[str, float] | None = None


class BatchDetectResponse(DetectResponse):
    feature_names: list[str] = Field(default_factory=list)


class RootCauseItem(BaseModel):
    metric: str
    score: float
    rank: int


class RootCauseRequest(BaseModel):
    anomaly_id: str = Field(..., min_length=1)
    metrics: dict[str, list[float]] = Field(
        ...,
        description="Metric time series as column name -> values.",
        min_length=1,
    )
    timestamp: int | str | None = Field(
        default=None,
        description="Anomaly timestamp (row index or index label). Defaults to last row.",
    )
    top_k: int = Field(default=3, ge=1, le=20)


class RootCauseResponse(BaseModel):
    anomaly_id: str
    causes: list[RootCauseItem]
    graph: dict[str, Any]
    method: str
    disclaimer: str
