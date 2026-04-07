from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_scores: dict[str, float]
    heatmap_url: str | None = None


class IncidentReport(BaseModel):
    summary: str
    threat_level: int = Field(ge=1, le=5)
    location_indicators: list[str]
    time_references: list[str]
    actor_count: str
    ml_consistency: str
    recommendation: str


class AnalysisResponse(BaseModel):
    submission_id: str
    media_type: Literal["image", "video"]
    classification: ClassificationResult
    incident_report: IncidentReport | None = None
    processing_time_ms: float


class SubmissionCreate(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    recaptcha_token: str = Field(min_length=1)


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "complete", "failed"]
    result: AnalysisResponse | None = None
    error: str | None = None
