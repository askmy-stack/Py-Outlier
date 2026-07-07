"""Anomaly explanation API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from anomaly_detection.api.schemas import ExplainRequest, ExplainResponse
from anomaly_detection.config.loader import load_config
from anomaly_detection.llm.explainer import explain_anomaly
from anomaly_detection.pipeline import _deep_merge

router = APIRouter(tags=["explain"])

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def _default_config() -> dict:
    return load_config(DEFAULT_CONFIG_PATH)


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    if len(request.scores) != len(request.predictions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scores and predictions must have the same length",
        )

    config = _default_config()
    if request.config:
        config = _deep_merge(config, request.config)

    try:
        report = explain_anomaly(
            scores=request.scores,
            predictions=request.predictions,
            config=config,
            feature_names=request.feature_names,
            feature_values=request.feature_values,
            model_name=request.model_name,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ExplainResponse(
        summary=report["summary"],
        remediation_steps=report["remediation_steps"],
        llm_response=report["llm_response"],
        source=report["source"],
        context=report.get("context"),
        error=report.get("error"),
    )
