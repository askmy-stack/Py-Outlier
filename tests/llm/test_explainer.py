"""Tests for LLM anomaly explainer."""

from __future__ import annotations

from pathlib import Path

from anomaly_detection.llm.explainer import AnomalyExplainer, explain_anomaly
from anomaly_detection.llm.redaction import redact_pii

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "configs" / "prompts" / "anomaly_report.yaml"


def test_redact_pii_masks_common_patterns() -> None:
    text = "Contact alice@example.com or 555-123-4567, SSN 123-45-6789"
    redacted = redact_pii(text)
    assert "alice@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_SSN]" in redacted


def test_explainer_returns_template_report_when_disabled() -> None:
    explainer = AnomalyExplainer(enabled=False, prompt_path=PROMPT_PATH)
    report = explainer.explain(
        scores=[0.1, 0.9, 0.2],
        predictions=[0, 1, 0],
        feature_names=["latency", "errors", "cpu"],
        model_name="isolation_forest",
    )

    assert report["source"] == "template"
    assert "1 anomal" in report["summary"]
    assert report["llm_response"] == report["summary"]
    assert len(report["remediation_steps"]) >= 1
    assert "sample 1" in report["context"]


def test_explain_anomaly_reads_config_defaults() -> None:
    config = {
        "llm": {
            "enabled": False,
            "prompt_path": str(PROMPT_PATH),
        }
    }
    report = explain_anomaly(
        scores=[0.5, 0.8],
        predictions=[0, 1],
        config=config,
        model_name="zscore",
    )
    assert report["source"] == "template"
    assert "anomal" in report["summary"].lower()
