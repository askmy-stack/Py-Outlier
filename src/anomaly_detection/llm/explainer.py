"""Anomaly explanation via templates and optional Anthropic LLM calls."""

from __future__ import annotations

import os
from pathlib import Path
from string import Template
from typing import Any

import yaml

from anomaly_detection.llm.redaction import redact_pii

DEFAULT_PROMPT_PATH = Path("configs/prompts/anomaly_report.yaml")


def _load_prompt_template(path: Path | str = DEFAULT_PROMPT_PATH) -> dict[str, Any]:
    prompt_path = Path(path)
    with prompt_path.open(encoding="utf-8") as handle:
        prompt_config = yaml.safe_load(handle)
    if not isinstance(prompt_config, dict):
        raise ValueError(f"Expected mapping in prompt file: {prompt_path}")
    return prompt_config


def _format_feature_context(
    scores: list[float],
    predictions: list[int],
    feature_names: list[str] | None,
    feature_values: list[float] | None,
) -> str:
    lines: list[str] = []
    for index, (score, prediction) in enumerate(zip(scores, predictions, strict=False)):
        status = "anomaly" if prediction == 1 else "normal"
        feature_part = ""
        if feature_values is not None and index < len(feature_values):
            name = feature_names[index] if feature_names and index < len(feature_names) else f"feature_{index}"
            feature_part = f", {name}={feature_values[index]:.4f}"
        lines.append(f"sample {index}: score={score:.4f}, status={status}{feature_part}")
    return "\n".join(lines)


def _render_template_report(
    prompt_config: dict[str, Any],
    *,
    scores: list[float],
    predictions: list[int],
    feature_names: list[str] | None = None,
    feature_values: list[float] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    n_anomalies = sum(predictions)
    context = _format_feature_context(scores, predictions, feature_names, feature_values)
    context = redact_pii(context)

    system_template = Template(prompt_config.get("system", "You are an anomaly detection assistant."))
    user_template = Template(
        prompt_config.get(
            "user",
            "Summarize anomalies for model $model_name.\n$context",
        )
    )

    summary = (
        f"Detected {n_anomalies} anomal{'y' if n_anomalies == 1 else 'ies'} "
        f"across {len(scores)} samples using {model_name or 'unknown model'}."
    )
    remediation = prompt_config.get(
        "remediation_steps",
        [
            "Review flagged samples and validate sensor or transaction integrity.",
            "Compare anomaly scores against historical baselines.",
            "Escalate persistent anomalies to the on-call engineer.",
        ],
    )

    return {
        "summary": summary,
        "context": context,
        "remediation_steps": list(remediation),
        "system_prompt": system_template.safe_substitute(model_name=model_name or "unknown"),
        "user_prompt": user_template.safe_substitute(
            model_name=model_name or "unknown",
            context=context,
            n_anomalies=n_anomalies,
            n_samples=len(scores),
        ),
        "source": "template",
    }


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    max_tokens: int,
) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic SDK not installed. Install with: pip install -e '.[llm]'"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text_blocks = [block.text for block in message.content if hasattr(block, "text")]
    return "\n".join(text_blocks).strip()


class AnomalyExplainer:
    """Generate plain-language anomaly reports from detection results."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        prompt_path: Path | str = DEFAULT_PROMPT_PATH,
        model: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 1024,
    ) -> None:
        self.enabled = enabled
        self.prompt_path = Path(prompt_path)
        self.model = model
        self.max_tokens = max_tokens

    def explain(
        self,
        *,
        scores: list[float],
        predictions: list[int],
        feature_names: list[str] | None = None,
        feature_values: list[float] | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        prompt_config = _load_prompt_template(self.prompt_path)
        report = _render_template_report(
            prompt_config,
            scores=scores,
            predictions=predictions,
            feature_names=feature_names,
            feature_values=feature_values,
            model_name=model_name,
        )

        if not self.enabled:
            report["llm_response"] = report["summary"]
            return report

        try:
            llm_response = _call_anthropic(
                report["system_prompt"],
                report["user_prompt"],
                model=self.model,
                max_tokens=self.max_tokens,
            )
            report["llm_response"] = redact_pii(llm_response)
            report["source"] = "anthropic"
        except Exception as exc:
            report["llm_response"] = report["summary"]
            report["source"] = "template_fallback"
            report["error"] = str(exc)

        return report


def explain_anomaly(
    *,
    scores: list[float],
    predictions: list[int],
    config: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    feature_values: list[float] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper that reads LLM settings from a config mapping."""
    llm_config = (config or {}).get("llm", {})
    explainer = AnomalyExplainer(
        enabled=bool(llm_config.get("enabled", False)),
        prompt_path=llm_config.get("prompt_path", DEFAULT_PROMPT_PATH),
        model=llm_config.get("model", "claude-3-5-haiku-20241022"),
        max_tokens=int(llm_config.get("max_tokens", 1024)),
    )
    return explainer.explain(
        scores=scores,
        predictions=predictions,
        feature_names=feature_names,
        feature_values=feature_values,
        model_name=model_name,
    )
