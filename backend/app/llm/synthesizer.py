"""
LLM-powered incident report synthesis via the Claude API.

Combines the ML model's classification output with the user's free-text
tip description to produce a structured IncidentReport. This demonstrates
when LLMs are more appropriate than classical ML: structured extraction
and synthesis from unstructured text, not classification.
"""

from __future__ import annotations

import json

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import IncidentReport

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an AI assistant that helps law enforcement analysts triage anonymous tip submissions.
You will be given:
1. The text of an anonymous tip describing suspicious activity
2. The output of an AI vision model that classified any attached media

Your job is to produce a structured incident analysis in JSON format.
Be factual and concise. Do not speculate beyond what the text and model output indicate.
If information is not available, use "unknown" or an empty list.

Output ONLY valid JSON. No markdown, no explanation — just the JSON object.

JSON schema:
{
  "summary": "string — 1-3 sentence factual summary of the incident",
  "threat_level": integer 1-5 (1=low/normal, 5=critical/violent),
  "location_indicators": ["list of location clues mentioned in the text"],
  "time_references": ["list of time/date references mentioned"],
  "actor_count": "string — estimated number of individuals involved or 'unknown'",
  "ml_consistency": "string — whether the ML classification is consistent with the text, and why",
  "recommendation": "string — suggested next action for a human reviewer"
}
"""


def _default_report(reason: str) -> IncidentReport:
    return IncidentReport(
        summary=f"Automated synthesis unavailable: {reason}",
        threat_level=3,
        location_indicators=[],
        time_references=[],
        actor_count="unknown",
        ml_consistency="Unable to assess — synthesis failed.",
        recommendation="Manual review required.",
    )


class IncidentSynthesizer:
    async def synthesize(
        self,
        text: str,
        ml_class: str,
        confidence: float,
        classes: list[str],
    ) -> IncidentReport:
        """
        Generate a structured incident report using Claude.

        Args:
            text: The user's free-text tip description.
            ml_class: The top predicted class from the ML model.
            confidence: Confidence score (0–1) for the top class.
            classes: All 14 class names for context.

        Returns:
            IncidentReport with structured fields.
        """
        if not settings.anthropic_api_key:
            logger.warning("anthropic_api_key_not_set", action="skipping_synthesis")
            return _default_report("ANTHROPIC_API_KEY not configured.")

        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=settings.anthropic_api_key)

            user_message = f"""\
Anonymous tip text:
\"\"\"
{text}
\"\"\"

ML vision model result:
- Predicted class: {ml_class}
- Confidence: {confidence:.1%}
- All possible classes: {", ".join(classes)}

Produce the structured incident report JSON."""

            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data = json.loads(raw)
            report = IncidentReport(**data)
            logger.info(
                "incident_report_synthesized",
                ml_class=ml_class,
                confidence=confidence,
                threat_level=report.threat_level,
            )
            return report

        except json.JSONDecodeError as exc:
            logger.error("synthesis_json_parse_error", error=str(exc))
            return _default_report(f"JSON parse error: {exc}")
        except Exception as exc:
            logger.error("synthesis_error", error=str(exc))
            return _default_report(str(exc))


synthesizer = IncidentSynthesizer()
