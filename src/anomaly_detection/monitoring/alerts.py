"""Alert hooks for streaming anomaly detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SlackAlerter:
    """Stub Slack webhook alerter; disabled by default."""

    threshold: float = 3.0
    webhook_url: str | None = None
    enabled: bool = False

    def check(self, score: float) -> bool:
        """Return True when an alert would fire for the given score."""
        if not self.enabled or score <= self.threshold:
            return False
        self._send_alert(score)
        return True

    def _send_alert(self, score: float) -> None:
        if self.webhook_url:
            logger.warning(
                "Slack alert stub: score=%.4f exceeds threshold=%.4f (webhook=%s)",
                score,
                self.threshold,
                self.webhook_url,
            )
        else:
            logger.warning(
                "Slack alert stub: score=%.4f exceeds threshold=%.4f (no webhook configured)",
                score,
                self.threshold,
            )
