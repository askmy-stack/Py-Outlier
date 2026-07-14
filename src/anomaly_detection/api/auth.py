"""Optional API key authentication for the anomaly detection service."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

API_KEY_HEADER = "X-API-Key"
API_KEY_ENV_VAR = "ANOMALY_API_KEY"

# Paths that never require an API key, regardless of configuration.
PUBLIC_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


def _configured_api_key() -> str:
    return os.environ.get(API_KEY_ENV_VAR, "").strip()


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Require ``X-API-Key`` when ``ANOMALY_API_KEY`` is configured.

    The service runs open (no auth) when ``ANOMALY_API_KEY`` is unset, which
    matches local/dev usage. Set ``ANOMALY_API_KEY`` to require a matching
    ``X-API-Key`` header on every request except the public paths above.
    """

    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable[Response]]) -> Response:
        configured = _configured_api_key()
        if not configured or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if request.headers.get(API_KEY_HEADER) == configured:
            return await call_next(request)

        return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})
