"""API key enforcement for the OCR endpoint.

When OCR_ENGINE_API_KEY is set, POST /ocr requires the key in either:
  - X-API-Key: <key>
  - Authorization: Bearer <key>

InsightEd AI's external_braille_ocr adapter sends Authorization: Bearer, so
both forms are accepted. Comparison is constant-time. The key value is never
logged.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.core.config import get_settings


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    expected = get_settings().ocr_engine_api_key
    if not expected:
        return

    candidates = [c for c in (x_api_key, _bearer_token(authorization)) if c]
    for candidate in candidates:
        if secrets.compare_digest(candidate, expected):
            return

    raise HTTPException(status_code=401, detail="Invalid or missing API key.")
