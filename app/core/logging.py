"""Logging setup.

Safety rule for every log line in this service: log request metadata only
(request ids, byte counts, durations, dot/cell counts, confidence, flag
counts). Task ids may appear only as a short non-reversible hash. Never log
image data, base64 payloads, transcription text, file names, task titles,
raw task ids, pupil data, or API keys.
"""

from __future__ import annotations

import logging

from app.core.config import get_settings

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
