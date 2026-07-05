"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import demo, health, ocr, version
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    description=(
        "Standalone draft-only Braille OCR engine for InsightEd AI. "
        "AI drafts, humans verify: all output is an unverified draft and "
        "requires QTVI or Braille-literate specialist verification before "
        "use in teacher feedback or export."
    ),
)

app.include_router(health.router)
app.include_router(version.router)
app.include_router(ocr.router)
app.include_router(demo.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log only the path and exception type — never request bodies or image data.
    logger.error(
        "unhandled_error path=%s type=%s", request.url.path, type(exc).__name__
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "Unexpected server error. No image or transcription data was logged.",
        },
    )
