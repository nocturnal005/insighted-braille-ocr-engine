"""GET /version — service identity plus the mandatory draft-only warning."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.responses import VersionResponse
from app.ocr.image_decode import SUPPORTED_MIME_TYPES

router = APIRouter()


@router.get("/version", response_model=VersionResponse, tags=["meta"])
def version() -> VersionResponse:
    settings = get_settings()
    return VersionResponse(
        name=settings.service_name,
        version=settings.service_version,
        apiVersion=settings.api_version,
        supportedMimeTypes=list(SUPPORTED_MIME_TYPES),
        warning=settings.draft_warning,
    )
