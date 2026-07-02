"""GET /health — liveness probe."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=get_settings().service_name)
