"""GET /demo — optional local human demo page (Stage 3D-H1).

Serves a self-contained static HTML page that posts an image to the existing
/ocr endpoint from the same origin and renders the draft response with the
mandatory draft-only warning. Disabled by default: unless DEMO_PAGE_ENABLED
is true the route returns 404, so a deployed integration never exposes it.
The page changes no OCR logic and no contract — it is a viewer for /ocr.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.core.config import get_settings

router = APIRouter()

_DEMO_PAGE_PATH = Path(__file__).resolve().parent.parent / "demo" / "demo_page.html"


@lru_cache
def _demo_page_html() -> str:
    return _DEMO_PAGE_PATH.read_text(encoding="utf-8")


@router.get("/demo", response_class=HTMLResponse, tags=["demo"], include_in_schema=False)
def demo_page() -> HTMLResponse:
    if not get_settings().demo_page_enabled:
        raise HTTPException(
            status_code=404,
            detail=(
                "The local demo page is disabled. Set DEMO_PAGE_ENABLED=true "
                "and restart the service to enable it for local use."
            ),
        )
    return HTMLResponse(_demo_page_html())
