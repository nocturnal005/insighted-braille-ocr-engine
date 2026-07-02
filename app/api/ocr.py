"""POST /ocr — draft-only Braille OCR.

Always returns the exact InsightEd AI contract shape. Failures return
controlled JSON (empty draftText, confidence 0, uncertainty flags) rather
than errors or fabricated text.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import require_api_key
from app.models.requests import OcrRequest
from app.models.responses import OcrResponse
from app.ocr.pipeline import run_ocr

router = APIRouter()


@router.post(
    "/ocr",
    response_model=OcrResponse,
    tags=["ocr"],
    summary="Draft Braille OCR (requires downstream specialist verification)",
)
def ocr(request: OcrRequest, _: None = Depends(require_api_key)) -> OcrResponse:
    return run_ocr(request)
