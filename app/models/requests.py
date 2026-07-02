"""Request models. The shape mirrors the InsightEd AI external_braille_ocr adapter."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    taskId: str
    title: str
    fileName: str
    mimeType: str
    dataUrl: str
    subject: str | None = None
    yearGroup: str | None = None
