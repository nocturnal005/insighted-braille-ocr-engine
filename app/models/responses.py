"""Response models. The OcrResponse shape is the fixed InsightEd AI contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

FlagCategory = Literal[
    "low_image_quality",
    "low_ocr_confidence",
    "unclear_braille_cell",
    "possible_contraction_issue",
    "possible_number_sign_issue",
    "possible_capitalisation_issue",
    "line_order_uncertainty",
    "word_spacing_uncertainty",
    "subject_specific_term",
]

FlagSeverity = Literal["low", "medium", "high"]


class Flag(BaseModel):
    text: str
    reason: str
    category: FlagCategory
    severity: FlagSeverity


class RawCell(BaseModel):
    line: int
    cellIndex: int
    dots: list[int]
    bbox: list[int]
    confidence: float


class PageResult(BaseModel):
    pageNumber: int
    text: str
    confidence: float
    flags: list[Flag]


class OcrResponse(BaseModel):
    draftText: str
    confidence: float
    rawBraille: str | None
    rawCells: list[RawCell]
    providerRequestId: str
    flags: list[Flag]
    pageResults: list[PageResult]


class HealthResponse(BaseModel):
    status: str
    service: str


class VersionResponse(BaseModel):
    name: str
    version: str
    apiVersion: str
    supportedMimeTypes: list[str]
    warning: str
