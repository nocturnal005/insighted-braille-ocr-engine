"""Uncertainty flag categories and helpers (fixed by the InsightEd AI contract)."""

from __future__ import annotations

from app.models.responses import Flag

CATEGORY_LOW_IMAGE_QUALITY = "low_image_quality"
CATEGORY_LOW_OCR_CONFIDENCE = "low_ocr_confidence"
CATEGORY_UNCLEAR_BRAILLE_CELL = "unclear_braille_cell"
CATEGORY_POSSIBLE_CONTRACTION_ISSUE = "possible_contraction_issue"
CATEGORY_POSSIBLE_NUMBER_SIGN_ISSUE = "possible_number_sign_issue"
CATEGORY_POSSIBLE_CAPITALISATION_ISSUE = "possible_capitalisation_issue"
CATEGORY_LINE_ORDER_UNCERTAINTY = "line_order_uncertainty"
CATEGORY_WORD_SPACING_UNCERTAINTY = "word_spacing_uncertainty"
CATEGORY_SUBJECT_SPECIFIC_TERM = "subject_specific_term"

ALL_CATEGORIES = (
    CATEGORY_LOW_IMAGE_QUALITY,
    CATEGORY_LOW_OCR_CONFIDENCE,
    CATEGORY_UNCLEAR_BRAILLE_CELL,
    CATEGORY_POSSIBLE_CONTRACTION_ISSUE,
    CATEGORY_POSSIBLE_NUMBER_SIGN_ISSUE,
    CATEGORY_POSSIBLE_CAPITALISATION_ISSUE,
    CATEGORY_LINE_ORDER_UNCERTAINTY,
    CATEGORY_WORD_SPACING_UNCERTAINTY,
    CATEGORY_SUBJECT_SPECIFIC_TERM,
)


def make_flag(text: str, reason: str, category: str, severity: str) -> Flag:
    return Flag(text=text, reason=reason, category=category, severity=severity)


def dedupe_flags(flags: list[Flag]) -> list[Flag]:
    seen: set[tuple[str, str, str]] = set()
    result: list[Flag] = []
    for flag in flags:
        key = (flag.category, flag.reason, flag.text)
        if key not in seen:
            seen.add(key)
            result.append(flag)
    return result
