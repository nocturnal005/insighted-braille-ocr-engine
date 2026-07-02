"""Shared test helpers: synthetic Braille request payloads."""

from __future__ import annotations

from app.evaluation.sample_generator import image_to_data_url, render_braille_image

EXPECTED_RESPONSE_KEYS = {
    "draftText",
    "confidence",
    "rawBraille",
    "rawCells",
    "providerRequestId",
    "flags",
    "pageResults",
}

EXPECTED_PAGE_KEYS = {"pageNumber", "text", "confidence", "flags"}
EXPECTED_FLAG_KEYS = {"text", "reason", "category", "severity"}
EXPECTED_RAW_CELL_KEYS = {"line", "cellIndex", "dots", "bbox", "confidence"}


def make_data_url(text: str = "hello world") -> str:
    return image_to_data_url(render_braille_image(text))


def make_payload(text: str = "hello world", **overrides) -> dict:
    payload = {
        "taskId": "task-test-001",
        "title": "Test Braille upload",
        "fileName": "test-page.png",
        "mimeType": "image/png",
        "dataUrl": make_data_url(text),
        "subject": None,
        "yearGroup": None,
    }
    payload.update(overrides)
    return payload
