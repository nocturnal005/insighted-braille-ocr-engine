"""End-to-end pipeline tests on synthetic Braille images.

Synthetic renders are the ideal case; these tests prove the geometric
pipeline (dots -> cells -> lines -> Unicode Braille -> text) round-trips
correctly when the input is clean.
"""

from __future__ import annotations

import pytest

from app.evaluation.metrics import normalise_text
from app.models.requests import OcrRequest
from app.ocr.pipeline import run_ocr
from app.tests.helpers import make_payload


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "the cell membrane",
        "add 12 and 34",
        "Year 10 Physics",
        "light travels\nin straight lines",
    ],
)
def test_synthetic_round_trip(text):
    response = run_ocr(OcrRequest(**make_payload(text)))
    assert normalise_text(response.draftText) == normalise_text(text)
    assert response.confidence > 0.5
    assert response.rawBraille
    assert response.rawCells
    assert len(response.pageResults) == 1


def test_single_letter_image():
    response = run_ocr(OcrRequest(**make_payload("a")))
    assert normalise_text(response.draftText) == "a"
    assert response.rawCells


def test_blank_image_returns_empty_draft_with_flags():
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (400, 200), color=255).save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    response = run_ocr(OcrRequest(**make_payload(dataUrl=data_url)))
    assert response.draftText == ""
    assert response.confidence == 0.0
    assert response.flags
