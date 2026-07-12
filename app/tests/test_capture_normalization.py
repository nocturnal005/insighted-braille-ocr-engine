"""Capture normalisation tests (Stage 3D-L1).

Real phone photos arrive oversized, rotated, and on cluttered backgrounds -
each was a total (0-cells) failure in the 2026-07-12 real-capture diagnostic.
These tests prove the normalisation layer fixes the synthetic equivalent of
each failure mode without altering inputs that already decode.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from app.evaluation.metrics import normalise_text
from app.evaluation.sample_generator import image_to_data_url, render_braille_image
from app.models.requests import OcrRequest
from app.ocr.capture_normalization import (
    MAX_LONG_SIDE,
    TARGET_LONG_SIDE,
    crop_to_bright_region,
    normalise_scale,
)
from app.ocr.flags import CATEGORY_LINE_ORDER_UNCERTAINTY, CATEGORY_LOW_IMAGE_QUALITY
from app.ocr.pipeline import run_ocr

TEXT = "light travels\nin straight lines"


def _request(image: Image.Image) -> OcrRequest:
    return OcrRequest(
        taskId="task-capture-norm",
        title="capture normalisation test",
        fileName="test-page.png",
        mimeType="image/png",
        dataUrl=image_to_data_url(image),
    )


def _gray(text: str = TEXT) -> np.ndarray:
    return np.array(render_braille_image(text).convert("L"))


# --- normalise_scale unit behaviour ----------------------------------------


def test_small_images_pass_through_unchanged():
    gray = _gray()
    out, applied = normalise_scale(gray)
    assert not applied
    assert out is gray  # identity, not a copy: decoding inputs are untouched


def test_oversized_image_downscaled_to_target():
    gray = _gray()
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    assert max(big.shape) > MAX_LONG_SIDE
    out, applied = normalise_scale(big)
    assert applied
    assert max(out.shape) == TARGET_LONG_SIDE


# --- crop_to_bright_region unit behaviour -----------------------------------


def test_crop_finds_bright_page_on_dark_background():
    page = np.full((200, 300), 235, dtype=np.uint8)
    canvas = np.full((600, 800), 60, dtype=np.uint8)
    canvas[150:350, 250:550] = page
    cropped = crop_to_bright_region(canvas)
    assert cropped is not None
    # Bright region recovered to within the crop margin.
    assert abs(cropped.shape[0] - 200) <= 20
    assert abs(cropped.shape[1] - 300) <= 20


def test_crop_returns_none_when_page_fills_frame():
    assert crop_to_bright_region(np.full((300, 400), 235, dtype=np.uint8)) is None


# --- end-to-end: the three real-capture failure modes -----------------------


def test_oversized_photo_decodes_with_downscale_flag():
    gray = _gray()
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    response = run_ocr(_request(Image.fromarray(big)))
    assert normalise_text(response.draftText) == normalise_text(TEXT)
    assert any(
        f.category == CATEGORY_LOW_IMAGE_QUALITY and "downscaled" in f.reason
        for f in response.flags
    )


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_rotated_photo_decodes_with_rotation_flag(quarter_turns):
    rotated = Image.fromarray(np.rot90(_gray(), quarter_turns))
    response = run_ocr(_request(rotated))
    assert normalise_text(response.draftText) == normalise_text(TEXT)
    assert any(
        f.category == CATEGORY_LINE_ORDER_UNCERTAINTY and "rotation" in f.reason
        for f in response.flags
    )


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_oversized_and_rotated_photo_decodes(quarter_turns):
    gray = _gray()
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    response = run_ocr(_request(Image.fromarray(np.rot90(big, quarter_turns))))
    assert normalise_text(response.draftText) == normalise_text(TEXT)


def test_upright_decode_is_unaffected_and_unflagged():
    response = run_ocr(_request(render_braille_image(TEXT)))
    assert normalise_text(response.draftText) == normalise_text(TEXT)
    assert response.confidence > 0.5
    assert not any(
        "rotation" in f.reason or "downscaled" in f.reason for f in response.flags
    )


def test_small_genuine_decode_survives_the_ladder():
    # A single word forms fewer cells than _MIN_PLAUSIBLE_CELLS, so the
    # rescue ladder runs - the upright decode must win as the incumbent.
    response = run_ocr(_request(render_braille_image("hello")))
    assert normalise_text(response.draftText) == "hello"


def test_blank_image_still_fails_safely():
    blank = Image.fromarray(np.full((400, 600), 255, dtype=np.uint8))
    response = run_ocr(_request(blank))
    assert response.draftText == ""
    assert response.confidence == 0.0
    assert response.flags
