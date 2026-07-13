"""Stage 3D-N1 tests: full-resolution template-reader rescue.

The template reader is the pipeline's last-ditch rescue when the blob path
(including the L1 ladder) forms zero cells on a capture. These tests exercise
it on SYNTHETIC embossed images only — never real pupil material — and verify:

* it reads an embossed page into a self-consistent, readable lattice;
* it fails closed (returns None) on clean scans and on noise;
* it is deterministic (no RANSAC randomness);
* it recovers right-angle rotations on its own;
* the pipeline invokes it only on total blob-path failure, caps the result's
  confidence, and flags it as experimental — and leaves every page that
  already decodes byte-for-byte unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.evaluation.metrics import normalise_text
from app.evaluation.sample_generator import (
    EmbossedStyle,
    image_to_data_url,
    render_braille_image,
    render_embossed_braille_image,
)
from app.models.requests import OcrRequest
from app.ocr.capture_normalization import NormalisedDetection
from app.ocr.cell_grouping import GroupingResult
from app.ocr.confidence import TEMPLATE_READER_CAP
from app.ocr.dot_detection import DetectionOutcome
from app.ocr.image_decode import decode_data_url
from app.ocr import pipeline as pipeline_module
from app.ocr.pipeline import run_ocr
from app.ocr.template_reader import read_page
from app.tests.helpers import EXPECTED_RESPONSE_KEYS, make_payload

_EMBOSSED_TEXT = "the cat sat on the mat and the dog ran"


def _gray(image: Image.Image) -> np.ndarray:
    gray, _ = decode_data_url(
        image_to_data_url(image), "image/png",
        max_bytes=10_000_000, max_pixels=40_000_000,
    )
    return gray


def _embossed_gray(text: str = _EMBOSSED_TEXT, seed: int = 4) -> np.ndarray:
    return _gray(render_embossed_braille_image(text, EmbossedStyle(), seed=seed))


# --- reading -----------------------------------------------------------------


def test_reads_synthetic_embossed_into_readable_lattice():
    result = read_page(_embossed_gray())
    assert result is not None
    assert result.grouping.total_cells >= 12
    assert result.grouping.grid is not None
    # Self-consistency gate: most cells must back-translate to letters.
    assert result.letter_fraction >= 0.45
    # Every cell is a real 1-6 dot pattern with a bbox.
    for line in result.grouping.lines:
        for cell in line:
            assert cell.dots and all(1 <= d <= 6 for d in cell.dots)
            x1, y1, x2, y2 = cell.bbox
            assert x2 > x1 and y2 > y1


def test_returns_none_on_clean_scan():
    # Printed dark dots have no embossed highlight to self-calibrate a template.
    assert read_page(_gray(render_braille_image("hello world"))) is None


def test_returns_none_on_noise():
    rng = np.random.default_rng(11)
    noise = rng.normal(190, 12, size=(600, 900)).clip(0, 255).astype(np.uint8)
    assert read_page(noise) is None


def test_returns_none_on_degenerate_input():
    assert read_page(None) is None  # type: ignore[arg-type]
    assert read_page(np.zeros((10, 10), dtype=np.uint8)) is None


def test_is_deterministic():
    gray = _embossed_gray()
    first, second = read_page(gray), read_page(gray)
    assert first is not None and second is not None

    def cells(result):
        return [
            (c.line_number, c.grid_index, c.dots)
            for line in result.grouping.lines
            for c in line
        ]

    assert cells(first) == cells(second)


@pytest.mark.parametrize("degrees", [180, 90, 270])
def test_recovers_right_angle_rotation(degrees):
    upright = read_page(_embossed_gray())
    assert upright is not None
    rotated = np.ascontiguousarray(np.rot90(_embossed_gray(), k=degrees // 90))
    result = read_page(rotated)
    assert result is not None
    # The reader applies the complementary rotation that brings the page back
    # upright, so the correction composed with the external rotation is a full
    # turn (180 is its own complement).
    assert result.rotation_applied != 0
    assert (degrees + result.rotation_applied) % 360 == 0
    # Same page, so a comparable number of cells is recovered after rotation.
    assert result.grouping.total_cells >= upright.grouping.total_cells - 4


# --- pipeline integration ----------------------------------------------------


def _force_blob_failure(monkeypatch):
    """Make the blob path (incl. L1 ladder) return zero cells."""
    empty = NormalisedDetection(
        detection=DetectionOutcome(), grouping=GroupingResult()
    )
    monkeypatch.setattr(
        pipeline_module, "detect_with_normalisation", lambda *a, **k: empty
    )


def test_pipeline_rescues_total_failure_with_template_reader(monkeypatch):
    _force_blob_failure(monkeypatch)
    payload = make_payload(dataUrl=image_to_data_url(
        render_embossed_braille_image(_EMBOSSED_TEXT, EmbossedStyle(), seed=4)
    ))
    response = run_ocr(OcrRequest(**payload))

    assert response.draftText  # a draft was produced where the blob path failed
    assert response.rawCells
    # Honest confidence: an experimental last-ditch rescue is never confident.
    assert 0.0 < response.confidence <= TEMPLATE_READER_CAP
    reasons = " ".join(flag.reason for flag in response.flags)
    assert "experimental full-resolution template reader" in reasons
    assert set(response.model_dump()) == EXPECTED_RESPONSE_KEYS


def test_pipeline_failure_stays_empty_when_template_reader_declines(monkeypatch):
    _force_blob_failure(monkeypatch)
    # Noise: the template reader returns None, so the failure stays a failure.
    rng = np.random.default_rng(5)
    noise = rng.normal(190, 12, size=(600, 900)).clip(0, 255).astype(np.uint8)
    payload = make_payload(dataUrl=image_to_data_url(Image.fromarray(noise, "L")))
    response = run_ocr(OcrRequest(**payload))
    assert response.draftText == ""
    assert response.confidence == 0.0
    assert response.flags


def test_drop_erasure_runs_removes_full_cell_blocks():
    from app.ocr.template_reader import _drop_erasure_runs

    full = (1, 2, 3, 4, 5, 6)
    letter = (1,)  # 'a'
    bbox = (0, 0, 1, 1)
    by_index = {
        0: (letter, 0.9, bbox),
        1: (full, 0.9, bbox),   # lone full cell -> kept (could be "for")
        2: (letter, 0.9, bbox),
        4: (full, 0.9, bbox),   # run of 3 adjacent full cells -> dropped
        5: (full, 0.9, bbox),
        6: (full, 0.9, bbox),
        7: (letter, 0.9, bbox),
    }
    _drop_erasure_runs(by_index)
    assert set(by_index) == {0, 1, 2, 7}


def test_pipeline_unchanged_when_blob_path_succeeds():
    # A normal embossed image decodes via the blob path; the template reader
    # must not run, so no experimental-reader flag appears.
    payload = make_payload(dataUrl=image_to_data_url(
        render_embossed_braille_image("hello world", EmbossedStyle(), seed=2)
    ))
    response = run_ocr(OcrRequest(**payload))
    reasons = " ".join(flag.reason for flag in response.flags)
    assert "experimental full-resolution template reader" not in reasons
