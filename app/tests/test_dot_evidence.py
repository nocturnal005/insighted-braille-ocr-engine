"""Grid-evidence re-scoring tests (Stage 3D-M1).

The evidence stage targets embossed photographs only and must be provably
inert everywhere else: dark-path captures, pages with no grid, and the
whole controlled synthetic suite must be byte-for-byte unaffected.
"""

from __future__ import annotations

import numpy as np

from app.evaluation.metrics import normalise_text
from app.evaluation.sample_generator import (
    image_to_data_url,
    render_braille_image,
    render_embossed_braille_image,
)
from app.models.requests import OcrRequest
from app.ocr.capture_normalization import detect_with_normalisation
from app.ocr.cell_grouping import GroupingResult
from app.ocr.dot_detection import DetectionOutcome
from app.ocr.dot_evidence import refine_grouping
from app.ocr.pipeline import _select_variant, run_ocr

TEXT = "light travels\nin straight lines"


def _request(image) -> OcrRequest:
    return OcrRequest(
        taskId="task-evidence-test",
        title="evidence test",
        fileName="test-page.png",
        mimeType="image/png",
        dataUrl=image_to_data_url(image),
    )


def _detect(image):
    gray = np.array(image.convert("L"))
    normalised = detect_with_normalisation(gray, _select_variant)
    return normalised.detection, normalised.grouping


def test_dark_mode_is_never_refined():
    detection, grouping = _detect(render_braille_image(TEXT))
    assert detection.mode == "dark"
    outcome = refine_grouping(detection, grouping)
    assert not outcome.applied
    assert outcome.reason == "dark_mode_not_refined"


def test_empty_grouping_fails_closed():
    outcome = refine_grouping(DetectionOutcome(), GroupingResult())
    assert not outcome.applied
    assert outcome.reason == "no_grid_or_image"


def test_dark_synthetic_decode_is_unchanged_end_to_end():
    response = run_ocr(_request(render_braille_image(TEXT)))
    assert normalise_text(response.draftText) == normalise_text(TEXT)
    assert response.confidence > 0.5


def test_emboss_synthetic_still_decodes_with_refinement_active():
    image = render_embossed_braille_image(TEXT, seed=7)
    response = run_ocr(_request(image))
    # The embossed render must still decode to the right text whether or
    # not refinement fired; refinement may only ever change cells when its
    # evidence confirms the page's own detected dots.
    assert normalise_text(response.draftText) == normalise_text(TEXT)


def test_refinement_stats_are_consistent_when_applied():
    image = render_embossed_braille_image(TEXT, seed=11)
    detection, grouping = _detect(image)
    outcome = refine_grouping(detection, grouping)
    if not outcome.applied:
        # Fail-closed is a legitimate outcome on a synthetic render; the
        # reason must come from the fixed vocabulary.
        assert outcome.reason in {
            "dark_mode_not_refined",
            "too_few_cells",
            "no_light_direction",
            "too_few_background_samples",
            "degenerate_background",
            "evidence_contradicts_detection",
        }
        return
    assert outcome.total_cells == sum(len(line) for line in outcome.lines)
    assert outcome.cells_changed >= outcome.cells_recovered
    for line in outcome.lines:
        for cell in line:
            assert cell.dots, "refined cells never carry an empty pattern"
            assert all(1 <= n <= 6 for n in cell.dots)
