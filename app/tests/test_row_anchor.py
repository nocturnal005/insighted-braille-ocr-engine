"""Stage 3D-G4 tests: dot-row anchor robustness.

The G3 validation showed that a whole Braille line using only the middle/lower
rows of the cell (e.g. a line of comma cells, dots {2}) was read one row too
high — the comma cell {2} decoding as {1}. The fix anchors each line to the
page line ladder (`_line_lifts`) so a line that lacks a top-row dot is placed
at its true row position, while leaving every line that already carries a
top-row dot unchanged.

All fixtures are synthetic deterministic renders — never real pupil material
and never the local-only UKAAF files.
"""

from __future__ import annotations

from app.evaluation.sample_generator import image_to_data_url, render_cells_image
from app.ocr.braille_decode import dots_to_unicode_char
from app.ocr.cell_grouping import _line_lifts
from app.models.requests import OcrRequest
from app.ocr.pipeline import run_ocr
from app.tests.helpers import EXPECTED_RESPONSE_KEYS

# Cells spanning all three rows and both columns — a reliable reference line.
FULL = [frozenset({1, 2, 3}), frozenset({4, 5, 6}), frozenset({1, 2, 3}), frozenset({4, 5, 6})]
DOT2_CELL = frozenset({2})   # comma-like: middle row only
DOT3_CELL = frozenset({3})   # lower row only


def _ocr_cells(line_cells):
    response = run_ocr(
        OcrRequest(
            taskId="t",
            title="t",
            fileName="t.png",
            mimeType="image/png",
            dataUrl=image_to_data_url(render_cells_image(line_cells)),
        )
    )
    return response


def _line_of(response, char) -> bool:
    """True if some rawBraille line is entirely `char` (>=3 cells)."""
    for line in (response.rawBraille or "").split("\n"):
        stripped = line.strip()
        if len(stripped) >= 3 and set(stripped) == {char}:
            return True
    return False


# --- The specific G3 row-lift pattern ---------------------------------------


def test_middle_row_only_line_reads_as_dot_two_not_dot_one():
    # A line of comma cells among normal lines must decode as dots {2},
    # not be lifted to dots {1}.
    page = [FULL, FULL, FULL, [DOT2_CELL] * 4, FULL, FULL]
    response = _ocr_cells(page)
    assert _line_of(response, dots_to_unicode_char(DOT2_CELL))  # {2} present
    assert not _line_of(response, dots_to_unicode_char(frozenset({1})))  # not lifted to {1}


def test_lower_row_only_line_reads_as_dot_three():
    page = [FULL, FULL, FULL, [DOT3_CELL] * 4, FULL, FULL]
    response = _ocr_cells(page)
    assert _line_of(response, dots_to_unicode_char(DOT3_CELL))  # {3} present
    assert not _line_of(response, dots_to_unicode_char(frozenset({1})))


def test_normal_multirow_page_is_unaffected():
    # Every line carries a top-row dot: the ladder correction must be a no-op
    # and the page must still decode cleanly.
    page = [FULL, FULL, FULL, FULL]
    response = _ocr_cells(page)
    assert response.rawBraille
    # FULL lines contain {1,2,3} and {4,5,6} cells; both must be present.
    text = response.rawBraille
    assert dots_to_unicode_char(frozenset({1, 2, 3})) in text
    assert dots_to_unicode_char(frozenset({4, 5, 6})) in text


# --- _line_lifts unit level --------------------------------------------------


def _clean_line(origin, u_v):
    return [origin, origin + u_v, origin + 2 * u_v]


def test_line_lifts_detects_single_lifted_line():
    u_v = 10.0
    pitch = 5 * u_v
    # Six lines, origins 0..250; line index 4 is comma-only (top at origin+u_v).
    row_centers = []
    line_groups = []
    for k in range(6):
        origin = k * pitch
        start = len(row_centers)
        if k == 4:
            row_centers.append(origin + u_v)  # single middle-row cluster
            line_groups.append([start])
        else:
            row_centers.extend(_clean_line(origin, u_v))
            line_groups.append([start, start + 1, start + 2])
    lifts = _line_lifts(line_groups, row_centers, u_v)
    assert lifts == [0, 0, 0, 0, 1, 0]


def test_line_lifts_is_noop_on_all_normal_lines():
    u_v = 10.0
    pitch = 5 * u_v
    row_centers = []
    line_groups = []
    for k in range(5):
        start = len(row_centers)
        row_centers.extend(_clean_line(k * pitch, u_v))
        line_groups.append([start, start + 1, start + 2])
    assert _line_lifts(line_groups, row_centers, u_v) == [0, 0, 0, 0, 0]


def test_line_lifts_bails_out_without_reference_or_structure():
    u_v = 10.0
    # Too few lines -> no correction.
    assert _line_lifts([[0]], [0.0], u_v) == [0]
    # No full-height reference line (all single-cluster) -> conservative no-op.
    row_centers = [0.0, 50.0, 110.0]
    line_groups = [[0], [1], [2]]
    assert _line_lifts(line_groups, row_centers, u_v) == [0, 0, 0]


# --- Contract + scope --------------------------------------------------------


def test_contract_unchanged_and_no_grade2_translation():
    # A middle-row-only line is Grade-2-flavoured content; the response shape
    # must be unchanged and no Grade 2 English claim is introduced here.
    response = _ocr_cells([FULL, FULL, FULL, [DOT2_CELL] * 4, FULL])
    assert set(response.model_dump().keys()) == EXPECTED_RESPONSE_KEYS
