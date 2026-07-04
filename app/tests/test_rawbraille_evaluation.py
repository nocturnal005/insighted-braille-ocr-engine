"""Stage 3D-G3 tests: cell-level (rawBraille) evaluation of the visual pipeline.

Covers the Braille-ASCII (BRF transport) codec, BRF normalisation rules, the
cell-level metrics, the sanitized report shape, the dataset gating, and the
unchanged /ocr contract. English CER/WER is never computed for Grade 2.

All fixtures are synthetic deterministic strings and renders - never real
pupil material and never the local-only UKAAF files.
"""

from __future__ import annotations

import json

from PIL import Image

from app.evaluation.braille_ascii import (
    BRF_CHAR_TO_DOTS,
    brf_first_page_lines,
    brf_line_to_cells,
    brf_line_to_unicode,
    content_lines,
    expected_rawbraille,
    normalise_brf_text,
    verify_table,
)
from app.evaluation.rawbraille_dataset import discover_samples
from app.evaluation.rawbraille_metrics import (
    cell_counts,
    cell_error_rate,
    line_metrics,
    rawbraille_cer,
    sample_metrics,
)
from app.evaluation.run_rawbraille_evaluation import _build_report
from app.evaluation.sample_generator import (
    image_to_data_url,
    render_cells_image,
)
from app.models.requests import OcrRequest
from app.ocr.braille_decode import dots_to_unicode_char
from app.ocr.pipeline import run_ocr
from app.tests.helpers import EXPECTED_RESPONSE_KEYS


# --- Braille ASCII codec -----------------------------------------------------


def test_ascii_braille_table_valid_and_engine_consistent():
    verify_table()  # bijection over 64 cells + agreement with engine maps
    assert BRF_CHAR_TO_DOTS["A"] == frozenset({1})
    assert BRF_CHAR_TO_DOTS["Z"] == frozenset({1, 3, 5, 6})
    assert BRF_CHAR_TO_DOTS["#"] == frozenset({3, 4, 5, 6})
    assert BRF_CHAR_TO_DOTS[","] == frozenset({6})
    assert BRF_CHAR_TO_DOTS["1"] == frozenset({2})  # lowered 'a', transport code
    assert BRF_CHAR_TO_DOTS[" "] == frozenset()


def test_brf_line_decode_known_values():
    assert brf_line_to_unicode("ABC") == "".join(
        dots_to_unicode_char(d) for d in (frozenset({1}), frozenset({1, 2}), frozenset({1, 4}))
    )
    cells = brf_line_to_cells("A B")
    assert cells == [frozenset({1}), None, frozenset({1, 2})]


# --- Normalisation rules -----------------------------------------------------


def test_normalisation_line_endings_and_pages():
    assert normalise_brf_text("a\r\nb\rc") == "a\nb\nc"
    # First page ends at the form feed; outer blank lines trimmed.
    text = "\r\n  AB  \r\nCD\r\n\r\n\x0cPAGE2\r\n"
    assert brf_first_page_lines(text) == ["  AB", "CD"]


def test_content_lines_strip_and_drop_blanks():
    lines = ["  AB", "", "  C D ", "   "]
    assert content_lines(lines) == ["AB", "C D"]


def test_expected_rawbraille_caps_long_space_runs():
    # 8 spaces between two cells must collapse to the 5-space cap.
    line = "A" + " " * 8 + "B"
    out = expected_rawbraille([line])
    assert out.count(" ") == 5
    assert out.startswith(dots_to_unicode_char(frozenset({1})))


# --- Cell-level metrics ------------------------------------------------------


def _u(*dotsets: frozenset[int]) -> str:
    return "".join(dots_to_unicode_char(d) for d in dotsets)


def test_cell_and_rawbraille_error_rates():
    a, b, c = frozenset({1}), frozenset({1, 2}), frozenset({1, 4})
    expected = _u(a, b, c)
    assert cell_error_rate(expected, expected) == 0.0
    assert rawbraille_cer(expected, expected) == 0.0
    # One substitution out of three cells.
    predicted = _u(a, b, a)
    assert abs(cell_error_rate(expected, predicted) - 1 / 3) < 1e-9


def test_cell_error_rate_is_space_agnostic():
    a, b = frozenset({1}), frozenset({1, 2})
    assert cell_error_rate(_u(a, b), _u(a) + " " + _u(b)) == 0.0


def test_line_and_cell_count_metrics():
    a, b = frozenset({1}), frozenset({1, 2})
    expected = _u(a, b) + "\n" + _u(a)
    predicted = _u(a, b)  # a whole line dropped
    lines = line_metrics(expected, predicted)
    assert lines["expected_lines"] == 2
    assert lines["predicted_lines"] == 1
    assert lines["line_count_mismatch"] == 1
    assert lines["exact_line_matches"] == 1
    assert cell_counts(expected, predicted) == (3, 2)


def test_sample_metrics_has_no_english_keys():
    a = frozenset({1})
    metrics = sample_metrics(_u(a), _u(a))
    assert metrics["exact_sample_match"] is True
    assert "cell_error_rate" in metrics and "rawbraille_cer" in metrics
    # This is a cell-level metric set only - no English draft-text scoring.
    for forbidden in ("wer", "word_error_rate", "draft", "draftText", "english"):
        assert forbidden not in metrics


# --- Report sanitisation -----------------------------------------------------


def test_report_excludes_all_braille_text():
    a, b = frozenset({1}), frozenset({1, 2})
    expected = _u(a, b)
    predicted = _u(a, a)
    row = sample_metrics(expected, predicted)
    row.update(
        {
            "confidence": 0.9,
            "flag_categories": {"unclear_braille_cell"},
            "ms": 12.0,
            "repeatable": True,
            "failed": False,
            "category": "prose",
            "variant": "clean",
            "label": "sample_x",
        }
    )
    report = _build_report([row], samples=1, skipped=0)
    blob = json.dumps(report)
    # No Unicode Braille glyph may appear anywhere in the report.
    assert not any("⠀" <= ch <= "⣿" for ch in blob)
    assert expected not in blob and predicted not in blob
    assert report["summary"]["mean_cell_error_rate"] >= 0.0


# --- Dataset gating ----------------------------------------------------------


def _write_sample(base, stem, *, permission="approved_for_testing", expected=True):
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "expected").mkdir(parents=True, exist_ok=True)
    (base / "metadata").mkdir(parents=True, exist_ok=True)
    Image.new("L", (60, 60), color=255).save(base / "images" / f"{stem}.png")
    if expected:
        (base / "expected" / f"{stem}.braille").write_text(
            dots_to_unicode_char(frozenset({1})), encoding="utf-8"
        )
    (base / "metadata" / f"{stem}.json").write_text(
        json.dumps(
            {
                "source_type": "controlled_ukaaf_grade2",
                "braille_type": "ueb_grade_2",
                "capture_method": "rendered",
                "permission_status": permission,
                "contains_real_pupil_data": False,
                "contains_live_assessment_material": False,
                "category": "prose",
                "variant": "clean",
            }
        ),
        encoding="utf-8",
    )


def test_dataset_gating(tmp_path):
    base = tmp_path / "ds"
    _write_sample(base, "ok_sample")
    _write_sample(base, "not_approved_sample", permission="not_approved")
    _write_sample(base, "no_expected_sample", expected=False)

    samples = {
        s.sample_id: s
        for s in discover_samples(base / "images", base / "expected", base / "metadata")
    }
    assert samples["ok_sample"].evaluable
    assert not samples["not_approved_sample"].evaluable
    assert not samples["no_expected_sample"].evaluable
    assert "missing expected .braille file" in samples["no_expected_sample"].skip_reasons


def test_missing_dataset_returns_empty(tmp_path):
    assert discover_samples(tmp_path / "nope", tmp_path / "nope2", tmp_path / "nope3") == []


# --- Contract + controlled round trip ----------------------------------------


def test_ocr_contract_unchanged_on_rendered_cells():
    # Render arbitrary (non-Grade-1-text) cells and confirm the response shape.
    cells = [[frozenset({1, 2}), frozenset({1, 4}), None, frozenset({1, 4, 5})]]
    response = run_ocr(
        OcrRequest(
            taskId="t",
            title="t",
            fileName="t.png",
            mimeType="image/png",
            dataUrl=image_to_data_url(render_cells_image(cells)),
        )
    )
    assert set(response.model_dump().keys()) == EXPECTED_RESPONSE_KEYS


def test_round_trip_reads_controlled_cells():
    # A small controlled page of full letter cells should read back with low
    # cell error - a sanity check that the visual pipeline reconstructs cells.
    letters = [frozenset({1, 2, 4, 5}), frozenset({1, 3, 5}), frozenset({1, 3, 5, 6})]
    line_cells = [[letters[0], letters[1], letters[2], None, letters[0], letters[1]]]
    expected = expected_rawbraille(
        ["".join({v: k for k, v in BRF_CHAR_TO_DOTS.items()}[c] for c in row if c)
         for row in line_cells]
    )
    response = run_ocr(
        OcrRequest(
            taskId="t",
            title="t",
            fileName="t.png",
            mimeType="image/png",
            dataUrl=image_to_data_url(render_cells_image(line_cells)),
        )
    )
    assert cell_error_rate(expected, response.rawBraille or "") <= 0.10
