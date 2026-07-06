"""Stage 3D-I1 tests: Liblouis integration and Grade 2 evaluation path.

Covers: adapter DLL/table path wiring, Grade 2 table detection, pipeline
Grade 2 flags, English CER/WER gating in the rawBraille evaluator, and
report schema when English scoring is enabled.

Tests that require the Liblouis DLL and tables are skipped when unavailable
(the expected state on a fresh checkout without a local Liblouis install).
"""

from __future__ import annotations

import pytest

from app.evaluation.rawbraille_metrics import sample_metrics
from app.evaluation.run_rawbraille_evaluation import _build_report
from app.ocr.braille_decode import dots_to_unicode_char
from app.translation.liblouis_adapter import is_grade2_table, liblouis_available


# --- Grade 2 table detection heuristic ----------------------------------------


def test_is_grade2_detects_g2_tables():
    assert is_grade2_table("en-ueb-g2.ctb") is True
    assert is_grade2_table("en-ueb-grade2.ctb") is True
    assert is_grade2_table("contracted-ueb.ctb") is True


def test_is_grade2_rejects_g1_tables():
    assert is_grade2_table("en-ueb-g1.ctb") is False
    assert is_grade2_table("en-ueb-g1.utb") is False
    assert is_grade2_table("unicode.dis") is False


# --- Fallback translator still does NOT interpret Grade 2 ---------------------


def test_fallback_translator_ignores_grade2_contractions():
    from app.translation.fallback_translator import back_translate_unicode_lines

    and_contraction = dots_to_unicode_char(frozenset({1, 2, 3, 4, 6}))
    outcome = back_translate_unicode_lines([and_contraction])
    assert "and" not in outcome.text.lower()


# --- Default config is still Grade 1 -----------------------------------------


def test_default_config_is_grade1():
    from app.core.config import Settings

    assert "g1" in Settings.model_fields["liblouis_table"].default.lower()


def test_default_config_has_liblouis_path_settings():
    from app.core.config import get_settings

    s = get_settings()
    assert hasattr(s, "liblouis_dll_dir")
    assert hasattr(s, "liblouis_table_path")


# --- English scoring gating ---------------------------------------------------


def test_english_scoring_unavailable_with_grade1_table(monkeypatch):
    import app.evaluation.run_rawbraille_evaluation as harness
    from app.core.config import Settings

    monkeypatch.setattr(
        harness, "get_settings", lambda: Settings(liblouis_table="en-ueb-g1.ctb")
    )
    assert harness._english_scoring_available() is False


# --- Report schema with and without English scoring ---------------------------


def _row(english_cer=None, english_wer=None, capture_type="controlled_render"):
    a = dots_to_unicode_char(frozenset({1}))
    row = sample_metrics(a, a)
    row.update(
        {
            "confidence": 0.9,
            "flag_categories": {"possible_contraction_issue"},
            "ms": 10.0,
            "repeatable": True,
            "failed": False,
            "category": "prose",
            "variant": "clean",
            "capture_type": capture_type,
            "label": "sample_x",
        }
    )
    if english_cer is not None:
        row["english_cer"] = english_cer
        row["english_wer"] = english_wer
    return row


def test_report_without_english_scoring():
    from app.evaluation.rawbraille_dataset import get_spec

    spec = get_spec("ukaaf_grade2_raw")
    report = _build_report(spec, [_row()], 1, 0, "run-test", english_scoring=False)
    assert report["english_cer_wer_computed"] is False
    assert report["grade2_english_transcription"] == "out_of_scope"
    assert "english_summary" not in report
    for sample in report["samples"]:
        assert "english_cer" not in sample


def test_report_with_english_scoring():
    from app.evaluation.rawbraille_dataset import get_spec

    spec = get_spec("ukaaf_grade2_raw")
    row = _row(english_cer=0.05, english_wer=0.10)
    report = _build_report(spec, [row], 1, 0, "run-test", english_scoring=True)
    assert report["english_cer_wer_computed"] is True
    assert report["grade2_english_transcription"] == "supplementary_via_liblouis"
    assert "english_summary" in report
    summary = report["english_summary"]
    assert summary["n"] == 1
    assert summary["mean_english_cer"] == 0.05
    assert summary["mean_english_wer"] == 0.10
    assert "note" in summary
    assert "supplementary" in summary["note"].lower()
    # The top-level note must not contradict the English scoring: no claim
    # that contractions are uninterpreted when Liblouis Grade 2 computed them.
    assert "does not interpret" not in report["note"].lower()
    assert "supplementary" in report["note"].lower()
    for sample in report["samples"]:
        assert "english_cer" in sample
        assert "english_wer" in sample


# --- Liblouis smoke tests (skipped when DLL not available) --------------------

needs_liblouis = pytest.mark.skipif(
    not liblouis_available(),
    reason="Liblouis DLL and tables not installed locally",
)


@needs_liblouis
def test_english_scoring_available_with_grade2_table(monkeypatch):
    import app.evaluation.run_rawbraille_evaluation as harness
    from app.core.config import Settings

    monkeypatch.setattr(
        harness, "get_settings", lambda: Settings(liblouis_table="en-ueb-g2.ctb")
    )
    assert harness._english_scoring_available() is True


@needs_liblouis
def test_liblouis_back_translate_grade1():
    from app.translation.liblouis_adapter import liblouis_back_translate

    result = liblouis_back_translate("⠓⠑⠇⠇⠕", "en-ueb-g1.ctb")
    assert result is not None
    assert result.strip().lower() == "hello"


@needs_liblouis
def test_liblouis_back_translate_returns_none_for_bad_table():
    from app.translation.liblouis_adapter import liblouis_back_translate

    result = liblouis_back_translate("⠓⠑⠇⠇⠕", "nonexistent.ctb")
    assert result is None
