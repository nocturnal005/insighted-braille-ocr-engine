"""Stage 3D-J1 tests: real-capture diagnostic probe, capture quality, CLI.

All fixtures are synthetic renders or degraded derivatives built in
tmp_path - never real captures, never pupil material, never committed
images. Liblouis-dependent behaviour is made deterministic by
monkeypatching the adapter functions inside the probe module, so the
suite passes with or without Liblouis installed and regardless of the
LIBLOUIS_* environment configuration.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image, ImageFilter

from app.core.config import Settings
from app.evaluation.capture_quality import (
    CLASS_BORDERLINE,
    CLASS_READABLE,
    CLASS_RETAKE,
    CLASS_UNUSABLE,
    REASON_LIKELY_BLUR,
    REASON_LOW_CONTRAST,
    REASON_TOO_SMALL,
    REASON_TOO_SPARSE,
    assess_capture_quality,
)
from app.evaluation.diagnostic_probe import (
    FAILURE_DECODE_REJECTED,
    FAILURE_NO_DOT_CANDIDATES,
    FAILURE_UNSUPPORTED_FILE,
    STAGE_L0,
    STAGE_L4,
    STAGE_L5,
    STAGE_L6,
    probe_image_file,
    score_against_expected,
)
from app.evaluation.run_real_capture_diagnostic import main as diagnostic_main
from app.evaluation.sample_generator import render_braille_image, text_line_to_cells
from app.models.requests import OcrRequest
from app.ocr.braille_decode import dots_to_unicode_char
from app.ocr.pipeline import run_ocr
from app.tests.helpers import EXPECTED_RESPONSE_KEYS, make_payload

SAMPLE_TEXT = "hello world"


def _clean_image(tmp_path, name="rb_diag_clean.png"):
    path = tmp_path / name
    render_braille_image(SAMPLE_TEXT).save(path)
    return path


def _expected_rawbraille() -> str:
    cells = text_line_to_cells(SAMPLE_TEXT)
    return "".join(
        " " if cell is None else dots_to_unicode_char(cell) for cell in cells
    )


def _grade1_settings():
    return Settings(liblouis_table="en-ueb-g1.ctb", liblouis_enabled=True)


def _grade2_settings():
    return Settings(liblouis_table="en-ueb-g2.ctb", liblouis_enabled=True)


def _force_grade1_no_liblouis(monkeypatch):
    monkeypatch.setattr(
        "app.evaluation.diagnostic_probe.get_settings", _grade1_settings
    )
    monkeypatch.setattr(
        "app.evaluation.diagnostic_probe.liblouis_back_translate",
        lambda *_args: None,
    )


def _force_grade2_liblouis(monkeypatch):
    monkeypatch.setattr(
        "app.evaluation.diagnostic_probe.get_settings", _grade2_settings
    )
    monkeypatch.setattr(
        "app.evaluation.diagnostic_probe.liblouis_back_translate",
        lambda *_args: SAMPLE_TEXT,
    )


# --- Capture-quality classification ------------------------------------------


def test_clean_render_is_readable(tmp_path):
    path = _clean_image(tmp_path)
    quality = assess_capture_quality(path, probe_image_file(path))
    assert quality.classification == CLASS_READABLE
    assert quality.reasons == []
    assert quality.retake_recommended is False


def test_tiny_image_is_unusable(tmp_path):
    path = tmp_path / "rb_diag_tiny.png"
    Image.new("L", (40, 30), 255).save(path)
    quality = assess_capture_quality(path)
    assert quality.classification == CLASS_UNUSABLE
    assert REASON_TOO_SMALL in quality.reasons
    assert quality.retake_recommended is True


def test_heavy_blur_recommends_retake(tmp_path):
    path = tmp_path / "rb_diag_blur.png"
    render_braille_image(SAMPLE_TEXT).filter(
        ImageFilter.GaussianBlur(radius=6)
    ).save(path)
    quality = assess_capture_quality(path)
    assert quality.classification in (CLASS_RETAKE, CLASS_BORDERLINE)
    assert quality.retake_recommended is (quality.classification == CLASS_RETAKE)
    assert any("blur" in reason or "focus" in reason for reason in quality.reasons)


def test_severe_blur_reason_is_likely_blur(tmp_path):
    path = tmp_path / "rb_diag_blur_heavy.png"
    render_braille_image(SAMPLE_TEXT).filter(
        ImageFilter.GaussianBlur(radius=12)
    ).save(path)
    quality = assess_capture_quality(path)
    assert quality.classification == CLASS_RETAKE
    assert REASON_LIKELY_BLUR in quality.reasons


def test_blank_page_flags_low_contrast_and_sparse(tmp_path):
    path = tmp_path / "rb_diag_blank.png"
    Image.new("L", (400, 300), 255).save(path)
    quality = assess_capture_quality(path, probe_image_file(path))
    assert quality.classification == CLASS_RETAKE
    assert REASON_LOW_CONTRAST in quality.reasons
    assert REASON_TOO_SPARSE in quality.reasons


def test_unreadable_file_is_unusable(tmp_path):
    path = tmp_path / "rb_diag_bad.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot-an-image")
    quality = assess_capture_quality(path)
    assert quality.classification == CLASS_UNUSABLE


# --- Probe stage ladder --------------------------------------------------------


def test_clean_render_reaches_l4_without_liblouis(tmp_path, monkeypatch):
    _force_grade1_no_liblouis(monkeypatch)
    probe = probe_image_file(_clean_image(tmp_path))
    assert probe.stage == STAGE_L4
    assert probe.failure_point == "none"
    assert probe.accepted_dots > 0
    assert probe.lines_detected == 1
    assert probe.total_cells == 10
    assert probe.rawbraille_nonempty is True
    assert probe.liblouis_used is False
    assert probe.grade2_draft_produced is False
    assert probe.draft_nonempty is True  # Grade 1 fallback draft


def test_clean_render_reaches_l5_with_grade2_liblouis(tmp_path, monkeypatch):
    _force_grade2_liblouis(monkeypatch)
    probe = probe_image_file(_clean_image(tmp_path))
    assert probe.stage == STAGE_L5
    assert probe.liblouis_used is True
    assert probe.grade2_table_configured is True
    assert probe.grade2_draft_produced is True


def test_blank_image_stops_at_l0_no_dots(tmp_path):
    path = tmp_path / "rb_diag_blank.png"
    Image.new("L", (400, 300), 255).save(path)
    probe = probe_image_file(path)
    assert probe.stage == STAGE_L0
    assert probe.failure_point == FAILURE_NO_DOT_CANDIDATES
    assert probe.decode_ok is True


def test_corrupt_file_stops_at_l0_decode_rejected(tmp_path):
    path = tmp_path / "rb_diag_corrupt.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot-an-image")
    probe = probe_image_file(path)
    assert probe.stage == STAGE_L0
    assert probe.failure_point == FAILURE_DECODE_REJECTED
    assert probe.decode_ok is False


def test_unsupported_extension_is_safe(tmp_path):
    path = tmp_path / "rb_diag_page.bmp"
    Image.new("L", (200, 200), 255).save(path)
    probe = probe_image_file(path)
    assert probe.stage == STAGE_L0
    assert probe.failure_point == FAILURE_UNSUPPORTED_FILE


def test_probe_safe_dict_contains_no_content(tmp_path, monkeypatch):
    _force_grade2_liblouis(monkeypatch)
    probe = probe_image_file(_clean_image(tmp_path))
    blob = json.dumps(probe.to_safe_dict())
    assert not any(0x2800 <= ord(c) <= 0x28FF for c in blob)
    assert SAMPLE_TEXT.split()[0] not in blob.lower()
    assert "raw_braille" not in blob and "draft_text" not in blob


# --- Ground-truth scoring (L6) --------------------------------------------------


def test_score_against_expected_exact_match(tmp_path, monkeypatch):
    _force_grade1_no_liblouis(monkeypatch)
    probe = probe_image_file(_clean_image(tmp_path))
    scores = score_against_expected(probe, _expected_rawbraille())
    assert probe.stage == STAGE_L6
    assert scores["cell_error_rate"] == 0.0
    assert scores["exact_sample_match"] is True
    assert scores["english_cer"] is None  # no Liblouis Grade 2 in play


def test_score_adds_english_metrics_with_grade2(tmp_path, monkeypatch):
    _force_grade2_liblouis(monkeypatch)
    probe = probe_image_file(_clean_image(tmp_path))
    scores = score_against_expected(probe, _expected_rawbraille())
    assert scores["english_cer"] == 0.0
    assert scores["english_wer"] == 0.0


# --- Diagnostic CLI --------------------------------------------------------------


def _intake(tmp_path):
    images = tmp_path / "images"
    expected = tmp_path / "expected"
    metadata = tmp_path / "metadata"
    for folder in (images, expected, metadata):
        folder.mkdir()
    return images, expected, metadata


def _approved_metadata() -> dict:
    return {
        "permission_status": "approved_for_testing",
        "contains_real_pupil_data": False,
        "contains_live_assessment_material": False,
        "requires_english_transcript": False,
        "capture_type": "controlled_render",
        "source_type": "synthetic",
        "consent_note": "synthetic render created by the project for testing",
    }


def _run_cli(images, expected, metadata, report):
    return diagnostic_main(
        [
            "--input", str(images),
            "--expected", str(expected),
            "--metadata", str(metadata),
            "--report", str(report),
        ]
    )


def test_cli_scores_gated_sample_and_leaks_nothing(tmp_path, capsys):
    images, expected, metadata = _intake(tmp_path)
    _clean_image(images, "rb_diag_001.png")
    (expected / "rb_diag_001.braille").write_text(
        _expected_rawbraille(), encoding="utf-8"
    )
    (metadata / "rb_diag_001.json").write_text(
        json.dumps(_approved_metadata()), encoding="utf-8"
    )
    # An unsafely named image must appear only as a withheld label.
    _clean_image(images, "pupil_page.png")

    report_path = tmp_path / "report.json"
    exit_code = _run_cli(images, expected, metadata, report_path)
    assert exit_code == 0

    raw = report_path.read_text(encoding="utf-8")
    console = capsys.readouterr().out
    for blob in (raw, console):
        assert not any(0x2800 <= ord(c) <= 0x28FF for c in blob)
        assert "hello" not in blob.lower()
        assert "pupil_page" not in blob

    report = json.loads(raw)
    assert report["counts"]["scored"] == 1
    labels = {entry["label"] for entry in report["candidates"]}
    assert "rb_diag_001" in labels
    assert any(label.startswith("withheld_") for label in labels)
    scored = next(e for e in report["candidates"] if "scores" in e)
    assert scored["probe"]["stage"] == STAGE_L6
    assert scored["scores"]["cell_error_rate"] == 0.0


def test_cli_blocks_pupil_data_sample_without_processing(tmp_path, monkeypatch):
    images, expected, metadata = _intake(tmp_path)
    _clean_image(images, "rb_diag_002.png")
    bad = _approved_metadata()
    bad["contains_real_pupil_data"] = True
    (metadata / "rb_diag_002.json").write_text(json.dumps(bad), encoding="utf-8")

    probed = []
    monkeypatch.setattr(
        "app.evaluation.run_real_capture_diagnostic.probe_image_file",
        lambda path: probed.append(path),
    )
    report_path = tmp_path / "report.json"
    exit_code = _run_cli(images, expected, metadata, report_path)
    assert exit_code == 2
    assert probed == []  # the blocked sample was never OCR'd
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["blocked"] == 1
    assert report["candidates"] == []
    assert "BLOCKED" in report["verdict"].upper()


def test_cli_empty_intake_reports_blocked(tmp_path):
    images, expected, metadata = _intake(tmp_path)
    report_path = tmp_path / "report.json"
    exit_code = _run_cli(images, expected, metadata, report_path)
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"].startswith("BLOCKED")
    assert report["counts"] == {
        "candidates": 0,
        "probed": 0,
        "blocked": 0,
        "scored": 0,
    }


def test_cli_without_ground_truth_is_preview_only(tmp_path):
    images, expected, metadata = _intake(tmp_path)
    _clean_image(images, "rb_diag_003.png")  # no .braille, no metadata
    report_path = tmp_path / "report.json"
    exit_code = _run_cli(images, expected, metadata, report_path)
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["scored"] == 0
    entry = report["candidates"][0]
    assert "scores" not in entry
    assert any("permission" in note for note in entry["gating_notes"])
    assert "BLOCKED" in report["verdict"]  # formal evaluation still blocked


def test_cli_refuses_ungitignored_report_inside_repo(tmp_path):
    images, expected, metadata = _intake(tmp_path)
    from app.evaluation.run_real_capture_diagnostic import _REPO_ROOT

    # Explicitly anchored to the repo root (not CWD-relative) so the test is
    # deterministic regardless of where pytest is invoked from.
    report_path = _REPO_ROOT / "diag_report_should_never_exist.json"
    try:
        exit_code = _run_cli(images, expected, metadata, report_path)
        assert exit_code == 2
        assert not report_path.exists()
    finally:
        report_path.unlink(missing_ok=True)


def test_cli_allows_report_under_gitignored_reports_dir(tmp_path):
    from app.evaluation.run_real_capture_diagnostic import _REPO_ROOT

    images, expected, metadata = _intake(tmp_path)
    report_path = _REPO_ROOT / "reports" / "j1_test_artifact" / "run.json"
    try:
        exit_code = _run_cli(images, expected, metadata, report_path)
        assert exit_code == 0
        assert report_path.exists()  # reports/ is gitignored, so allowed
    finally:
        report_path.unlink(missing_ok=True)
        # clean the temp subdir if empty
        try:
            report_path.parent.rmdir()
        except OSError:
            pass


def test_cli_blocks_live_assessment_material(tmp_path, monkeypatch):
    # The second hard-rule flag must block exactly like the pupil-data flag.
    images, expected, metadata = _intake(tmp_path)
    _clean_image(images, "rb_diag_010.png")
    bad = _approved_metadata()
    bad["contains_live_assessment_material"] = True
    (metadata / "rb_diag_010.json").write_text(json.dumps(bad), encoding="utf-8")

    probed = []
    monkeypatch.setattr(
        "app.evaluation.run_real_capture_diagnostic.probe_image_file",
        lambda path: probed.append(path),
    )
    report_path = tmp_path / "report.json"
    assert _run_cli(images, expected, metadata, report_path) == 2
    assert probed == []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["blocked"] == 1


def test_cli_blocks_unparseable_metadata_fail_closed(tmp_path, monkeypatch):
    # A metadata file that cannot be parsed might be the one marking forbidden
    # material - it must FAIL CLOSED (block), never fall through to OCR.
    images, expected, metadata = _intake(tmp_path)
    _clean_image(images, "rb_diag_011.png")
    (metadata / "rb_diag_011.json").write_text(
        '{"contains_real_pupil_data": true,}', encoding="utf-8"  # trailing comma
    )
    probed = []
    monkeypatch.setattr(
        "app.evaluation.run_real_capture_diagnostic.probe_image_file",
        lambda path: probed.append(path),
    )
    report_path = tmp_path / "report.json"
    assert _run_cli(images, expected, metadata, report_path) == 2
    assert probed == []  # never OCR'd
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["blocked"] == 1


def test_cli_blocks_non_boolean_truthy_safety_flag(tmp_path, monkeypatch):
    # A "true"/"yes"/1 typo on a blocking flag must still block.
    images, expected, metadata = _intake(tmp_path)
    _clean_image(images, "rb_diag_012.png")
    bad = _approved_metadata()
    bad["contains_real_pupil_data"] = "yes"  # non-boolean truthy
    (metadata / "rb_diag_012.json").write_text(json.dumps(bad), encoding="utf-8")
    probed = []
    monkeypatch.setattr(
        "app.evaluation.run_real_capture_diagnostic.probe_image_file",
        lambda path: probed.append(path),
    )
    report_path = tmp_path / "report.json"
    assert _run_cli(images, expected, metadata, report_path) == 2
    assert probed == []


def test_cli_gated_sample_without_rawbraille_not_scored(tmp_path):
    # A gated sample (approved + .braille) that fails before L4 must NOT be
    # scored or promoted to L6 - it has no rawBraille to score.
    images, expected, metadata = _intake(tmp_path)
    # A blank page: decodes but yields no dots -> stops at L0.
    Image.new("L", (400, 300), 255).save(images / "rb_diag_013.png")
    (expected / "rb_diag_013.braille").write_text(
        _expected_rawbraille(), encoding="utf-8"
    )
    (metadata / "rb_diag_013.json").write_text(
        json.dumps(_approved_metadata()), encoding="utf-8"
    )
    report_path = tmp_path / "report.json"
    assert _run_cli(images, expected, metadata, report_path) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["scored"] == 0
    entry = report["candidates"][0]
    assert "scores" not in entry
    assert entry["probe"]["stage"] == STAGE_L0  # not falsely promoted to L6
    assert any("no rawBraille" in note for note in entry["gating_notes"])


def test_cli_bad_utf8_ground_truth_does_not_crash(tmp_path):
    # A .braille file that is not valid UTF-8 must not crash the run; the
    # sample degrades to preview-only and the report is still written.
    images, expected, metadata = _intake(tmp_path)
    _clean_image(images, "rb_diag_014.png")
    (expected / "rb_diag_014.braille").write_bytes(b"\xff\xfe\x00 not utf-8")
    (metadata / "rb_diag_014.json").write_text(
        json.dumps(_approved_metadata()), encoding="utf-8"
    )
    report_path = tmp_path / "report.json"
    assert _run_cli(images, expected, metadata, report_path) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["scored"] == 0
    entry = report["candidates"][0]
    assert "scores" not in entry


def test_cli_missing_input_directory_exits_2(tmp_path):
    _, expected, metadata = _intake(tmp_path)
    missing = tmp_path / "does_not_exist"
    report_path = tmp_path / "report.json"
    assert _run_cli(missing, expected, metadata, report_path) == 2


def test_markdown_report_renders_blocked_and_scored_and_withheld(tmp_path):
    # Exercise the blocked-entries, scored-summary, and withheld-label
    # branches of the markdown renderer, and assert nothing leaks.
    images, expected, metadata = _intake(tmp_path)
    # scored sample
    _clean_image(images, "rb_diag_020.png")
    (expected / "rb_diag_020.braille").write_text(
        _expected_rawbraille(), encoding="utf-8"
    )
    (metadata / "rb_diag_020.json").write_text(
        json.dumps(_approved_metadata()), encoding="utf-8"
    )
    # blocked sample
    _clean_image(images, "rb_diag_021.png")
    blk = _approved_metadata()
    blk["contains_real_pupil_data"] = True
    (metadata / "rb_diag_021.json").write_text(json.dumps(blk), encoding="utf-8")
    # unsafely named sample -> withheld label
    _clean_image(images, "pupil_page.png")

    report_path = tmp_path / "report.md"
    # exit 2 because a blocked sample is present
    assert _run_cli(images, expected, metadata, report_path) == 2
    text = report_path.read_text(encoding="utf-8")
    assert "## Blocked" in text
    assert "## Scored summary" in text
    assert "withheld_" in text
    assert "pupil_page" not in text
    assert "hello" not in text.lower()
    assert not any(0x2800 <= ord(c) <= 0x28FF for c in text)


def test_probe_never_raises_on_unreadable_supported_file(tmp_path):
    # A path with a supported extension that stats fine but cannot be read
    # (here: a directory named like a .png) must not raise (never-raises
    # contract); it is recorded as a read error, not a crash.
    from app.evaluation.diagnostic_probe import FAILURE_READ_ERROR, probe_image_file

    tricky = tmp_path / "isdir.png"
    tricky.mkdir()
    probe = probe_image_file(tricky)
    assert probe.failure_point == FAILURE_READ_ERROR
    assert probe.stage == STAGE_L0


def test_cli_markdown_report(tmp_path):
    images, expected, metadata = _intake(tmp_path)
    _clean_image(images, "rb_diag_004.png")
    report_path = tmp_path / "report.md"
    assert _run_cli(images, expected, metadata, report_path) == 0
    text = report_path.read_text(encoding="utf-8")
    assert text.startswith("# Real-capture Braille OCR diagnostic report")
    assert "hello" not in text.lower()
    assert not any(0x2800 <= ord(c) <= 0x28FF for c in text)


# --- Contract and safety locks ----------------------------------------------------


def test_ocr_contract_unchanged():
    response = run_ocr(OcrRequest(**make_payload()))
    assert set(response.model_dump().keys()) == EXPECTED_RESPONSE_KEYS


def test_diagnostic_note_keeps_draft_only_language():
    from app.evaluation.run_real_capture_diagnostic import DIAGNOSTIC_NOTE

    lowered = DIAGNOSTIC_NOTE.lower()
    assert "draft" in lowered
    assert "qtvi" in lowered
    assert "not accuracy" in lowered or "not accuracy measurements" in lowered
