"""Stage 3D-G5 tests: rawBraille evaluation hardening and real-capture readiness.

Covers: dataset descriptors clearly separating controlled renders from real
captures, reports excluding English CER/WER for Grade 2, manifest validation
(required fields, safe sample ids, no English modes), readiness-audit blocking
behaviour on missing/unsafe material, gitignore coverage of local-only sample
folders, the unchanged /ocr contract, and the absence of any Grade 2 English
back-translation path.

All fixtures are synthetic temporary files - never real pupil material and
never the local-only UKAAF files.
"""

from __future__ import annotations

import json
import subprocess

from PIL import Image

from app.evaluation.audit_rawbraille_dataset import audit
from app.evaluation.rawbraille_dataset import (
    DATASETS,
    RawBrailleDatasetSpec,
    get_spec,
)
from app.evaluation.rawbraille_manifest import (
    EVALUATION_MODES,
    REQUIRED_FIELDS,
    is_safe_sample_id,
    validate_entry,
    validate_manifest,
)
from app.evaluation.rawbraille_metrics import sample_metrics
from app.evaluation.run_rawbraille_evaluation import (
    REPORT_SCHEMA_VERSION,
    _build_report,
    _dataset_descriptor,
)
from app.models.requests import OcrRequest
from app.ocr.braille_decode import dots_to_unicode_char
from app.ocr.pipeline import run_ocr
from app.tests.helpers import EXPECTED_RESPONSE_KEYS, make_payload


# --- Dataset descriptors: controlled render vs real capture -------------------


def test_registry_separates_controlled_render_from_real_capture():
    controlled = get_spec("ukaaf_grade2_raw")
    real = get_spec("real_capture_grade2_raw")
    assert controlled.capture_type == "controlled_render"
    assert real.capture_type == "real_capture"
    # Every registered rawBraille dataset is cell-level only.
    for spec in DATASETS.values():
        assert spec.evaluation_mode == "rawbraille_cell_level"


def _row(capture_type="controlled_render"):
    a = dots_to_unicode_char(frozenset({1}))
    row = sample_metrics(a, a)
    row.update(
        {
            "confidence": 0.9,
            "flag_categories": {"unclear_braille_cell"},
            "ms": 10.0,
            "repeatable": True,
            "failed": False,
            "category": "prose",
            "variant": "clean",
            "capture_type": capture_type,
            "label": "sample_x",
        }
    )
    return row


def test_report_labels_capture_type_clearly():
    for name in ("ukaaf_grade2_raw", "real_capture_grade2_raw"):
        spec = get_spec(name)
        report = _build_report(spec, [_row(spec.capture_type)], 1, 0, "run-test")
        assert report["dataset"]["capture_type"] == spec.capture_type
        assert report["dataset"]["name"] == spec.name
        assert spec.capture_type.split("_")[0] in report["capture_type_note"].lower()
        assert report["run_id"] == "run-test"
        assert report["schema_version"] == REPORT_SCHEMA_VERSION
        assert report["samples"][0]["capture_type"] == spec.capture_type


# --- English CER/WER exclusion -------------------------------------------------


def test_report_excludes_english_cer_wer():
    spec = get_spec("ukaaf_grade2_raw")
    report = _build_report(spec, [_row()], 1, 0, "run-test")
    assert report["english_cer_wer_computed"] is False
    assert report["grade2_english_transcription"] == "out_of_scope"
    blob = json.dumps(report).lower()
    assert '"wer"' not in blob and "word_error_rate" not in blob
    assert "not english" in blob


def test_no_english_evaluation_mode_exists():
    # The manifest schema structurally cannot request English scoring.
    assert EVALUATION_MODES == {"rawbraille_cell_level"}
    entry = _manifest_entry(evaluation_mode="english_cer_wer")
    assert any("evaluation_mode" in issue for issue in validate_entry(entry))


def test_no_grade2_back_translation_support():
    # Behavioural lock: a Grade 2 contraction cell must NOT decode to its
    # English word. Dots {1,2,3,4,6} is the UEB "and" contraction; the
    # Grade 1 fallback must not know it.
    from app.core.config import get_settings
    from app.translation.fallback_translator import back_translate_unicode_lines

    and_contraction = dots_to_unicode_char(frozenset({1, 2, 3, 4, 6}))
    outcome = back_translate_unicode_lines([and_contraction])
    assert "and" not in outcome.text.lower()
    # And the configured Liblouis table remains Grade 1.
    assert "g1" in get_settings().liblouis_table.lower()


# --- Metrics still intact after G4 ---------------------------------------------


def test_rawbraille_metrics_still_work():
    a = dots_to_unicode_char(frozenset({1}))
    b = dots_to_unicode_char(frozenset({1, 2}))
    perfect = sample_metrics(a + b, a + b)
    assert perfect["cell_error_rate"] == 0.0 and perfect["exact_sample_match"] is True
    off = sample_metrics(a + b, a + a)
    assert off["cell_error_rate"] > 0.0 and off["exact_sample_match"] is False


# --- Manifest validation ---------------------------------------------------------


def _manifest_entry(**overrides) -> dict:
    entry = {
        "sample_id": "rb_capture_001",
        "image_path": "samples/real_rawbraille_images/rb_capture_001.png",
        "expected_rawbraille_path": "samples/real_rawbraille_expected/rb_capture_001.braille",
        "dataset_category": "rawbraille_validation",
        "capture_type": "real_capture",
        "source_type": "real_photo",
        "consent_or_safety_note": "anonymised sample, approved for testing",
        "grade_mode": "ueb_grade_2",
        "evaluation_mode": "rawbraille_cell_level",
    }
    entry.update(overrides)
    return entry


def test_manifest_valid_entry_passes():
    assert validate_entry(_manifest_entry()) == []


def test_manifest_requires_all_fields():
    for field_name in REQUIRED_FIELDS:
        entry = _manifest_entry()
        del entry[field_name]
        assert any(field_name in issue for issue in validate_entry(entry))


def test_manifest_rejects_unsafe_sample_ids():
    assert not is_safe_sample_id("John-Smith")
    assert not is_safe_sample_id("pupil_3_homework")
    assert is_safe_sample_id("rb_capture_001")
    issues = validate_entry(_manifest_entry(sample_id="pupil_3_homework"))
    assert any("unsafe" in issue for issue in issues)


def test_manifest_rejects_english_transcript_requirement():
    issues = validate_entry(_manifest_entry(requires_english_transcript=True))
    assert any("out of scope" in issue for issue in issues)


def test_validate_manifest_summary_counts():
    summary = validate_manifest([_manifest_entry(), _manifest_entry(sample_id="")])
    assert summary["entries"] == 2
    assert summary["valid"] == 1 and summary["invalid"] == 1


# --- Readiness audit ---------------------------------------------------------------


def _tmp_spec(tmp_path, name="test_raw"):
    return RawBrailleDatasetSpec(
        name=name,
        images_dir=tmp_path / "images",
        expected_dir=tmp_path / "expected",
        metadata_dir=tmp_path / "metadata",
        capture_type="real_capture",
        source_type="real_photo",
    )


def _add_sample(tmp_path, stem, *, expected=True, metadata=True, meta_overrides=None):
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "expected").mkdir(exist_ok=True)
    (tmp_path / "metadata").mkdir(exist_ok=True)
    Image.new("L", (200, 200), color=255).save(tmp_path / "images" / f"{stem}.png")
    if expected:
        (tmp_path / "expected" / f"{stem}.braille").write_text(
            dots_to_unicode_char(frozenset({1})), encoding="utf-8"
        )
    if metadata:
        meta = {
            "permission_status": "approved_for_testing",
            "capture_type": "real_capture",
            "source_type": "real_photo",
            "braille_type": "ueb_grade_2",
            "contains_real_pupil_data": False,
            "contains_live_assessment_material": False,
        }
        meta.update(meta_overrides or {})
        (tmp_path / "metadata" / f"{stem}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )


def test_audit_empty_dataset_is_expected_not_blocking(tmp_path):
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(tmp_path))
    assert result["images_present"] is False
    assert result["blocking"] == []
    assert result["ready"] is False  # not ready, but cleanly empty


def test_audit_ready_dataset(tmp_path):
    _add_sample(tmp_path, "rb_capture_001")
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(tmp_path))
    assert result["blocking"] == []
    assert result["ready"] is True
    assert result["english_transcript_required"] is False


def test_audit_blocks_missing_expected_rawbraille(tmp_path):
    _add_sample(tmp_path, "rb_capture_001", expected=False)
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(tmp_path))
    assert any(".braille" in issue for issue in result["blocking"])
    assert result["ready"] is False


def test_audit_blocks_unsafe_sample_names(tmp_path):
    _add_sample(tmp_path, "John-Smith")
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(tmp_path))
    assert any("unsafe sample id" in issue for issue in result["blocking"])
    assert result["ready"] is False
    # The unsafe name itself must not appear in the audit output.
    assert "John-Smith" not in json.dumps(result["blocking"])


def test_audit_blocks_english_transcripts_and_pupil_data(tmp_path):
    _add_sample(
        tmp_path,
        "rb_capture_001",
        meta_overrides={"contains_real_pupil_data": True},
    )
    (tmp_path / "expected" / "rb_capture_001.txt").write_text("x", encoding="utf-8")
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(tmp_path))
    assert any("contains_real_pupil_data" in issue for issue in result["blocking"])
    assert any("transcripts are out of scope" in issue for issue in result["blocking"])


def test_audit_validates_manifest(tmp_path):
    _add_sample(tmp_path, "rb_capture_001")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([_manifest_entry(), _manifest_entry(evaluation_mode="english")]),
        encoding="utf-8",
    )
    result = audit("real_capture_grade2_raw", manifest_path=manifest, spec=_tmp_spec(tmp_path))
    assert result["manifest"]["entries"] == 2
    assert result["manifest"]["invalid"] == 1
    assert result["ready"] is False


# --- Repository safety --------------------------------------------------------------


def test_local_only_folders_are_gitignored():
    probes = [
        "reports/some-report.json",
        "samples/real_rawbraille_images/photo.png",
        "samples/real_rawbraille_expected/photo.braille",
        "samples/real_rawbraille_metadata/photo.json",
        "_external_sources/ukaaf/anything.pdf",
    ]
    for probe in probes:
        result = subprocess.run(
            ["git", "check-ignore", "-q", probe], capture_output=True, timeout=10
        )
        assert result.returncode == 0, f"{probe} is not gitignored"


def test_gitkeep_files_are_trackable():
    for probe in (
        "samples/real_rawbraille_images/.gitkeep",
        "samples/real_rawbraille_expected/.gitkeep",
        "samples/real_rawbraille_metadata/.gitkeep",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", probe], capture_output=True, timeout=10
        )
        assert result.returncode != 0, f"{probe} should NOT be ignored"


# --- Contract ------------------------------------------------------------------------


def test_ocr_contract_unchanged():
    response = run_ocr(OcrRequest(**make_payload()))
    assert set(response.model_dump().keys()) == EXPECTED_RESPONSE_KEYS
