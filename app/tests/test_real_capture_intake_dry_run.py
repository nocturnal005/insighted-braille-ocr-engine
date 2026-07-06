"""Stage 3D-G6 tests: real-capture collection protocol and intake dry run.

Proves the real-capture intake workflow is operational BEFORE any real sample
exists: the protocol and templates are present and safe, a temporary intake
built from the templates passes the readiness audit end-to-end, the audit
blocks every unsafe variation, the real intake folders remain empty (.gitkeep
only) and gitignored, and no Grade 2 English path exists.

All fixtures are synthetic temporary files - never real pupil material, never
photographs, and never the local-only UKAAF files. Nothing is written into the
real intake folders.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

from app.evaluation.audit_rawbraille_dataset import audit
from app.evaluation.rawbraille_dataset import (
    RawBrailleDatasetSpec,
    REAL_EXPECTED_DIR,
    REAL_IMAGES_DIR,
    REAL_METADATA_DIR,
    _validate_metadata,
)
from app.evaluation.rawbraille_manifest import (
    is_safe_sample_id,
    validate_manifest,
)
from app.models.requests import OcrRequest
from app.ocr.braille_decode import dots_to_unicode_char
from app.ocr.pipeline import run_ocr
from app.tests.helpers import EXPECTED_RESPONSE_KEYS, make_payload

DOCS = Path("docs")
PROTOCOL = DOCS / "stage_3d_g6_real_capture_collection_protocol.md"
MANIFEST_TEMPLATE = DOCS / "templates" / "real_capture_rawbraille_manifest.template.json"
METADATA_TEMPLATE = DOCS / "templates" / "real_capture_rawbraille_metadata.template.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- Protocol and template files exist and are safe ---------------------------


def test_protocol_and_templates_exist():
    assert PROTOCOL.is_file()
    assert MANIFEST_TEMPLATE.is_file()
    assert METADATA_TEMPLATE.is_file()
    text = PROTOCOL.read_text(encoding="utf-8").lower()
    # The protocol must state the core boundaries.
    for phrase in (
        "forbidden",
        "draft-only",
        "cell level",
        "real pupil work",
        "english",
    ):
        assert phrase in text, f"protocol missing '{phrase}'"


def test_templates_contain_no_braille_or_real_content():
    for path in (MANIFEST_TEMPLATE, METADATA_TEMPLATE):
        text = path.read_text(encoding="utf-8")
        assert not any("⠀" <= ch <= "⣿" for ch in text), path
        assert "ukaaf" not in text.lower()


def test_manifest_template_validates_with_zero_invalid():
    entries = _load(MANIFEST_TEMPLATE)
    assert isinstance(entries, list) and entries
    summary = validate_manifest(entries)
    assert summary["invalid"] == 0


def test_manifest_template_is_cell_level_only():
    for entry in _load(MANIFEST_TEMPLATE):
        assert entry["evaluation_mode"] == "rawbraille_cell_level"
        assert entry.get("requires_english_transcript") is False
        assert entry["capture_type"] == "real_capture"


def test_metadata_template_safety_flags():
    meta = _load(METADATA_TEMPLATE)
    assert meta["contains_real_pupil_data"] is False
    assert meta["contains_live_assessment_material"] is False
    assert meta["requires_english_transcript"] is False
    assert meta["evaluation_mode"] == "rawbraille_cell_level"
    assert meta["permission_status"] == "approved_for_testing"
    assert _validate_metadata(meta) == []


def test_template_sample_ids_are_anonymised():
    for entry in _load(MANIFEST_TEMPLATE):
        assert is_safe_sample_id(entry["sample_id"])


# --- Intake dry run: templates -> temporary intake -> audit READY --------------


def _tmp_spec(tmp_path) -> RawBrailleDatasetSpec:
    return RawBrailleDatasetSpec(
        name="dry_run",
        images_dir=tmp_path / "images",
        expected_dir=tmp_path / "expected",
        metadata_dir=tmp_path / "metadata",
        capture_type="real_capture",
        source_type="real_photo",
    )


def _build_intake_from_templates(tmp_path, stem="rb_capture_001", meta_overrides=None):
    """Simulate the protocol: image + expected cells + template-based metadata."""
    for sub in ("images", "expected", "metadata"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    Image.new("L", (200, 200), color=255).save(tmp_path / "images" / f"{stem}.png")
    (tmp_path / "expected" / f"{stem}.braille").write_text(
        dots_to_unicode_char(frozenset({1})), encoding="utf-8"
    )
    meta = _load(METADATA_TEMPLATE)
    meta.update(meta_overrides or {})
    (tmp_path / "metadata" / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


def test_dry_run_template_based_intake_is_ready(tmp_path):
    _build_intake_from_templates(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(MANIFEST_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    result = audit("real_capture_grade2_raw", manifest_path=manifest, spec=_tmp_spec(tmp_path))
    assert result["blocking"] == []
    assert result["ready"] is True
    assert result["capture_type"] == "real_capture"
    assert result["manifest"]["invalid"] == 0


def test_dry_run_audit_blocks_unsafe_variations(tmp_path):
    # Unsafe name.
    unsafe = tmp_path / "a"
    _build_intake_from_templates(unsafe, stem="John-Smith")
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(unsafe))
    assert any("unsafe sample id" in issue for issue in result["blocking"])

    # English transcript file.
    transcript = tmp_path / "b"
    _build_intake_from_templates(transcript)
    (transcript / "expected" / "rb_capture_001.txt").write_text("x", encoding="utf-8")
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(transcript))
    assert any("out of scope" in issue for issue in result["blocking"])

    # Missing permission.
    unapproved = tmp_path / "c"
    _build_intake_from_templates(unapproved, meta_overrides={"permission_status": "not_approved"})
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(unapproved))
    assert any("approved_for_testing" in issue for issue in result["blocking"])

    # Pupil-data marker.
    pupil = tmp_path / "d"
    _build_intake_from_templates(pupil, meta_overrides={"contains_real_pupil_data": True})
    result = audit("real_capture_grade2_raw", spec=_tmp_spec(pupil))
    assert any("contains_real_pupil_data" in issue for issue in result["blocking"])


# --- Real intake folders: empty, .gitkeep only, gitignored ---------------------


def test_real_intake_folders_contain_only_gitkeep():
    for directory in (REAL_IMAGES_DIR, REAL_EXPECTED_DIR, REAL_METADATA_DIR):
        contents = [p.name for p in directory.iterdir()]
        assert contents == [".gitkeep"], f"{directory} must stay empty: {contents}"


def test_empty_real_intake_audit_state_unchanged():
    result = audit("real_capture_grade2_raw")
    assert result["images_present"] is False
    assert result["blocking"] == []
    assert result["ready"] is False  # cleanly empty, not ready, not an error


def test_real_intake_content_is_gitignored():
    for probe in (
        "samples/real_rawbraille_images/rb_capture_001.png",
        "samples/real_rawbraille_expected/rb_capture_001.braille",
        "samples/real_rawbraille_metadata/rb_capture_001.json",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", probe], capture_output=True, timeout=10
        )
        assert result.returncode == 0, f"{probe} is not gitignored"


# --- Contract + scope -----------------------------------------------------------


def test_ocr_contract_unchanged():
    response = run_ocr(OcrRequest(**make_payload()))
    assert set(response.model_dump().keys()) == EXPECTED_RESPONSE_KEYS


def test_still_no_grade2_english_translation():
    # The built-in fallback still ignores contractions, and the *default*
    # Liblouis table remains Grade 1 (Grade 2 is an explicit opt-in via
    # LIBLOUIS_TABLE since Stage 3D-I1).
    from app.core.config import Settings
    from app.translation.fallback_translator import back_translate_unicode_lines

    and_contraction = dots_to_unicode_char(frozenset({1, 2, 3, 4, 6}))
    outcome = back_translate_unicode_lines([and_contraction])
    assert "and" not in outcome.text.lower()
    assert "g1" in Settings.model_fields["liblouis_table"].default.lower()
