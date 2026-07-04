"""Manifest schema and validation for rawBraille evaluation datasets (Stage 3D-G5).

A rawBraille dataset (controlled render, synthetic, or - in future - real
photographed/scanned Braille) is described by one manifest entry per sample.
The manifest keeps datasets safe and self-describing without exposing any
Braille content or confidential data: it records paths and safe labels only.

The schema enforces the project's standing safety boundaries as data rules:

* ``evaluation_mode`` may only be ``rawbraille_cell_level`` - English
  translation / CER-WER evaluation modes are deliberately not allowed here, so
  a manifest can never ask for English Grade 2 scoring.
* ``capture_type`` must state whether samples are controlled renders, synthetic,
  or real captures, so reports never blur the two.
* ``sample_id`` is screened for pupil-, school-, or assessment-identifying
  patterns (reusing the real-photo screen) so identifying names cannot leak.

This module validates and reports only; it never reads image or Braille
content and never modifies files.
"""

from __future__ import annotations

from app.evaluation.real_dataset import unsafe_name_reasons

# Whether the samples are locally rendered, synthetic, or real captures.
CAPTURE_TYPES = {"controlled_render", "synthetic", "real_capture"}

# rawBraille validation is cell-level only. English transcription / CER-WER
# evaluation modes are intentionally absent so no manifest can request them.
EVALUATION_MODES = {"rawbraille_cell_level"}

GRADE_MODES = {"ueb_grade_1", "ueb_grade_2", "unknown"}

SOURCE_TYPES = {
    "controlled_ukaaf_grade2",
    "synthetic",
    "real_photo",
    "real_scan",
    "other",
}

# Every manifest entry must carry these fields.
REQUIRED_FIELDS = (
    "sample_id",
    "image_path",
    "expected_rawbraille_path",
    "dataset_category",
    "capture_type",
    "source_type",
    "consent_or_safety_note",
    "grade_mode",
    "evaluation_mode",
)

_ENUM_FIELDS = {
    "capture_type": CAPTURE_TYPES,
    "evaluation_mode": EVALUATION_MODES,
    "grade_mode": GRADE_MODES,
    "source_type": SOURCE_TYPES,
}


def is_safe_sample_id(sample_id: str) -> bool:
    """True when the sample id shows no pupil/school/assessment patterns."""
    return not unsafe_name_reasons(str(sample_id))


def validate_entry(entry: dict) -> list[str]:
    """Non-fatal issues for one manifest entry (empty list = valid).

    Never raises on content; it only inspects the manifest fields.
    """
    if not isinstance(entry, dict):
        return ["entry is not a JSON object"]

    issues: list[str] = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in entry or entry[field_name] in (None, ""):
            issues.append(f"missing field '{field_name}'")

    for field_name, allowed in _ENUM_FIELDS.items():
        value = entry.get(field_name)
        if value is not None and value not in allowed:
            issues.append(f"field '{field_name}' has unexpected value '{value}'")

    sample_id = entry.get("sample_id")
    if sample_id is not None:
        for reason in unsafe_name_reasons(str(sample_id)):
            issues.append(f"sample_id looks unsafe: {reason}")

    # A rawBraille-only dataset must not require an English transcript.
    if entry.get("requires_english_transcript"):
        issues.append(
            "requires_english_transcript is set - English transcripts are out "
            "of scope for rawBraille cell-level evaluation"
        )
    return issues


def validate_manifest(entries: list[dict]) -> dict:
    """Validate a list of manifest entries. Returns a safe summary dict.

    Contains counts, per-entry issues keyed by safe label, and no Braille or
    image content.
    """
    per_entry: list[dict] = []
    valid = 0
    for index, entry in enumerate(entries):
        issues = validate_entry(entry)
        raw_id = str(entry.get("sample_id", f"entry_{index}")) if isinstance(entry, dict) else f"entry_{index}"
        label = raw_id if is_safe_sample_id(raw_id) else "(id withheld - unsafe)"
        per_entry.append({"label": label, "issues": issues})
        if not issues:
            valid += 1
    return {
        "entries": len(entries),
        "valid": valid,
        "invalid": len(entries) - valid,
        "per_entry": per_entry,
    }
