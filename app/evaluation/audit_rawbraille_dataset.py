"""rawBraille dataset readiness audit (Stage 3D-G5).

Usage:
    python -m app.evaluation.audit_rawbraille_dataset --dataset real_capture_grade2_raw
    python -m app.evaluation.audit_rawbraille_dataset --dataset ukaaf_grade2_raw
    python -m app.evaluation.audit_rawbraille_dataset --dataset real_capture_grade2_raw \\
        --manifest path/to/manifest.json

Checks whether a rawBraille dataset (controlled render now, real photographed/
scanned Braille later) is ready for cell-level evaluation - without exposing
any confidential content. It reports, and never modifies or deletes files:

- images exist (an empty intake folder is a normal, clearly-reported state)
- every image has its expected ``.braille`` file
- NO English transcript is required or requested (rawBraille evaluation is
  cell-level only; Grade 2 English transcription is out of scope)
- sample IDs / file names carry no pupil-, school-, or assessment-identifying
  patterns
- per-sample metadata has safe labels and explicit permission
- an optional dataset manifest validates against the Stage 3D-G5 schema
- sample folders are covered by .gitignore (generated/real material stays
  local-only)
- the dataset is clearly marked controlled_render / synthetic / real_capture

Exit codes: 0 = ready (or empty intake, which is expected), 2 = blocking
issues found. The audit prints metrics and safe labels only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from app.evaluation.audit_dataset import _image_checks
from app.evaluation.rawbraille_dataset import (
    DATASETS,
    EXPECTED_SUFFIX,
    SUPPORTED_EXTENSIONS,
    discover_dataset,
    get_spec,
)
from app.evaluation.rawbraille_manifest import validate_manifest
from app.evaluation.real_dataset import unsafe_name_reasons

SCOPE_NOTE = (
    "rawBraille cell-level evaluation only. No English transcript is required "
    "or accepted; Grade 2 English transcription is out of scope. OCR output "
    "is draft-only and requires QTVI/Braille-literate specialist verification."
)


def _is_gitignored(path: Path) -> bool | None:
    """True/False when git answers; None when it cannot be verified.

    ``git check-ignore -q`` exits 0 (ignored), 1 (not ignored), or 128
    (error - e.g. the path lies outside the repository, as test temp
    folders do). Only a definite 0/1 is trusted.
    """
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def audit(
    dataset: str,
    manifest_path: Path | None = None,
    spec=None,
) -> dict:
    """Run all readiness checks. Returns a safe summary dict (no content).

    ``spec`` overrides the registry lookup (used by tests to audit a temporary
    dataset folder without touching the real local-only locations).
    """
    if spec is None:
        spec = get_spec(dataset)
    blocking: list[str] = []
    warnings: list[str] = []

    samples = discover_dataset(spec)
    images_present = bool(samples)

    # --- English transcripts must not exist in the expected folder ----------
    if spec.expected_dir.is_dir():
        transcripts = [
            p.name
            for p in spec.expected_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        ]
        if transcripts:
            blocking.append(
                f"{len(transcripts)} .txt file(s) in the expected folder - "
                "English transcripts are out of scope; only .braille cell "
                "files are accepted"
            )

    # --- Per-sample checks ---------------------------------------------------
    for sample in samples:
        label = sample.safe_label
        if sample.expected_path is None:
            blocking.append(f"{label}: missing expected {EXPECTED_SUFFIX} file")
        for reason in unsafe_name_reasons(sample.sample_id):
            blocking.append(
                f"(id withheld): unsafe sample id ({reason}) - rename/anonymise "
                "before evaluation"
            )
        metadata = sample.metadata or {}
        if sample.metadata is None:
            blocking.append(f"{label}: no valid metadata (permission unknown)")
        elif metadata.get("permission_status") != "approved_for_testing":
            blocking.append(f"{label}: permission_status is not approved_for_testing")
        for safety_flag in (
            "contains_real_pupil_data",
            "contains_live_assessment_material",
        ):
            if metadata.get(safety_flag) is True:
                blocking.append(f"{label}: {safety_flag}=true - not allowed")
        if metadata.get("requires_english_transcript"):
            blocking.append(
                f"{label}: requires_english_transcript set - out of scope for "
                "rawBraille evaluation"
            )
        capture = metadata.get("capture_type")
        if capture is not None and capture != spec.capture_type:
            warnings.append(
                f"{label}: metadata capture_type '{capture}' differs from the "
                f"dataset's '{spec.capture_type}'"
            )
        for issue in sample.warnings:
            warnings.append(f"{label}: {issue}")
        for note in _image_checks(sample.image_path):
            warnings.append(f"{label}: {note}")

    # --- Optional manifest ----------------------------------------------------
    manifest_summary = None
    if manifest_path is not None:
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            blocking.append(f"manifest unreadable ({type(error).__name__})")
            entries = None
        if entries is not None:
            if not isinstance(entries, list):
                blocking.append("manifest must be a JSON list of entries")
            else:
                manifest_summary = validate_manifest(entries)
                for item in manifest_summary["per_entry"]:
                    for issue in item["issues"]:
                        blocking.append(f"manifest {item['label']}: {issue}")

    # --- gitignore coverage ----------------------------------------------------
    gitignore_ok = True
    for directory in (spec.images_dir, spec.expected_dir, spec.metadata_dir):
        ignored = _is_gitignored(directory / "___audit_probe___.png")
        if ignored is False:
            gitignore_ok = False
            blocking.append(
                f"{directory} is NOT covered by .gitignore - sample material "
                "could be committed"
            )
        elif ignored is None:
            warnings.append(f"could not verify .gitignore coverage for {directory}")

    ready = images_present and not blocking
    return {
        "dataset": spec.name,
        "capture_type": spec.capture_type,
        "grade_mode": spec.grade_mode,
        "evaluation_mode": spec.evaluation_mode,
        "english_transcript_required": False,
        "samples": len(samples),
        "evaluable": sum(1 for s in samples if s.evaluable),
        "images_present": images_present,
        "gitignore_ok": gitignore_ok,
        "blocking": blocking,
        "warnings": warnings,
        "manifest": manifest_summary,
        "ready": ready,
        "note": SCOPE_NOTE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        required=True,
        help="rawBraille dataset to audit (all are cell-level only).",
    )
    parser.add_argument(
        "--manifest",
        metavar="PATH",
        help="Optional dataset manifest JSON to validate (list of entries).",
    )
    args = parser.parse_args(argv)

    result = audit(args.dataset, Path(args.manifest) if args.manifest else None)

    print(
        f"rawBraille dataset audit: {result['dataset']} "
        f"(capture_type={result['capture_type']}, grade_mode={result['grade_mode']}, "
        f"evaluation_mode={result['evaluation_mode']})"
    )
    print(
        f"samples={result['samples']} evaluable={result['evaluable']} "
        f"gitignore_ok={result['gitignore_ok']}"
    )

    if not result["images_present"]:
        print(
            "\nNo samples present. For a real-capture dataset this is the "
            "expected state until safe, anonymised, approved physical samples "
            "are added. Add images + expected .braille files + metadata, then "
            "re-run this audit."
        )

    if result["blocking"]:
        print("\nBLOCKING issues (must fix before evaluation):")
        for issue in result["blocking"]:
            print(f"  BLOCK: {issue}")
    if result["warnings"]:
        print("\nwarnings (review before evaluation):")
        for issue in result["warnings"]:
            print(f"  WARN: {issue}")
    if result["manifest"] is not None:
        print(
            f"\nmanifest: entries={result['manifest']['entries']} "
            f"valid={result['manifest']['valid']} invalid={result['manifest']['invalid']}"
        )

    verdict = "READY" if result["ready"] else (
        "EMPTY (expected until samples are added)"
        if not result["images_present"] and not result["blocking"]
        else "NOT READY"
    )
    print(f"\nverdict: {verdict}")
    print(f"note: {SCOPE_NOTE}")
    print("The audit only reports - it never deletes or modifies files.")

    if result["blocking"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
