"""Dataset audit: safety and completeness checks for real validation samples.

Usage:
    python -m app.evaluation.audit_dataset --dataset real_anonymised

Reports, without ever modifying or deleting files:
- image / ground-truth / metadata counts and mismatches
- unsupported file types
- unsafe file names (pupil/school/assessment-style identifiers, dates,
  email-like strings, personal-looking names)
- samples marked not_approved, or with missing/invalid metadata
- image dimensions, size sanity (too small / too large), and rough quality
  indicators (brightness, contrast)

Exit code 0 always (audit is advisory) unless the arguments are invalid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from app.evaluation.real_dataset import (
    DATASET_NAME,
    IMAGES_DIR,
    METADATA_DIR,
    SUPPORTED_EXTENSIONS,
    TRUTH_DIR,
    discover_samples,
    unsafe_name_reasons,
)

# Sanity bounds: a Braille dot needs ~6px, a page shouldn't be a wall poster.
MIN_DIMENSION = 80
MAX_DIMENSION = 8000
LOW_CONTRAST_STD = 8.0


def _image_checks(path: Path) -> list[str]:
    """Dimension and rough quality indicators for one image."""
    notes: list[str] = []
    try:
        with Image.open(path) as img:
            width, height = img.size
            gray = np.asarray(img.convert("L"), dtype=np.float32)
    except Exception as error:  # report, never crash the audit
        return [f"unreadable image ({type(error).__name__})"]

    if min(width, height) < MIN_DIMENSION:
        notes.append(f"very small ({width}x{height}) - dots may be under the ~6px floor")
    if max(width, height) > MAX_DIMENSION:
        notes.append(f"very large ({width}x{height}) - may exceed pixel limits")
    mean = float(gray.mean())
    std = float(gray.std())
    if mean < 40:
        notes.append(f"very dark (mean {mean:.0f}/255)")
    elif mean > 235:
        notes.append(f"very bright (mean {mean:.0f}/255)")
    if std < LOW_CONTRAST_STD:
        notes.append(f"very low contrast (std {std:.1f})")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=[DATASET_NAME], default=DATASET_NAME,
        help="Dataset to audit (currently only real_anonymised)",
    )
    parser.add_argument("--images", help="Override images directory (mainly for tests)")
    parser.add_argument("--truth", help="Override ground-truth directory")
    parser.add_argument("--metadata", help="Override metadata directory")
    args = parser.parse_args(argv)

    images_dir = Path(args.images) if args.images else IMAGES_DIR
    truth_dir = Path(args.truth) if args.truth else TRUTH_DIR
    metadata_dir = Path(args.metadata) if args.metadata else METADATA_DIR

    print(f"Dataset audit: {args.dataset}")
    for label, directory in (("images", images_dir), ("ground truth", truth_dir), ("metadata", metadata_dir)):
        status = "ok" if directory.is_dir() else "MISSING"
        print(f"  {label} dir: {directory} [{status}]")

    def _files(directory: Path, suffixes: set[str]) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.name != ".gitkeep" and p.suffix.lower() in suffixes
        )

    image_files = _files(images_dir, SUPPORTED_EXTENSIONS)
    truth_files = _files(truth_dir, {".txt"})
    metadata_files = _files(metadata_dir, {".json"})
    stray = [
        p.name for p in (images_dir.iterdir() if images_dir.is_dir() else [])
        if p.is_file() and p.name != ".gitkeep" and p.suffix.lower() not in SUPPORTED_EXTENSIONS
    ]

    print(
        f"\ncounts: images={len(image_files)} ground_truth={len(truth_files)} "
        f"metadata={len(metadata_files)}"
    )
    if not image_files and not stray:
        print(
            "\nNo real validation images present. This is expected until safe, "
            "anonymised, approved samples are added - see "
            "docs/real_photo_validation_protocol.md."
        )
        return 0

    warnings = 0

    def warn(message: str) -> None:
        nonlocal warnings
        warnings += 1
        print(f"  WARN: {message}")

    print("\nper-file checks:")
    for name in stray:
        warn(f"{name}: unsupported file type (PNG/JPEG only)")

    samples = discover_samples(images_dir, truth_dir, metadata_dir)
    for sample in samples:
        header_printed = False

        def report(message: str) -> None:
            nonlocal header_printed
            if not header_printed:
                print(f"  {sample.sample_id}:")
                header_printed = True
            warn(message)

        for reason in unsafe_name_reasons(sample.sample_id):
            report(f"unsafe file name - {reason}; rename before evaluation")
        if sample.truth_path is None:
            report("missing ground-truth .txt")
        if sample.metadata_path is None:
            report("missing metadata .json (sample will be skipped: permission unknown)")
        for issue in sample.warnings:
            if issue != "missing metadata .json" and not issue.startswith("unsafe file name"):
                report(issue)
        permission = (sample.metadata or {}).get("permission_status")
        if permission == "not_approved":
            report("permission_status=not_approved - will be skipped by evaluation")
        if (sample.metadata or {}).get("crop_quality") == "includes_non_braille":
            report("crop_quality=includes_non_braille - crop to the Braille-only area")
        if (sample.metadata or {}).get("braille_type") == "ueb_grade_2":
            report("braille_type=ueb_grade_2 - reported separately; Grade 2 is not supported")
        for note in _image_checks(sample.image_path):
            report(note)

    orphan_truth = {p.stem for p in truth_files} - {s.sample_id for s in samples}
    for stem in sorted(orphan_truth):
        warn(f"ground truth '{stem}.txt' has no matching image")
    orphan_metadata = {p.stem for p in metadata_files} - {s.sample_id for s in samples}
    for stem in sorted(orphan_metadata):
        warn(f"metadata '{stem}.json' has no matching image")

    evaluable = sum(1 for s in samples if s.evaluable)
    print(
        f"\nsummary: samples={len(samples)} evaluable={evaluable} "
        f"skipped={len(samples) - evaluable} warnings={warnings}"
    )
    print(
        "note: the audit only reports - it never deletes or modifies files. "
        "Fix warnings, then run: python -m app.evaluation.run_evaluation "
        "--dataset real_anonymised"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
