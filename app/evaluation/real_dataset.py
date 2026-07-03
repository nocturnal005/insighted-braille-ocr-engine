"""Real anonymised photograph dataset support (Stage 3D-E).

The real-photo dataset lives in local-only, gitignored folders:

    samples/real_anonymised_images/        *.png / *.jpg / *.jpeg
    samples/real_anonymised_ground_truth/  <stem>.txt
    samples/real_anonymised_metadata/      <stem>.json

Every sample needs a matching ground-truth file; metadata is strongly
recommended and gates evaluation:

* ``permission_status: "not_approved"`` samples are always skipped.
* ``crop_quality: "includes_non_braille"`` samples are skipped unless
  explicitly allowed (they violate the Braille-only capture rule).
* ``braille_type: "ueb_grade_2"`` samples are reported separately - the
  engine does not claim Grade 2 support.

This module also screens file names for unsafe patterns (pupil-, school-,
or assessment-identifying text) so that reports never propagate sensitive
names. It never deletes or modifies files - callers report warnings only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATASET_NAME = "real_anonymised"
IMAGES_DIR = Path("samples/real_anonymised_images")
TRUTH_DIR = Path("samples/real_anonymised_ground_truth")
METADATA_DIR = Path("samples/real_anonymised_metadata")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Fields and allowed values for the per-sample metadata JSON.
METADATA_ALLOWED_VALUES: dict[str, set[str]] = {
    "source_type": {"anonymised_school_sample", "synthetic_printout", "demo_sample", "other"},
    "braille_type": {"ueb_grade_1", "ueb_grade_2", "unknown"},
    "capture_method": {"phone_photo", "scanner", "screenshot", "other"},
    "lighting": {"good_even", "directional", "shadowed", "low_light", "unknown"},
    "contrast": {"high", "medium", "low", "unknown"},
    "skew": {"none", "mild", "moderate", "severe", "unknown"},
    "crop_quality": {"braille_only", "extra_margin", "includes_non_braille", "unknown"},
    "permission_status": {"synthetic", "anonymised_only", "approved_for_testing", "not_approved"},
}

GROUPING_FIELDS = (
    "lighting",
    "contrast",
    "skew",
    "capture_method",
    "braille_type",
    "crop_quality",
)

# File-name patterns that suggest pupil-, school-, or assessment-identifying
# content. Case-insensitive; matched against the file stem.
UNSAFE_NAME_PATTERNS: list[tuple[str, str]] = [
    (r"student", "contains 'student'"),
    (r"pupil", "contains 'pupil'"),
    (r"exam", "contains 'exam'"),
    (r"(?<![a-z])test(?![a-z])", "contains 'test'"),
    (r"school", "contains 'school'"),
    (r"assessment", "contains 'assessment'"),
    (r"homework", "contains 'homework'"),
    (r"(?<![a-z])name(?![a-z])", "contains 'name'"),
    (r"\s", "contains spaces (personal-looking name?)"),
    (r"\d{1,2}[-_.]\d{1,2}[-_.](19|20)\d{2}", "date-of-birth style string"),
    (r"(19|20)\d{2}[-_.]\d{1,2}[-_.]\d{1,2}", "date style string"),
    (r"[a-z0-9._%+-]+@[a-z0-9.-]+", "email-like string"),
    (r"^[A-Z][a-z]+[-_][A-Z][a-z]+$", "looks like Firstname-Surname"),
]


def unsafe_name_reasons(stem: str) -> list[str]:
    """Reasons a file stem looks unsafe; empty when it looks fine."""
    reasons = []
    for pattern, reason in UNSAFE_NAME_PATTERNS:
        flags = 0 if pattern.startswith("^[A-Z]") else re.IGNORECASE
        if re.search(pattern, stem, flags):
            reasons.append(reason)
    return reasons


@dataclass
class RealSample:
    sample_id: str
    image_path: Path
    truth_path: Path | None
    metadata: dict | None
    metadata_path: Path | None
    skip_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def evaluable(self) -> bool:
        return not self.skip_reasons

    @property
    def safe_label(self) -> str:
        """Sample id, masked when the file name itself looks unsafe."""
        if unsafe_name_reasons(self.sample_id):
            return "(id withheld - unsafe file name)"
        return self.sample_id


def load_metadata(path: Path) -> tuple[dict | None, list[str]]:
    """Load one metadata JSON. Returns (metadata, issues)."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"metadata unreadable ({type(error).__name__})"]
    if not isinstance(metadata, dict):
        return None, ["metadata is not a JSON object"]
    return metadata, validate_metadata(metadata)


def validate_metadata(metadata: dict) -> list[str]:
    """Non-fatal schema issues for one metadata record."""
    issues = []
    for fieldname, allowed in METADATA_ALLOWED_VALUES.items():
        value = metadata.get(fieldname)
        if value is None:
            issues.append(f"missing field '{fieldname}'")
        elif value not in allowed:
            issues.append(f"field '{fieldname}' has unexpected value '{value}'")
    dot_size = metadata.get("dot_size_px_estimate")
    if dot_size is not None and not isinstance(dot_size, (int, float)):
        issues.append("dot_size_px_estimate must be a number or null")
    return issues


def dot_size_bucket(metadata: dict | None) -> str:
    """Bucket the estimated dot size for grouped reporting."""
    value = (metadata or {}).get("dot_size_px_estimate")
    if not isinstance(value, (int, float)):
        return "unknown"
    if value < 6:
        return "under_6px"
    if value <= 10:
        return "6_to_10px"
    return "over_10px"


def discover_samples(
    images_dir: Path,
    truth_dir: Path,
    metadata_dir: Path,
    *,
    allow_non_braille_crop: bool = False,
) -> list[RealSample]:
    """Find and gate every image in the real dataset (never modifies files)."""
    samples: list[RealSample] = []
    if not images_dir.is_dir():
        return samples

    for image_path in sorted(images_dir.iterdir()):
        if image_path.name == ".gitkeep" or image_path.is_dir():
            continue
        stem = image_path.stem
        sample = RealSample(
            sample_id=stem,
            image_path=image_path,
            truth_path=None,
            metadata=None,
            metadata_path=None,
        )

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            sample.skip_reasons.append(f"unsupported file type '{image_path.suffix}'")

        name_issues = unsafe_name_reasons(stem)
        if name_issues:
            sample.warnings.extend(f"unsafe file name: {issue}" for issue in name_issues)

        truth_path = truth_dir / f"{stem}.txt"
        if truth_path.exists():
            sample.truth_path = truth_path
        else:
            sample.skip_reasons.append("missing ground-truth .txt")

        metadata_path = metadata_dir / f"{stem}.json"
        if metadata_path.exists():
            sample.metadata_path = metadata_path
            metadata, issues = load_metadata(metadata_path)
            sample.metadata = metadata
            sample.warnings.extend(issues)
        else:
            sample.warnings.append("missing metadata .json")

        permission = (sample.metadata or {}).get("permission_status")
        if sample.metadata is None:
            # No usable metadata means no recorded permission: do not evaluate.
            sample.skip_reasons.append("no valid metadata (permission unknown)")
        elif permission == "not_approved":
            sample.skip_reasons.append("permission_status is not_approved")
        elif permission not in METADATA_ALLOWED_VALUES["permission_status"]:
            sample.skip_reasons.append("permission_status missing or invalid")

        crop = (sample.metadata or {}).get("crop_quality")
        if crop == "includes_non_braille" and not allow_non_braille_crop:
            sample.skip_reasons.append(
                "crop_quality is includes_non_braille (Braille-only crops required)"
            )

        samples.append(sample)
    return samples
