"""Local-only UKAAF Grade 2 rawBraille validation dataset (Stage 3D-G3).

Controlled Braille images rendered locally from UKAAF Grade 2 BRF cells, paired
with expected ``rawBraille`` decoded from the same BRF (see
``app/evaluation/braille_ascii.py``). The dataset is used to score the visual
pipeline at the cell level only - never English draft-text accuracy.

The dataset lives in local-only, gitignored folders (UKAAF source material is
copyrighted and must never be committed):

    _external_sources/ukaaf/generated_grade2_rawbraille/images/    *.png
    _external_sources/ukaaf/generated_grade2_rawbraille/expected/  <stem>.braille
    _external_sources/ukaaf/generated_grade2_rawbraille/metadata/  <stem>.json

Gating (mirrors the real-photo dataset's safety stance):

* an image with no matching ``<stem>.braille`` expected file is skipped;
* ``permission_status`` must be ``approved_for_testing`` (anything else,
  including missing metadata, is skipped - permission must be explicit);
* file stems are screened for identifying patterns so reports never propagate
  sensitive names.

An empty or missing dataset is the expected state on a fresh checkout (the
generated files are local-only); callers exit cleanly in that case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.real_dataset import unsafe_name_reasons

DATASET_NAME = "ukaaf_grade2_raw"
BASE_DIR = Path("_external_sources/ukaaf/generated_grade2_rawbraille")
IMAGES_DIR = BASE_DIR / "images"
EXPECTED_DIR = BASE_DIR / "expected"
METADATA_DIR = BASE_DIR / "metadata"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
EXPECTED_SUFFIX = ".braille"

# Allowed metadata values for this controlled dataset. Distinct from the
# real-photo dataset: these samples are locally rendered from BRF cells.
METADATA_ALLOWED_VALUES: dict[str, set] = {
    "source_type": {"controlled_ukaaf_grade2"},
    "braille_type": {"ueb_grade_2"},
    "capture_method": {"rendered"},
    "permission_status": {"approved_for_testing"},
    "contains_real_pupil_data": {False},
    "contains_live_assessment_material": {False},
}


@dataclass
class RawBrailleSample:
    sample_id: str
    image_path: Path
    expected_path: Path | None
    metadata: dict | None
    metadata_path: Path | None
    category: str = "unknown"
    variant: str = "clean"
    skip_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def evaluable(self) -> bool:
        return not self.skip_reasons

    @property
    def safe_label(self) -> str:
        if unsafe_name_reasons(self.sample_id):
            return "(id withheld - unsafe file name)"
        return self.sample_id

    def expected_rawbraille(self) -> str:
        """Read the expected rawBraille (local-only). Never logged by callers."""
        return self.expected_path.read_text(encoding="utf-8")


def _validate_metadata(metadata: dict) -> list[str]:
    issues: list[str] = []
    for fieldname, allowed in METADATA_ALLOWED_VALUES.items():
        if fieldname not in metadata:
            issues.append(f"missing field '{fieldname}'")
        elif metadata[fieldname] not in allowed:
            issues.append(f"field '{fieldname}' has unexpected value")
    return issues


def discover_samples(
    images_dir: Path = IMAGES_DIR,
    expected_dir: Path = EXPECTED_DIR,
    metadata_dir: Path = METADATA_DIR,
) -> list[RawBrailleSample]:
    """Find and gate every rendered Grade 2 sample (never modifies files)."""
    samples: list[RawBrailleSample] = []
    if not images_dir.is_dir():
        return samples

    for image_path in sorted(images_dir.iterdir()):
        if image_path.name == ".gitkeep" or image_path.is_dir():
            continue
        stem = image_path.stem
        sample = RawBrailleSample(
            sample_id=stem,
            image_path=image_path,
            expected_path=None,
            metadata=None,
            metadata_path=None,
        )

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            sample.skip_reasons.append(f"unsupported file type '{image_path.suffix}'")

        for issue in unsafe_name_reasons(stem):
            sample.warnings.append(f"unsafe file name: {issue}")

        expected_path = expected_dir / f"{stem}{EXPECTED_SUFFIX}"
        if expected_path.exists():
            sample.expected_path = expected_path
        else:
            sample.skip_reasons.append("missing expected .braille file")

        metadata_path = metadata_dir / f"{stem}.json"
        if metadata_path.exists():
            sample.metadata_path = metadata_path
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                metadata = None
                sample.warnings.append(f"metadata unreadable ({type(error).__name__})")
            if isinstance(metadata, dict):
                sample.metadata = metadata
                sample.warnings.extend(_validate_metadata(metadata))
                sample.category = str(metadata.get("category", "unknown"))
                sample.variant = str(metadata.get("variant", "clean"))
            else:
                sample.warnings.append("metadata is not a JSON object")
        else:
            sample.warnings.append("missing metadata .json")

        permission = (sample.metadata or {}).get("permission_status")
        if sample.metadata is None:
            sample.skip_reasons.append("no valid metadata (permission unknown)")
        elif permission != "approved_for_testing":
            sample.skip_reasons.append("permission_status is not approved_for_testing")

        samples.append(sample)
    return samples
