"""Stage-by-stage diagnostic probe for one Braille image (Stage 3D-J1).

Answers, for a single PNG/JPEG capture, how far the existing OCR pipeline
gets and where it stops:

    L0  safe rejection (decode refused, or no usable dot candidates)
    L1  dot candidates detected
    L2  rows (Braille lines) separated
    L3  cells formed
    L4  rawBraille produced
    L5  Grade 2 draft produced via Liblouis (configured table is Grade 2)
    L6  scored against .braille ground truth (assigned by the caller after
        a valid, gated ground-truth comparison - never by this module)

The probe runs the production ``run_ocr`` once (contract-level truth:
confidence, flags, draft emptiness) and then replays the pipeline stages
directly to observe internals that the /ocr contract deliberately does not
expose (raw candidate counts, grouping quality, which translator ran).
It never modifies the pipeline and never raises: every failure is recorded
as a classification.

Safety: ``to_safe_dict()`` is the only serialisation surface and contains
counts, scores, and category names only - never rawBraille content, draft
text, expected text, flag reason prose, or file names. The in-memory
``raw_braille``/``draft_text`` attributes exist solely so a gated caller
can score against ground truth; report writers must use ``to_safe_dict()``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path

from app.core.config import get_settings
from app.evaluation.metrics import (
    character_error_rate,
    normalise_text,
    word_error_rate,
)
from app.evaluation.rawbraille_metrics import sample_metrics
from app.models.requests import OcrRequest
from app.ocr.flags import (
    CATEGORY_LINE_ORDER_UNCERTAINTY,
    CATEGORY_LOW_OCR_CONFIDENCE,
)
from app.ocr.image_decode import ImageDecodeError, decode_data_url
from app.ocr.line_reconstruction import reconstruct_lines
from app.ocr.braille_decode import token_lines_to_unicode

# The variant-selection helper is deliberately reused from the production
# pipeline so the probe sees exactly what /ocr sees. It is module-private
# there; if it is ever renamed this import - and only this import - breaks.
from app.ocr.capture_normalization import detect_with_normalisation
from app.ocr.dot_evidence import refine_grouping
from app.ocr.pipeline import _select_variant, run_ocr
from app.translation.fallback_translator import back_translate_unicode_lines
from app.translation.liblouis_adapter import is_grade2_table, liblouis_back_translate

MIME_BY_EXTENSION = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

STAGE_L0 = "L0"
STAGE_L1 = "L1"
STAGE_L2 = "L2"
STAGE_L3 = "L3"
STAGE_L4 = "L4"
STAGE_L5 = "L5"
STAGE_L6 = "L6"

# Where the pipeline stopped (independent of the highest stage reached).
FAILURE_NONE = "none"
FAILURE_UNSUPPORTED_FILE = "unsupported_file"
FAILURE_READ_ERROR = "read_error"
FAILURE_DECODE_REJECTED = "decode_rejected"
FAILURE_NO_DOT_CANDIDATES = "no_dot_candidates"
FAILURE_DOTS_REJECTED_BY_FILTERS = "dots_rejected_by_filters"
FAILURE_ROW_SEPARATION = "row_separation_failed"
FAILURE_NO_CELLS = "no_cells_formed"
FAILURE_TRANSLATION = "translation_failed"
FAILURE_INTERNAL_ERROR = "internal_error"


@dataclass
class ProbeResult:
    """Per-image diagnostic outcome. Serialise only via ``to_safe_dict()``."""

    stage: str = STAGE_L0
    failure_point: str = FAILURE_NONE

    # Intake
    decode_ok: bool = False
    file_size_bytes: int = 0
    width: int = 0
    height: int = 0

    # Capture normalisation (Stage 3D-L1): what was applied to decode
    capture_rescaled: bool = False
    capture_rotation_applied: int = 0
    capture_cropped: bool = False

    # Grid-evidence re-scoring (Stage 3D-M1)
    evidence_applied: bool = False
    evidence_cells_changed: int = 0
    evidence_cells_recovered: int = 0
    evidence_cells_dropped: int = 0

    # Dot detection (winning variant, exactly as /ocr would select it)
    mode: str = ""
    raw_candidates: int = 0
    accepted_dots: int = 0
    median_dot_radius_px: float = 0.0
    spacing_regularity: float = 0.0
    image_quality: float = 0.0

    # Grouping / lines
    lines_detected: int = 0
    total_cells: int = 0
    grouping_quality: float = 0.0
    line_quality: float = 0.0

    # rawBraille
    rawbraille_nonempty: bool = False
    rawbraille_cell_count: int = 0

    # Translation
    liblouis_enabled: bool = False
    liblouis_table: str = ""
    grade2_table_configured: bool = False
    liblouis_used: bool = False
    grade2_draft_produced: bool = False
    draft_nonempty: bool = False

    # Contract-level truth from the production run_ocr
    confidence: float = 0.0
    flag_categories: list[str] = field(default_factory=list)
    response_rawbraille_nonempty: bool = False
    response_draft_nonempty: bool = False

    # In-memory only - required for gated ground-truth scoring; never
    # serialised. Report writers must go through to_safe_dict().
    raw_braille: str = ""
    draft_text: str = ""

    def to_safe_dict(self) -> dict:
        """Numbers, booleans, and category/stage labels only - no content."""
        return {
            "stage": self.stage,
            "failure_point": self.failure_point,
            "decode_ok": self.decode_ok,
            "file_size_bytes": self.file_size_bytes,
            "width": self.width,
            "height": self.height,
            "capture_rescaled": self.capture_rescaled,
            "capture_rotation_applied": self.capture_rotation_applied,
            "capture_cropped": self.capture_cropped,
            "evidence_applied": self.evidence_applied,
            "evidence_cells_changed": self.evidence_cells_changed,
            "evidence_cells_recovered": self.evidence_cells_recovered,
            "evidence_cells_dropped": self.evidence_cells_dropped,
            "mode": self.mode,
            "raw_candidates": self.raw_candidates,
            "accepted_dots": self.accepted_dots,
            "median_dot_radius_px": round(self.median_dot_radius_px, 2),
            "spacing_regularity": round(self.spacing_regularity, 3),
            "image_quality": round(self.image_quality, 3),
            "lines_detected": self.lines_detected,
            "total_cells": self.total_cells,
            "grouping_quality": round(self.grouping_quality, 3),
            "line_quality": round(self.line_quality, 3),
            "rawbraille_nonempty": self.rawbraille_nonempty,
            "rawbraille_cell_count": self.rawbraille_cell_count,
            "liblouis_enabled": self.liblouis_enabled,
            "liblouis_table": self.liblouis_table,
            "grade2_table_configured": self.grade2_table_configured,
            "liblouis_used": self.liblouis_used,
            "grade2_draft_produced": self.grade2_draft_produced,
            "draft_nonempty": self.draft_nonempty,
            "confidence": round(self.confidence, 3),
            "flag_categories": sorted(self.flag_categories),
            "response_rawbraille_nonempty": self.response_rawbraille_nonempty,
            "response_draft_nonempty": self.response_draft_nonempty,
        }


def _data_url_for(path: Path) -> tuple[str, str] | None:
    """(data_url, mime) or None for an unsupported extension. Reads the file."""
    mime = MIME_BY_EXTENSION.get(path.suffix.lower())
    if mime is None:
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}", mime


def probe_image_file(path: Path) -> ProbeResult:
    """Run the full staged diagnostic on one image file. Never raises."""
    result = ProbeResult()
    settings = get_settings()
    result.liblouis_enabled = settings.liblouis_enabled
    result.liblouis_table = settings.liblouis_table
    result.grade2_table_configured = is_grade2_table(settings.liblouis_table)

    try:
        result.file_size_bytes = path.stat().st_size
    except OSError:
        result.failure_point = FAILURE_UNSUPPORTED_FILE
        return result

    try:
        prepared = _data_url_for(path)
    except OSError:
        # Supported extension but the bytes could not be read (deleted or
        # locked mid-run, sharing violation). Honour the never-raises
        # contract: record it and stop, never surface the file name.
        result.failure_point = FAILURE_READ_ERROR
        return result
    if prepared is None:
        result.failure_point = FAILURE_UNSUPPORTED_FILE
        return result
    data_url, mime = prepared

    # --- Contract-level truth: one production run --------------------------
    try:
        response = run_ocr(
            OcrRequest(
                taskId=f"diagnostic-{path.stem}",
                title="real-capture-diagnostic",
                fileName=f"diagnostic{path.suffix.lower()}",
                mimeType=mime,
                dataUrl=data_url,
            )
        )
        result.confidence = response.confidence
        result.flag_categories = sorted({f.category for f in response.flags})
        result.response_rawbraille_nonempty = bool(response.rawBraille)
        result.response_draft_nonempty = bool(response.draftText)
    except Exception:
        # run_ocr is designed never to raise; if it somehow does, record it
        # and continue with the staged replay, which has its own guards.
        result.failure_point = FAILURE_INTERNAL_ERROR

    # --- Staged replay: observe the internals ------------------------------
    try:
        gray, _ = decode_data_url(
            data_url,
            mime,
            max_bytes=settings.max_image_bytes,
            max_pixels=settings.max_image_pixels,
        )
    except ImageDecodeError:
        result.failure_point = FAILURE_DECODE_REJECTED
        return result
    except Exception:
        result.failure_point = FAILURE_INTERNAL_ERROR
        return result

    result.decode_ok = True
    result.height, result.width = gray.shape[:2]

    try:
        # Same capture-normalisation path as /ocr (Stage 3D-L1): oversized
        # photos are downscaled and a bounded rotation/crop rescue ladder
        # runs when the upright attempt forms no cells.
        normalised = detect_with_normalisation(gray, _select_variant)
        detection, grouping = normalised.detection, normalised.grouping
        result.capture_rescaled = normalised.rescaled
        result.capture_rotation_applied = normalised.rotation_applied
        result.capture_cropped = normalised.cropped
    except Exception:
        result.failure_point = FAILURE_INTERNAL_ERROR
        return result

    result.mode = detection.mode
    result.raw_candidates = detection.raw_candidates
    result.accepted_dots = len(detection.dots)
    result.median_dot_radius_px = detection.median_radius
    result.spacing_regularity = detection.spacing_regularity
    result.image_quality = detection.image_quality

    if result.accepted_dots == 0:
        result.failure_point = (
            FAILURE_DOTS_REJECTED_BY_FILTERS
            if result.raw_candidates > 0
            else FAILURE_NO_DOT_CANDIDATES
        )
        return result
    result.stage = STAGE_L1

    result.lines_detected = len(grouping.lines)
    result.total_cells = grouping.total_cells
    result.grouping_quality = grouping.quality
    result.line_quality = grouping.line_quality

    if result.lines_detected == 0:
        # Dots existed but grouping produced nothing. Distinguish collapsed
        # rows from an empty result by the grouping's own flag categories.
        grouping_categories = {
            (f.category, f.severity) for f in grouping.flags
        }
        if (CATEGORY_LINE_ORDER_UNCERTAINTY, "high") in grouping_categories:
            result.failure_point = FAILURE_ROW_SEPARATION
        elif (CATEGORY_LOW_OCR_CONFIDENCE, "high") in grouping_categories:
            result.failure_point = FAILURE_NO_DOT_CANDIDATES
        else:
            result.failure_point = FAILURE_NO_CELLS
        return result
    result.stage = STAGE_L2

    if result.total_cells == 0:
        result.failure_point = FAILURE_NO_CELLS
        return result
    result.stage = STAGE_L3

    try:
        # Same grid-evidence re-scoring as /ocr (Stage 3D-M1).
        refinement = refine_grouping(detection, grouping)
        if refinement.applied and refinement.cells_changed:
            grouping = dataclass_replace(
                grouping,
                lines=refinement.lines,
                total_cells=refinement.total_cells,
            )
            result.evidence_applied = True
            result.evidence_cells_changed = refinement.cells_changed
            result.evidence_cells_recovered = refinement.cells_recovered
            result.evidence_cells_dropped = refinement.cells_dropped
            result.total_cells = grouping.total_cells
        token_lines, _raw_cells, _reconstruction_flags = reconstruct_lines(grouping)
        unicode_lines = token_lines_to_unicode(token_lines)
        raw_braille = "\n".join(unicode_lines)
    except Exception:
        result.failure_point = FAILURE_INTERNAL_ERROR
        return result

    result.raw_braille = raw_braille
    result.rawbraille_nonempty = bool(raw_braille.strip())
    result.rawbraille_cell_count = sum(
        1 for c in raw_braille if c not in (" ", "\n")
    )
    if not result.rawbraille_nonempty:
        result.failure_point = FAILURE_NO_CELLS
        return result
    result.stage = STAGE_L4

    # --- Translation: which translator actually runs on this machine -------
    try:
        liblouis_text = (
            liblouis_back_translate(raw_braille, settings.liblouis_table)
            if settings.liblouis_enabled
            else None
        )
        if liblouis_text is not None:
            result.liblouis_used = True
            result.draft_text = liblouis_text
        else:
            result.draft_text = back_translate_unicode_lines(unicode_lines).text
    except Exception:
        result.failure_point = FAILURE_TRANSLATION
        return result

    result.draft_nonempty = bool(result.draft_text.strip())
    if not result.draft_nonempty:
        result.failure_point = FAILURE_TRANSLATION
        return result

    result.grade2_draft_produced = (
        result.liblouis_used and result.grade2_table_configured
    )
    if result.grade2_draft_produced:
        result.stage = STAGE_L5
    return result


def score_against_expected(probe: ProbeResult, expected_rawbraille: str) -> dict:
    """Cell-level metrics vs gated .braille ground truth (metrics only).

    The caller is responsible for gating (metadata permission, safe naming,
    real-capture protocol) AND for only scoring a probe that produced
    rawBraille (stage L4+). Scoring is refused for a probe that never
    produced rawBraille: promoting an L0-L3 failure to L6 would falsely
    imply it progressed further than it did. On success the probe is
    promoted to L6. Supplementary English CER/WER is added only when
    Liblouis Grade 2 actually translated both texts.

    Raises ValueError when the probe produced no rawBraille to score.
    """
    if not probe.rawbraille_nonempty:
        raise ValueError(
            "cannot score a probe that produced no rawBraille "
            f"(stopped at {probe.stage}/{probe.failure_point})"
        )
    metrics = sample_metrics(expected_rawbraille, probe.raw_braille)
    scores: dict = {
        "cell_error_rate": round(metrics["cell_error_rate"], 4),
        "rawbraille_cer": round(metrics["rawbraille_cer"], 4),
        "expected_cells": metrics["expected_cells"],
        "predicted_cells": metrics["predicted_cells"],
        "cell_count_mismatch": metrics["cell_count_mismatch"],
        "exact_sample_match": metrics["exact_sample_match"],
        "line_count_mismatch": metrics["line_count_mismatch"],
        "line_reconstruction_accuracy": round(
            metrics["line_reconstruction_accuracy"], 4
        ),
        "english_cer": None,
        "english_wer": None,
    }

    settings = get_settings()
    if (
        settings.liblouis_enabled
        and is_grade2_table(settings.liblouis_table)
        and probe.liblouis_used
    ):
        reference = liblouis_back_translate(
            expected_rawbraille, settings.liblouis_table
        )
        if reference is not None:
            reference = normalise_text(reference)
            hypothesis = normalise_text(probe.draft_text)
            scores["english_cer"] = round(
                character_error_rate(reference, hypothesis), 4
            )
            scores["english_wer"] = round(
                word_error_rate(reference, hypothesis), 4
            )

    probe.stage = STAGE_L6
    return scores
