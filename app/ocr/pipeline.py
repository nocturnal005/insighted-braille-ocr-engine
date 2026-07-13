"""OCR pipeline orchestration.

Every run produces a contract-valid OcrResponse. Failures never crash and
never fabricate text: they return an empty draftText, confidence 0, and
clear uncertainty flags. All output is draft-only and requires QTVI or
Braille-literate specialist verification downstream in InsightEd AI.

Logging policy: metadata only (request ids, counts, durations, confidence).
Task ids are logged only as a short non-reversible hash. Never image data,
transcription text, titles, file names, raw task ids, or pupil data.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace as dataclass_replace
from time import perf_counter
from uuid import uuid4

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.requests import OcrRequest
from app.models.responses import Flag, OcrResponse, PageResult
from app.ocr.braille_decode import token_lines_to_unicode
from app.ocr.capture_normalization import detect_with_normalisation, _quick_readability
from app.ocr.cell_grouping import GroupingResult, group_dots
from app.ocr.dot_evidence import refine_grouping
from app.ocr.confidence import (
    EMBOSS_MODE_CAP,
    FALLBACK_TRANSLATION_CAP,
    LATTICE_RECOVERY_CAP,
    TEMPLATE_READER_CAP,
    combined_confidence,
    dot_size_cap,
    noise_ratio_factor,
)
from app.ocr.dot_detection import (
    DetectionOutcome,
    detect_variant,
    selection_flags,
    strict_variant,
)
from app.ocr.flags import (
    CATEGORY_LOW_IMAGE_QUALITY,
    CATEGORY_LOW_OCR_CONFIDENCE,
    CATEGORY_POSSIBLE_CONTRACTION_ISSUE,
    CATEGORY_SUBJECT_SPECIFIC_TERM,
    dedupe_flags,
    make_flag,
)
from app.ocr.image_decode import SUPPORTED_MIME_TYPES, ImageDecodeError, decode_data_url
from app.ocr.line_reconstruction import reconstruct_lines
from app.ocr.preprocessing import MODE_EMBOSS
from app.ocr.template_reader import read_page as template_read_page
from app.translation.fallback_translator import back_translate_unicode_lines
from app.translation.liblouis_adapter import is_grade2_table, liblouis_back_translate

logger = get_logger(__name__)

_BRAILLE_BASE = 0x2800

# Stage 3D-N1 template-reader gate. The full-resolution template reader runs
# only when the blob path's decode is unreadable — either zero cells or a
# draft the Grade 1 fallback can barely read (real embossed photos, with or
# without handwriting ink, land at 0.00-0.60; every synthetic/clean input
# measures >=0.94, so they never trigger it). It then replaces the blob decode
# only when it reads materially better, so a page the blob path already reads
# well is never touched.
_TEMPLATE_TRIGGER_READABILITY = 0.75
_TEMPLATE_WIN_MARGIN = 0.10


def _new_request_id() -> str:
    return "ocr_" + uuid4().hex


def _select_variant(variants) -> tuple[DetectionOutcome, GroupingResult]:
    """Detect and group on every preprocessing variant, keep the best.

    The winner is the variant whose dots actually form a Braille grid — a
    blend of grid-fit quality, per-dot shape quality, and spacing
    regularity. Shape metrics alone are not enough: on an embossed photo the
    dark path picks up shadow crescents that look dot-like but sit off the
    true centres, decoding to garbage; the grid-fit term catches that.
    Strict `>` keeps the first (dark) variant on ties, preserving the
    original behaviour for clean scans.
    """
    detections = [detect_variant(variant) for variant in variants]
    # Stage 3D-G2: on pages with evidence of noise (size filter rejected
    # extra marks), also offer a strict candidate with low-confidence dots
    # dropped. Specks that slip the size gate score poorly on circularity /
    # size consistency; removing them lets the true grid win the same
    # grid-fit scoring below, instead of the noise garbling cell grouping.
    detections.extend(
        strict for strict in (strict_variant(d) for d in list(detections))
        if strict is not None
    )
    candidates: list[tuple[DetectionOutcome, GroupingResult]] = [
        (detection, group_dots(detection.dots)) for detection in detections
    ]
    max_dots = max((len(d.dots) for d, _ in candidates), default=0)

    best_detection = DetectionOutcome()
    best_grouping = GroupingResult()
    best_score = -1.0
    for detection, grouping in candidates:
        # Count factor: absolute floor (a handful of dots is never a page)
        # plus a relative term so a variant that reconstructs only a small
        # fraction of what another variant sees cannot win on shape alone.
        # The 0.6 headroom keeps a noisy variant's inflated count (false
        # positives) from suppressing a cleaner variant with fewer dots.
        count_factor = min(1.0, len(detection.dots) / 6.0)
        if max_dots > 0:
            count_factor *= min(1.0, len(detection.dots) / (0.6 * max_dots))
        score = count_factor * (
            0.45 * grouping.quality
            + 0.30 * detection.quality
            + 0.25 * detection.spacing_regularity
        )
        if score > best_score:
            best_score = score
            best_detection = detection
            best_grouping = grouping
    return best_detection, best_grouping


def _task_ref(task_id: str) -> str:
    """Short non-reversible reference so logs can be correlated with a task
    without ever recording the raw taskId (which the caller controls and
    could contain identifying text)."""
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]


def _failure_response(request_id: str, flags: list[Flag]) -> OcrResponse:
    flags = dedupe_flags(flags)
    return OcrResponse(
        draftText="",
        confidence=0.0,
        rawBraille=None,
        rawCells=[],
        providerRequestId=request_id,
        flags=flags,
        pageResults=[PageResult(pageNumber=1, text="", confidence=0.0, flags=flags)],
    )


def _safe_mime_for_log(mime: str) -> str:
    return mime if mime in SUPPORTED_MIME_TYPES else "unsupported"


def run_ocr(request: OcrRequest) -> OcrResponse:
    settings = get_settings()
    request_id = _new_request_id()
    started = perf_counter()

    # --- Stage 2: image intake ------------------------------------------------
    try:
        gray, byte_count = decode_data_url(
            request.dataUrl,
            request.mimeType,
            max_bytes=settings.max_image_bytes,
            max_pixels=settings.max_image_pixels,
        )
    except ImageDecodeError as error:
        logger.info(
            "ocr_rejected request_id=%s task_ref=%s stage=decode mime=%s",
            request_id,
            _task_ref(request.taskId),
            _safe_mime_for_log(request.mimeType),
        )
        return _failure_response(request_id, [error.flag])
    except Exception as exc:
        logger.warning(
            "ocr_decode_error request_id=%s type=%s", request_id, type(exc).__name__
        )
        return _failure_response(
            request_id,
            [
                make_flag(
                    text="",
                    reason="The uploaded image could not be decoded safely.",
                    category=CATEGORY_LOW_IMAGE_QUALITY,
                    severity="high",
                )
            ],
        )

    try:
        flags: list[Flag] = []

        # --- Stage 3: preprocessing + capture normalisation (Stage 3D-L1) ------
        # Oversized phone photos are downscaled to the calibrated dot scale;
        # when the upright attempt forms no cells, a bounded rescue ladder
        # retries rotations and a background crop. Images that already decode
        # are never altered.
        normalised = detect_with_normalisation(gray, _select_variant)

        # --- Stages 4-5: dot detection + grouping (best of dark/emboss) ---------
        detection, grouping = normalised.detection, normalised.grouping
        flags.extend(normalised.flags)

        # --- Stage 3D-N1: full-resolution template-reader rescue ----------------
        # When the blob pipeline's decode is unreadable (zero cells, or a draft
        # the Grade 1 fallback can barely read — the signature of a real
        # embossed photo, with or without handwriting ink), re-read the
        # full-resolution image by self-calibrated template matching. It
        # recovers faint embossed dots the downscaled blob path cannot. Gated on
        # the blob decode being unreadable AND the template read being clearly
        # more readable, so a page the blob path already reads well (every
        # synthetic/clean input, which measures >=0.94) is never touched. Fails
        # closed: read_page returns None unless the matched dots form a
        # self-consistent, readable lattice.
        used_template_reader = False
        blob_readability = (
            _quick_readability(grouping) if grouping.total_cells else 0.0
        )
        if blob_readability < _TEMPLATE_TRIGGER_READABILITY:
            template_result = template_read_page(gray)
            if template_result is not None and (
                _quick_readability(template_result.grouping)
                > blob_readability + _TEMPLATE_WIN_MARGIN
            ):
                detection = template_result.detection
                grouping = template_result.grouping
                flags.extend(template_result.flags)
                used_template_reader = True

        dots = detection.dots
        detection_quality = detection.quality
        image_quality = detection.image_quality
        flags.extend(selection_flags(detection))

        # Image-quality flags reflect the variant that was actually used.
        if image_quality < 0.30:
            flags.append(
                make_flag(
                    text="",
                    reason=(
                        "Image quality appears very low (blur or poor contrast); "
                        "dot detection is unreliable."
                    ),
                    category=CATEGORY_LOW_IMAGE_QUALITY,
                    severity="high",
                )
            )
        elif image_quality < 0.55:
            flags.append(
                make_flag(
                    text="",
                    reason="Image quality appears reduced; some Braille dots may be missed.",
                    category=CATEGORY_LOW_IMAGE_QUALITY,
                    severity="medium",
                )
            )

        flags.extend(grouping.flags)

        if grouping.total_cells == 0:
            logger.info(
                "ocr_empty request_id=%s task_ref=%s dots=%d duration_ms=%d",
                request_id,
                _task_ref(request.taskId),
                len(dots),
                int((perf_counter() - started) * 1000),
            )
            return _failure_response(request_id, flags)

        # --- Stage 5b: grid-evidence re-scoring (Stage 3D-M1) -------------------
        # Re-read each fitted cell's dot pattern directly from the image at
        # the exact slot positions. Fail-closed: when evidence cannot even
        # confirm the blob-detected dots, the page passes through unchanged.
        refinement = refine_grouping(detection, grouping)
        if refinement.applied and refinement.cells_changed:
            grouping = dataclass_replace(
                grouping,
                lines=refinement.lines,
                total_cells=refinement.total_cells,
            )
            flags.extend(refinement.flags)
            if grouping.total_cells == 0:
                return _failure_response(request_id, flags)

        token_lines, raw_cells, reconstruction_flags = reconstruct_lines(grouping)
        flags.extend(reconstruction_flags)

        # --- Stage 6: Unicode Braille -------------------------------------------
        unicode_lines = token_lines_to_unicode(token_lines)
        raw_braille = "\n".join(unicode_lines)

        # --- Stage 7: back-translation -------------------------------------------
        draft_text: str
        translation_completeness: float
        used_liblouis = False
        if settings.liblouis_enabled:
            liblouis_text = liblouis_back_translate(raw_braille, settings.liblouis_table)
            if liblouis_text is not None:
                used_liblouis = True
                draft_text = liblouis_text
                remaining_braille = sum(
                    1
                    for c in liblouis_text
                    if _BRAILLE_BASE <= ord(c) <= _BRAILLE_BASE + 0xFF
                )
                translation_completeness = 1.0 - (
                    remaining_braille / max(len(liblouis_text), 1)
                )
                if is_grade2_table(settings.liblouis_table):
                    flags.append(
                        make_flag(
                            text="",
                            reason=(
                                "Back-translation used Liblouis with a Grade 2 "
                                "(contracted) UEB table. Contractions were "
                                "interpreted but the result is an unverified draft."
                            ),
                            category=CATEGORY_POSSIBLE_CONTRACTION_ISSUE,
                            severity="low",
                        )
                    )
        if not used_liblouis:
            fallback = back_translate_unicode_lines(unicode_lines)
            draft_text = fallback.text
            translation_completeness = fallback.completeness
            flags.extend(fallback.flags)
            if settings.liblouis_enabled:
                flags.append(
                    make_flag(
                        text="",
                        reason=(
                            "Liblouis back-translation was not available; the "
                            "built-in Grade 1 fallback translator was used."
                        ),
                        category=CATEGORY_POSSIBLE_CONTRACTION_ISSUE,
                        severity="low",
                    )
                )

        # --- Stage 8: confidence and flags ----------------------------------------
        if request.subject:
            flags.append(
                make_flag(
                    text="",
                    reason=(
                        "Subject context was provided; subject-specific or technical "
                        "terms may be transcribed incorrectly and must be checked "
                        "by a specialist."
                    ),
                    category=CATEGORY_SUBJECT_SPECIFIC_TERM,
                    severity="low",
                )
            )

        confidence = combined_confidence(
            image_quality=image_quality,
            detection_quality=detection_quality,
            grouping_quality=grouping.quality,
            line_quality=grouping.line_quality,
            translation_completeness=translation_completeness,
            has_cells=True,
            spacing_regularity=detection.spacing_regularity,
        )

        # Honesty caps (see confidence.py): embossed-photo relief detection,
        # noise and near-floor dot sizes (dark path only - emboss discs are
        # painted reconstructions, so their raw/accepted ratio and radius
        # reflect the pairing step, not capture quality; emboss has its own
        # cap), and non-Liblouis translation must never read as
        # near-certainty.
        if detection.mode == MODE_EMBOSS:
            confidence = min(confidence, EMBOSS_MODE_CAP)
        else:
            # A page where many candidate marks were rejected as non-dots is
            # less trustworthy even when it decodes (Stage 3D-G2).
            confidence = round(
                confidence * noise_ratio_factor(len(dots), detection.raw_candidates),
                3,
            )
            size_cap = dot_size_cap(detection.median_radius)
            if size_cap is not None:
                confidence = min(confidence, size_cap)
        if not used_liblouis:
            confidence = min(confidence, FALLBACK_TRANSLATION_CAP)
        # A lattice-recovered page only decoded because normal row separation
        # failed; cap it hard so a clean column fit cannot make a last-ditch
        # recovery read as confident (Stage 3D-K2).
        if grouping.recovered_via_fallback:
            confidence = min(confidence, LATTICE_RECOVERY_CAP)
        # The template reader is a last-ditch rescue on total blob-pipeline
        # failure; its inferred structure must never read as confident (N1).
        if used_template_reader:
            confidence = min(confidence, TEMPLATE_READER_CAP)

        if confidence < 0.55:
            flags.append(
                make_flag(
                    text="",
                    reason=(
                        "Overall OCR confidence is low; treat this draft with "
                        "particular caution."
                    ),
                    category=CATEGORY_LOW_OCR_CONFIDENCE,
                    severity="high" if confidence < 0.30 else "medium",
                )
            )
        elif confidence < 0.85:
            flags.append(
                make_flag(
                    text="",
                    reason="OCR confidence is moderate; the draft likely contains errors.",
                    category=CATEGORY_LOW_OCR_CONFIDENCE,
                    severity="low",
                )
            )

        flags = dedupe_flags(flags)
        duration_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "ocr_completed request_id=%s task_ref=%s mime=%s bytes=%d mode=%s dots=%d "
            "cells=%d lines=%d liblouis=%s table=%s confidence=%.3f flags=%d "
            "duration_ms=%d",
            request_id,
            _task_ref(request.taskId),
            _safe_mime_for_log(request.mimeType),
            byte_count,
            detection.mode,
            len(dots),
            grouping.total_cells,
            len(grouping.lines),
            used_liblouis,
            settings.liblouis_table if used_liblouis else "none",
            confidence,
            len(flags),
            duration_ms,
        )

        page = PageResult(
            pageNumber=1, text=draft_text, confidence=confidence, flags=flags
        )
        return OcrResponse(
            draftText=draft_text,
            confidence=confidence,
            rawBraille=raw_braille,
            rawCells=raw_cells,
            providerRequestId=request_id,
            flags=flags,
            pageResults=[page],
        )
    except Exception as exc:
        logger.error(
            "ocr_processing_error request_id=%s type=%s", request_id, type(exc).__name__
        )
        return _failure_response(
            request_id,
            [
                make_flag(
                    text="",
                    reason=(
                        "OCR processing failed unexpectedly; no draft transcription "
                        "could be produced."
                    ),
                    category=CATEGORY_LOW_OCR_CONFIDENCE,
                    severity="high",
                )
            ],
        )
