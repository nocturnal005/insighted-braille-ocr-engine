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
from time import perf_counter
from uuid import uuid4

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.requests import OcrRequest
from app.models.responses import Flag, OcrResponse, PageResult
from app.ocr.braille_decode import token_lines_to_unicode
from app.ocr.cell_grouping import group_dots
from app.ocr.confidence import combined_confidence
from app.ocr.dot_detection import detect_dots
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
from app.ocr.preprocessing import preprocess
from app.translation.fallback_translator import back_translate_unicode_lines
from app.translation.liblouis_adapter import liblouis_back_translate

logger = get_logger(__name__)

_BRAILLE_BASE = 0x2800


def _new_request_id() -> str:
    return "ocr_" + uuid4().hex


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

        # --- Stage 3: preprocessing --------------------------------------------
        pre = preprocess(gray)
        if pre.quality < 0.30:
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
        elif pre.quality < 0.55:
            flags.append(
                make_flag(
                    text="",
                    reason="Image quality appears reduced; some Braille dots may be missed.",
                    category=CATEGORY_LOW_IMAGE_QUALITY,
                    severity="medium",
                )
            )

        # --- Stage 4: dot detection --------------------------------------------
        dots, detection_quality = detect_dots(pre.binary)

        # --- Stage 5: cell grouping --------------------------------------------
        grouping = group_dots(dots)
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
            image_quality=pre.quality,
            detection_quality=detection_quality,
            grouping_quality=grouping.quality,
            line_quality=grouping.line_quality,
            translation_completeness=translation_completeness,
            has_cells=True,
        )

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
            "ocr_completed request_id=%s task_ref=%s mime=%s bytes=%d dots=%d "
            "cells=%d lines=%d liblouis=%s confidence=%.3f flags=%d duration_ms=%d",
            request_id,
            _task_ref(request.taskId),
            _safe_mime_for_log(request.mimeType),
            byte_count,
            len(dots),
            grouping.total_cells,
            len(grouping.lines),
            used_liblouis,
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
