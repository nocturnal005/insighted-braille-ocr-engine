"""Capture-quality preflight assessment for real Braille photographs (3D-J1).

Estimates, from the image file plus (optionally) a diagnostic probe result,
whether a capture is worth formal intake or should be retaken. All outputs
are heuristics for triage - they are not accuracy predictions and never
certify anything.

Classifications, in decreasing usability:

    readable_candidate    no blocking issues found - worth formal intake
    borderline_candidate  soft issues; may work, expect degraded results
    retake_recommended    hard issues; a better capture will help more
                          than any pipeline tuning
    unusable              cannot be assessed at all (unreadable/too small)

Reasons are fixed safe strings - never file names or image content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.evaluation.diagnostic_probe import (
    FAILURE_DECODE_REJECTED,
    FAILURE_ROW_SEPARATION,
    FAILURE_UNSUPPORTED_FILE,
    ProbeResult,
)

CLASS_READABLE = "readable_candidate"
CLASS_BORDERLINE = "borderline_candidate"
CLASS_RETAKE = "retake_recommended"
CLASS_UNUSABLE = "unusable"

# Heuristic thresholds. Dimensions/contrast follow the Stage 3D-E dataset
# audit (min dimension 80, contrast std 8.0); the dot-size floor follows the
# confidence module (radius 3.2 px ~= 6.4 px dot diameter); sharpness uses
# the variance of a 3x3 Laplacian on the grayscale image, where clean
# renders and crisp photos score in the hundreds-to-thousands and defocused
# captures fall to low double digits.
MIN_DIMENSION_PX = 80
SMALL_DIMENSION_PX = 200
LOW_CONTRAST_STD = 8.0
BORDERLINE_CONTRAST_STD = 20.0
BLUR_SHARPNESS_FLOOR = 30.0
BORDERLINE_SHARPNESS = 120.0
DOT_RADIUS_FLOOR_PX = 3.2
MIN_DOTS_FOR_PAGE = 6
JPEG_LOW_BYTES_PER_PIXEL = 0.08

REASON_TOO_SMALL = "image too small"
REASON_SMALL = "image dimensions are small; dots may fall below the readable floor"
REASON_LOW_CONTRAST = "low contrast"
REASON_REDUCED_CONTRAST = "reduced contrast"
REASON_LIKELY_BLUR = "likely blur"
REASON_SOFT_FOCUS = "soft focus / mild blur"
REASON_DOT_SIZE = "dot size below reliable threshold"
REASON_TOO_SPARSE = "too few dot candidates - Braille area too small or too sparse"
REASON_ROWS_UNRELIABLE = "row separation unreliable"
REASON_COMPRESSION = "possible compression artefacts (heavily compressed JPEG)"
REASON_UNREADABLE = "image could not be read"
REASON_REJECTED = "image was rejected by safe intake validation"


@dataclass
class CaptureQuality:
    classification: str
    reasons: list[str] = field(default_factory=list)
    retake_recommended: bool = False

    # Numeric proxies (heuristics, for the local report only)
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    sharpness: float = 0.0
    contrast_std: float = 0.0

    def to_safe_dict(self) -> dict:
        return {
            "classification": self.classification,
            "reasons": list(self.reasons),
            "retake_recommended": self.retake_recommended,
            "width": self.width,
            "height": self.height,
            "file_size_bytes": self.file_size_bytes,
            "sharpness": round(self.sharpness, 1),
            "contrast_std": round(self.contrast_std, 1),
        }


def _sharpness_proxy(gray: np.ndarray) -> float:
    """Variance of a 3x3 Laplacian - a standard defocus/blur proxy."""
    arr = gray.astype(np.float32)
    lap = (
        -4.0 * arr[1:-1, 1:-1]
        + arr[:-2, 1:-1]
        + arr[2:, 1:-1]
        + arr[1:-1, :-2]
        + arr[1:-1, 2:]
    )
    return float(lap.var())


def assess_capture_quality(
    path: Path, probe: ProbeResult | None = None
) -> CaptureQuality:
    """Classify one capture. Never raises; unreadable files are 'unusable'."""
    quality = CaptureQuality(classification=CLASS_UNUSABLE)
    try:
        quality.file_size_bytes = path.stat().st_size
        with Image.open(path) as img:
            gray = np.asarray(img.convert("L"))
    except Exception:
        quality.reasons.append(REASON_UNREADABLE)
        quality.retake_recommended = True
        return quality

    quality.height, quality.width = gray.shape[:2]
    hard: list[str] = []
    soft: list[str] = []

    if min(quality.width, quality.height) < MIN_DIMENSION_PX:
        quality.reasons.append(REASON_TOO_SMALL)
        quality.retake_recommended = True
        return quality  # unusable: nothing else can be judged reliably
    if min(quality.width, quality.height) < SMALL_DIMENSION_PX and (
        # A tight crop is fine when the dots are demonstrably large enough;
        # caution only when dot size is unknown or unmeasured.
        probe is None
        or probe.median_dot_radius_px < DOT_RADIUS_FLOOR_PX
    ):
        soft.append(REASON_SMALL)

    quality.contrast_std = float(gray.std())
    if quality.contrast_std < LOW_CONTRAST_STD:
        hard.append(REASON_LOW_CONTRAST)
    elif quality.contrast_std < BORDERLINE_CONTRAST_STD:
        soft.append(REASON_REDUCED_CONTRAST)

    if gray.shape[0] > 2 and gray.shape[1] > 2:
        quality.sharpness = _sharpness_proxy(gray)
        if quality.sharpness < BLUR_SHARPNESS_FLOOR:
            hard.append(REASON_LIKELY_BLUR)
        elif quality.sharpness < BORDERLINE_SHARPNESS:
            soft.append(REASON_SOFT_FOCUS)

    if path.suffix.lower() in (".jpg", ".jpeg"):
        bytes_per_pixel = quality.file_size_bytes / max(
            quality.width * quality.height, 1
        )
        if bytes_per_pixel < JPEG_LOW_BYTES_PER_PIXEL:
            soft.append(REASON_COMPRESSION)

    if probe is not None:
        if probe.failure_point in (
            FAILURE_DECODE_REJECTED,
            FAILURE_UNSUPPORTED_FILE,
        ):
            quality.reasons.append(REASON_REJECTED)
            quality.retake_recommended = True
            return quality
        if 0.0 < probe.median_dot_radius_px < DOT_RADIUS_FLOOR_PX:
            hard.append(REASON_DOT_SIZE)
        if probe.decode_ok and probe.accepted_dots < MIN_DOTS_FOR_PAGE:
            hard.append(REASON_TOO_SPARSE)
        if probe.failure_point == FAILURE_ROW_SEPARATION:
            hard.append(REASON_ROWS_UNRELIABLE)

    quality.reasons.extend(hard + soft)
    if hard:
        quality.classification = CLASS_RETAKE
        quality.retake_recommended = True
    elif soft:
        quality.classification = CLASS_BORDERLINE
    else:
        quality.classification = CLASS_READABLE
    return quality
