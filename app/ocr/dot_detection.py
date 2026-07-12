"""Braille dot candidate detection via contour analysis.

Stage 3D-D: detection now scores every preprocessing variant (dark printed
dots AND embossed relief blobs) and keeps the best one, so photographs of
embossed paper — where a dot is a merged highlight/shadow blob rather than
an inked disc — produce usable candidates instead of nothing. Each candidate
preserves its centre, radius, bounding box, area, and a per-dot confidence;
the selection step also reports spacing regularity and noise diagnostics so
the pipeline can flag uncertain images honestly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

import cv2
import numpy as np

from app.models.responses import Flag
from app.ocr.flags import (
    CATEGORY_LOW_IMAGE_QUALITY,
    CATEGORY_LOW_OCR_CONFIDENCE,
    CATEGORY_UNCLEAR_BRAILLE_CELL,
    make_flag,
)
from app.ocr.preprocessing import MODE_DARK, MODE_EMBOSS, BinaryVariant

# Embossed blobs are a closed highlight+shadow pair, so they are less
# circular than an inked disc; accept slightly rougher shapes there.
_MIN_CIRCULARITY = {MODE_DARK: 0.5, MODE_EMBOSS: 0.4}


@dataclass
class Dot:
    x: float
    y: float
    r: float
    confidence: float
    area: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2


@dataclass
class DetectionOutcome:
    """Best-variant detection result plus diagnostics for flags/confidence."""

    dots: list[Dot] = field(default_factory=list)
    quality: float = 0.0  # mean per-dot confidence in [0, 1]
    mode: str = MODE_DARK  # which preprocessing variant won
    image_quality: float = 0.0  # quality score of the winning variant
    spacing_regularity: float = 0.0  # nearest-neighbour regularity in [0, 1]
    raw_candidates: int = 0  # contour candidates before size filtering
    median_radius: float = 0.0  # median accepted dot radius in px (0 = none)
    noise_filtered: bool = False  # True when low-confidence dots were dropped
    flags: list[Flag] = field(default_factory=list)
    # Grayscale pixel-aligned with the dot coordinates (the variant's own
    # deskewed frame). In-memory only, for grid-evidence scoring; never
    # serialised. None in legacy construction paths.
    aligned_gray: np.ndarray | None = None


# Per-dot confidence below which a candidate is dropped in the strict retry
# pass (Stage 3D-G2). On noisy pages the true dots score ~1.0 while specks
# that slipped the size gate score lower on circularity/size consistency, so
# this threshold separates them cleanly; on clean pages nothing is removed.
STRICT_DOT_CONFIDENCE = 0.85


def detect_dots(
    binary: np.ndarray, min_circularity: float = 0.5
) -> tuple[list[Dot], float]:
    """Return (dot candidates, overall detection quality in [0, 1])."""
    dots, quality, _ = _detect(binary, min_circularity)
    return dots, quality


def _detect(
    binary: np.ndarray, min_circularity: float
) -> tuple[list[Dot], float, int]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, float, float, float, float, tuple[int, int, int, int]]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        radius = math.sqrt(area / math.pi)
        bx, by, bw, bh = cv2.boundingRect(contour)
        candidates.append(
            (cx, cy, radius, circularity, area, (bx, by, bx + bw, by + bh))
        )

    if not candidates:
        return [], 0.0, 0

    r_med = median(c[2] for c in candidates)
    dots: list[Dot] = []
    for cx, cy, radius, circularity, area, bbox in candidates:
        # Discard blobs far from the typical dot size (noise, smudges, text).
        if radius > 2.5 * r_med or radius < 0.4 * r_med:
            continue
        size_consistency = math.exp(-abs(radius - r_med) / max(r_med, 1e-6))
        confidence = min(
            1.0, 0.6 * min(circularity / 0.8, 1.0) + 0.4 * size_consistency
        )
        dots.append(
            Dot(x=cx, y=cy, r=radius, confidence=confidence, area=area, bbox=bbox)
        )

    if not dots:
        return [], 0.0, len(candidates)

    detection_quality = float(sum(d.confidence for d in dots) / len(dots))
    return dots, detection_quality, len(candidates)


def spacing_regularity(dots: list[Dot]) -> float:
    """Regularity of nearest-neighbour dot spacing in [0, 1].

    Real Braille dots sit on a fixed pitch, so nearest-neighbour distances
    cluster tightly. Random noise blobs and texture do not. A high
    coefficient of variation therefore signals unreliable structure, which
    feeds both variant selection and the final confidence.
    """
    if len(dots) < 4:
        return 0.5  # too few points to judge either way
    points = np.array([[d.x, d.y] for d in dots], dtype=np.float64)
    diffs = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((diffs**2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    mean = float(nearest.mean())
    if mean <= 1e-6:
        return 0.0
    cv_value = float(nearest.std() / mean)
    return float(np.clip(1.0 - cv_value, 0.0, 1.0))


def _median_radius(dots: list[Dot]) -> float:
    return float(median(d.r for d in dots)) if dots else 0.0


def detect_variant(variant: BinaryVariant) -> DetectionOutcome:
    """Run dot detection on one preprocessing variant."""
    dots, quality, raw_candidates = _detect(
        variant.binary, _MIN_CIRCULARITY.get(variant.mode, 0.5)
    )
    return DetectionOutcome(
        dots=dots,
        quality=quality,
        mode=variant.mode,
        image_quality=variant.quality,
        spacing_regularity=spacing_regularity(dots),
        raw_candidates=raw_candidates,
        median_radius=_median_radius(dots),
        aligned_gray=variant.aligned_gray,
    )


def strict_variant(outcome: DetectionOutcome) -> DetectionOutcome | None:
    """A stricter candidate with low-confidence dots dropped, or None.

    Only offered when the size filter already rejected extra marks
    (raw_candidates > accepted dots - independent evidence of a noisy page)
    and the strict subset actually removes something while keeping enough
    dots to form cells. Variant selection scores it like any other
    candidate, so it only wins when the filtered dots form a clearly better
    Braille grid; clean pages are never affected because nothing is removed.
    """
    if outcome.raw_candidates <= len(outcome.dots):
        return None
    strict = [d for d in outcome.dots if d.confidence >= STRICT_DOT_CONFIDENCE]
    if len(strict) < 6 or len(strict) == len(outcome.dots):
        return None
    return DetectionOutcome(
        dots=strict,
        quality=float(sum(d.confidence for d in strict) / len(strict)),
        mode=outcome.mode,
        image_quality=outcome.image_quality,
        spacing_regularity=spacing_regularity(strict),
        raw_candidates=outcome.raw_candidates,
        median_radius=_median_radius(strict),
        noise_filtered=True,
        aligned_gray=outcome.aligned_gray,
    )


def selection_flags(outcome: DetectionOutcome) -> list[Flag]:
    """Diagnostics for the chosen variant, reported as uncertainty flags.

    Few dots, heavy noise, irregular spacing, and embossed-photo mode are
    all honest reasons to trust the draft less; they are flagged rather
    than silently absorbed.
    """
    flags: list[Flag] = []

    if 0 < len(outcome.dots) < 6:
        flags.append(
            make_flag(
                text="",
                reason=(
                    "Very few Braille dot candidates were detected; the draft "
                    "is likely incomplete."
                ),
                category=CATEGORY_LOW_OCR_CONFIDENCE,
                severity="medium",
            )
        )
    if outcome.dots and outcome.raw_candidates > 3 * len(outcome.dots):
        flags.append(
            make_flag(
                text="",
                reason=(
                    "Many non-dot marks were detected (noise, texture, or "
                    "handwriting); some Braille dots may be misread."
                ),
                category=CATEGORY_LOW_IMAGE_QUALITY,
                severity="medium",
            )
        )
    elif outcome.dots and outcome.raw_candidates > 1.15 * len(outcome.dots):
        flags.append(
            make_flag(
                text="",
                reason=(
                    "Background noise or stray marks were detected alongside "
                    "the Braille dots; check the draft against the original."
                ),
                category=CATEGORY_LOW_IMAGE_QUALITY,
                severity="low",
            )
        )
    if outcome.noise_filtered:
        flags.append(
            make_flag(
                text="",
                reason=(
                    "Some detected marks were rejected as probable noise "
                    "before decoding; a real dot may have been discarded "
                    "with them, so verify the draft carefully."
                ),
                category=CATEGORY_UNCLEAR_BRAILLE_CELL,
                severity="medium",
            )
        )
    if (
        outcome.dots
        and outcome.mode == MODE_DARK
        and 0 < outcome.median_radius < 3.2
    ):
        flags.append(
            make_flag(
                text="",
                reason=(
                    "Braille dots are near or below the reliable size floor "
                    "(about 6 pixels across); misread cells are likely. "
                    "Recapture at higher resolution if possible."
                ),
                category=CATEGORY_LOW_IMAGE_QUALITY,
                severity="high" if outcome.median_radius < 2.4 else "medium",
            )
        )
    if len(outcome.dots) >= 4 and outcome.spacing_regularity < 0.55:
        flags.append(
            make_flag(
                text="",
                reason=(
                    "Detected dot spacing is inconsistent; cell boundaries "
                    "are uncertain."
                ),
                category=CATEGORY_UNCLEAR_BRAILLE_CELL,
                severity="medium",
            )
        )
    if outcome.dots and outcome.mode == MODE_EMBOSS:
        flags.append(
            make_flag(
                text="",
                reason=(
                    "The image looks like an embossed-paper photograph "
                    "(raised dots detected from light/shadow relief). This is "
                    "less reliable than a clean scan; specialist checking is "
                    "especially important."
                ),
                category=CATEGORY_LOW_IMAGE_QUALITY,
                severity="low",
            )
        )

    return flags
