"""Capture normalisation for real-world photographs (Stage 3D-L1).

Real phone photos violate three assumptions the rest of the pipeline makes,
and the 2026-07-12 real-capture diagnostic showed each one is a total
(0-cells) failure on genuine worksheet photos:

* **Scale** — dot detection assumes dots roughly 6-14 px across
  (see preprocessing.py). A modern phone photo (4000+ px wide) renders dots
  at ~30-40 px, so the detector floods with texture noise and row separation
  collapses. ``normalise_scale`` downscales oversized captures to the scale
  the pipeline was designed and calibrated for.
* **Orientation** — pages photographed upside-down or sideways never decode;
  nothing in the pipeline rotates. The retry ladder re-attempts detection at
  90/180/270 degrees when the upright attempt produces no cells.
* **Background clutter** — pages photographed on textured surfaces (fabric,
  wood grain) defeat row separation even at the right scale. As a last rung,
  the ladder crops to the bright page region and retries.

Honesty rules (matching the repo's conventions):

* Normalisation NEVER changes an image that already decodes: the ladder only
  runs when the plain attempt yields zero cells, so every previously working
  input is byte-for-byte unaffected.
* Every applied step is reported as a flag; rotated results additionally
  carry a line-order caution because cell bboxes are reported in the rotated
  frame.
* A ladder rescue is still an unverified draft. No step invents dots or
  cells; each rung re-runs the same detection the pipeline always uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.models.responses import Flag
from app.ocr.braille_decode import token_lines_to_unicode
from app.ocr.cell_grouping import GroupingResult
from app.ocr.dot_detection import DetectionOutcome
from app.ocr.flags import (
    CATEGORY_LINE_ORDER_UNCERTAINTY,
    CATEGORY_LOW_IMAGE_QUALITY,
    make_flag,
)
from app.ocr.line_reconstruction import reconstruct_lines
from app.ocr.preprocessing import preprocess
from app.translation.fallback_translator import back_translate_unicode_lines

# Images whose long side exceeds this are downscaled to TARGET_LONG_SIDE.
# 1600 keeps every existing sample/test input (<=1024) untouched while
# catching phone captures (typically 3000-6000 px).
MAX_LONG_SIDE = 1600
TARGET_LONG_SIDE = 1400

# Crop rescue: the page is the large bright region; keep a small margin.
_CROP_BRIGHT_PERCENTILE = 60
_CROP_MIN_THRESHOLD = 140
_CROP_MIN_COVERAGE = 0.05  # bright region must cover >=5% of the frame
_CROP_MARGIN_PX = 8

# Rotations tried (in addition to upright) when the base attempt yields
# nothing. 180 first: upside-down is by far the most common capture mistake.
_ROTATIONS = (180, 90, 270)

# Upside-down disambiguation: a 180-rotated Braille page still forms a
# perfectly regular grid (the dot lattice is symmetric under half-turn), so
# it decodes to cells - just mostly-invalid ones. When the upright decode's
# quick Grade 1 readability falls below this, the flipped frame is also
# decoded and must beat it by the margin to win (upright keeps ties).
_READABILITY_SUSPECT = 0.5
_READABILITY_MARGIN = 0.1

# Sideways-capture tell: a 90/270-rotated page shreds into a handful of
# accidental cells while most detected dots go unused (a cell holds at most
# 6 dots, so cells*6 bounds the dots a decode can account for). When the
# formed cells cannot account for even this fraction of the accepted dots,
# the decode is structurally suspect and the rescue ladder runs with the
# base attempt as the incumbent to beat.
_DOT_UTILISATION_SUSPECT = 0.5

# A real captured page yields dozens of cells. A tiny decode is not proof of
# orientation (sideways pages shred into a handful of accidental cells), so
# below this the rescue ladder always runs - with the base attempt as the
# incumbent, a genuine small decode (a single word photographed close up)
# survives unless a rotation is strictly more readable.
_MIN_PLAUSIBLE_CELLS = 12


@dataclass
class NormalisedDetection:
    """Outcome of detection with capture normalisation applied."""

    detection: DetectionOutcome
    grouping: GroupingResult
    flags: list[Flag] = field(default_factory=list)
    rescaled: bool = False
    rotation_applied: int = 0  # degrees counter-clockwise; 0 = upright
    cropped: bool = False


def normalise_scale(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Downscale an oversized capture to the pipeline's design scale.

    Deterministic and conservative: images at or below MAX_LONG_SIDE are
    returned unchanged, so calibrated/synthetic inputs are unaffected.
    """
    height, width = gray.shape[:2]
    long_side = max(height, width)
    if long_side <= MAX_LONG_SIDE:
        return gray, False
    scale = TARGET_LONG_SIDE / long_side
    resized = cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, True


def crop_to_bright_region(gray: np.ndarray) -> np.ndarray | None:
    """Crop to the bright (paper) region, or None when no meaningful crop.

    The page in a worksheet photo is the dominant bright area; textured
    backgrounds (fabric, wood) are darker. Uses axis projections of a
    brightness mask - deliberately simple and fail-safe: when the mask is
    tiny (no bright page) or the crop would not remove anything, returns
    None and the caller skips the rung.
    """
    threshold = max(_CROP_MIN_THRESHOLD, int(np.percentile(gray, _CROP_BRIGHT_PERCENTILE)))
    mask = gray > threshold
    if mask.sum() < mask.size * _CROP_MIN_COVERAGE:
        return None
    row_hits = np.where(mask.any(axis=1))[0]
    col_hits = np.where(mask.any(axis=0))[0]
    top = max(0, int(row_hits[0]) - _CROP_MARGIN_PX)
    bottom = min(gray.shape[0], int(row_hits[-1]) + 1 + _CROP_MARGIN_PX)
    left = max(0, int(col_hits[0]) - _CROP_MARGIN_PX)
    right = min(gray.shape[1], int(col_hits[-1]) + 1 + _CROP_MARGIN_PX)
    cropped = gray[top:bottom, left:right]
    # No-op crops (page already fills the frame) are not a rescue.
    if cropped.shape == gray.shape or cropped.size == 0:
        return None
    return cropped


def _detect(gray: np.ndarray, select_variant) -> tuple[DetectionOutcome, GroupingResult]:
    pre = preprocess(gray)
    return select_variant(pre.variants)


def _attempt_score(detection: DetectionOutcome, grouping: GroupingResult) -> float:
    """Rank rescue attempts: cells weighted by grid quality and readability.

    A wrong-orientation decode occasionally forms a handful of accidental
    cells; weighting by total_cells keeps a 200-cell true-orientation result
    ahead of a 10-cell accident. Grid geometry alone cannot separate a page
    from its 180-degree flip (the dot lattice is half-turn symmetric), so
    the quick Grade 1 readability of the decoded cells is factored in: the
    true orientation reads, the flip does not.
    """
    if grouping.total_cells == 0:
        return 0.0
    readability = max(_quick_readability(grouping), 0.05)
    return grouping.total_cells * max(grouping.quality, 0.05) * readability


def _rotate(gray: np.ndarray, degrees: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(gray, k=degrees // 90))


def _quick_readability(grouping: GroupingResult) -> float:
    """Fraction of decoded cells the Grade 1 fallback can actually read.

    Cheap content-level sanity check (string work only, no image passes).
    Valid pages score near 1.0 even before back-translation cleanup; a
    wrong-orientation decode is dominated by patterns that map to nothing.
    Never raises: any internal failure returns 0.0, which only means "no
    evidence of readability", never a crash.
    """
    try:
        token_lines, _cells, _flags = reconstruct_lines(grouping)
        unicode_lines = token_lines_to_unicode(token_lines)
        return back_translate_unicode_lines(unicode_lines).completeness
    except Exception:
        return 0.0


def detect_with_normalisation(gray: np.ndarray, select_variant) -> NormalisedDetection:
    """Run detection with scale normalisation and a bounded rescue ladder.

    ``select_variant`` is injected (it lives in pipeline.py) to avoid a
    circular import; the diagnostic probe passes the same function so the
    probe sees exactly what /ocr sees.

    Ladder (only entered when the plain upright attempt forms zero cells):

    1. upright / 90 / 180 / 270 on the scale-normalised frame - best wins;
    2. same four orientations on the bright-region crop - best wins.

    Bounded at 8 extra detection passes worst case, paid only by images
    that currently return nothing at all.
    """
    scaled, rescaled = normalise_scale(gray)
    flags: list[Flag] = []
    if rescaled:
        flags.append(
            make_flag(
                text="",
                reason=(
                    "The photo was much larger than the supported Braille dot "
                    "scale and was automatically downscaled before detection."
                ),
                category=CATEGORY_LOW_IMAGE_QUALITY,
                severity="low",
            )
        )

    detection, grouping = _detect(scaled, select_variant)
    dots_accounted_for = grouping.total_cells * 6 >= _DOT_UTILISATION_SUSPECT * len(
        detection.dots
    )
    plausible_size = grouping.total_cells >= _MIN_PLAUSIBLE_CELLS
    if grouping.total_cells > 0 and dots_accounted_for and plausible_size:
        upright = NormalisedDetection(
            detection=detection, grouping=grouping, flags=flags, rescaled=rescaled
        )
        # Upside-down disambiguation: a flipped page still forms a valid
        # grid, so "cells formed" is not proof of orientation. Only when the
        # upright decode is mostly unreadable is the flipped frame tried,
        # and it must be clearly more readable to win.
        readability = _quick_readability(grouping)
        if readability < _READABILITY_SUSPECT:
            flipped_detection, flipped_grouping = _detect(
                _rotate(scaled, 180), select_variant
            )
            if (
                flipped_grouping.total_cells > 0
                and _quick_readability(flipped_grouping)
                > readability + _READABILITY_MARGIN
            ):
                upright = NormalisedDetection(
                    detection=flipped_detection,
                    grouping=flipped_grouping,
                    flags=list(flags),
                    rescaled=rescaled,
                    rotation_applied=180,
                )
                upright.flags.append(
                    make_flag(
                        text="",
                        reason=(
                            "The page appeared to be photographed upside-down; "
                            "the draft was produced after automatic rotation by "
                            "180 degrees. Cell positions refer to the rotated "
                            "image and line order should be checked."
                        ),
                        category=CATEGORY_LINE_ORDER_UNCERTAINTY,
                        severity="medium",
                    )
                )
        return upright

    # --- Rescue ladder: the plain attempt produced nothing, or produced a
    # structurally suspect decode (cells cannot account for the detected
    # dots). The base attempt stays the incumbent: a rotation or crop must
    # strictly beat it to replace it.
    best = NormalisedDetection(
        detection=detection, grouping=grouping, flags=flags, rescaled=rescaled
    )
    best_score = _attempt_score(detection, grouping)

    frames: list[tuple[np.ndarray, bool]] = [(scaled, False)]
    cropped_frame = crop_to_bright_region(scaled)
    if cropped_frame is not None:
        frames.append((cropped_frame, True))

    for frame, is_cropped in frames:
        for degrees in (0, *_ROTATIONS) if is_cropped else _ROTATIONS:
            candidate_detection, candidate_grouping = _detect(
                _rotate(frame, degrees) if degrees else frame, select_variant
            )
            score = _attempt_score(candidate_detection, candidate_grouping)
            if score > best_score:
                best_score = score
                best = NormalisedDetection(
                    detection=candidate_detection,
                    grouping=candidate_grouping,
                    flags=list(flags),
                    rescaled=rescaled,
                    rotation_applied=degrees,
                    cropped=is_cropped,
                )

    if best.grouping.total_cells > 0:
        if best.rotation_applied:
            best.flags.append(
                make_flag(
                    text="",
                    reason=(
                        "The page appeared rotated; the draft was produced after "
                        f"automatic rotation by {best.rotation_applied} degrees. "
                        "Cell positions refer to the rotated image and line order "
                        "should be checked."
                    ),
                    category=CATEGORY_LINE_ORDER_UNCERTAINTY,
                    severity="medium",
                )
            )
        if best.cropped:
            best.flags.append(
                make_flag(
                    text="",
                    reason=(
                        "Braille was only detected after automatically cropping "
                        "away the image background; check that no Braille near "
                        "the page edges was lost."
                    ),
                    category=CATEGORY_LOW_IMAGE_QUALITY,
                    severity="medium",
                )
            )
    return best
