"""Combined confidence scoring.

The score is a heuristic in [0, 1], NOT a calibrated probability. It blends
image quality, dot detection certainty, dot spacing regularity, cell-grid
fit, line-order certainty, and translation completeness. A score of 0 always
means no usable draft.

Stage 3D-D honesty rules (applied by the pipeline on top of this blend):
embossed-photograph runs are capped because relief detection is inherently
less reliable than a clean scan, and fallback Grade 1 translation (no
Liblouis) is capped because the built-in translator is not table-driven.
"""

from __future__ import annotations

_WEIGHTS = {
    "image_quality": 0.15,
    "detection_quality": 0.25,
    "spacing_regularity": 0.10,
    "grouping_quality": 0.20,
    "line_quality": 0.10,
    "translation_completeness": 0.20,
}

# Caps keep uncertain conditions from reporting near-certainty. They are
# deliberately conservative: output is a draft for specialist review either way.
EMBOSS_MODE_CAP = 0.82
FALLBACK_TRANSLATION_CAP = 0.95
# Stage 3D-K2: a page whose rows were recovered by the lattice fallback only
# got a draft because normal row separation failed outright. That is inherently
# low-trust — the recovered row/line structure can still be wrong (pitch
# aliasing, tight interline spacing, or a regular non-Braille texture that
# slipped the spacing-regularity gate). Cap hard, below the emboss cap, so such
# a draft can never read as confident regardless of how cleanly its columns fit.
LATTICE_RECOVERY_CAP = 0.55

# Dot-size honesty (Stage 3D-G2): decoding degrades sharply as dots approach
# the ~6 px readable floor, but none of the blend factors senses absolute
# scale (small dots are still round and evenly spaced), so near-floor pages
# used to keep clean-scan confidence despite elevated error. The cap applies
# to measured dark-path dot radii only - emboss-mode discs are reconstructions
# painted at a synthetic radius, which says nothing about capture resolution.
DOT_FLOOR_RADIUS_PX = 3.2  # ~6.4 px dot diameter: caps start here
DOT_FLOOR_MIN_RADIUS_PX = 2.0  # at/below this the cap bottoms out
DOT_FLOOR_CAP_RANGE = (0.50, 0.80)  # cap at min radius .. cap at floor radius

# Noise honesty (Stage 3D-G2): when the size filter rejected many candidate
# marks, the page is noisy and surviving dots are less trustworthy even if
# they decode; scale confidence by the accepted/raw ratio's headroom.
NOISE_RATIO_CLEAN = 0.90  # accepted/raw at or above this: no penalty


def dot_size_cap(median_radius: float) -> float | None:
    """Confidence cap for near-floor dot sizes; None when dots are large enough."""
    if median_radius <= 0 or median_radius >= DOT_FLOOR_RADIUS_PX:
        return None
    low_cap, high_cap = DOT_FLOOR_CAP_RANGE
    span = DOT_FLOOR_RADIUS_PX - DOT_FLOOR_MIN_RADIUS_PX
    position = (median_radius - DOT_FLOOR_MIN_RADIUS_PX) / span
    return round(low_cap + (high_cap - low_cap) * min(max(position, 0.0), 1.0), 3)


def noise_ratio_factor(accepted: int, raw_candidates: int) -> float:
    """Multiplicative confidence factor in (0, 1] for noisy pages."""
    if accepted <= 0 or raw_candidates <= accepted:
        return 1.0
    ratio = accepted / raw_candidates
    if ratio >= NOISE_RATIO_CLEAN:
        return 1.0
    return round(max(0.6, 0.5 + 0.5 * ratio), 3)


def combined_confidence(
    *,
    image_quality: float,
    detection_quality: float,
    grouping_quality: float,
    line_quality: float,
    translation_completeness: float,
    has_cells: bool,
    spacing_regularity: float = 1.0,
) -> float:
    if not has_cells:
        return 0.0
    score = (
        _WEIGHTS["image_quality"] * image_quality
        + _WEIGHTS["detection_quality"] * detection_quality
        + _WEIGHTS["spacing_regularity"] * spacing_regularity
        + _WEIGHTS["grouping_quality"] * grouping_quality
        + _WEIGHTS["line_quality"] * line_quality
        + _WEIGHTS["translation_completeness"] * translation_completeness
    )
    return round(min(max(score, 0.0), 1.0), 3)
