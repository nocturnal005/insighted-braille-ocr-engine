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
