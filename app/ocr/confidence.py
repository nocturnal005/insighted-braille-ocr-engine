"""Combined confidence scoring.

The score is a heuristic in [0, 1], NOT a calibrated probability. It blends
image quality, dot detection certainty, cell-grid fit, line-order certainty,
and translation completeness. A score of 0 always means no usable draft.
"""

from __future__ import annotations

_WEIGHTS = {
    "image_quality": 0.20,
    "detection_quality": 0.25,
    "grouping_quality": 0.25,
    "line_quality": 0.10,
    "translation_completeness": 0.20,
}


def combined_confidence(
    *,
    image_quality: float,
    detection_quality: float,
    grouping_quality: float,
    line_quality: float,
    translation_completeness: float,
    has_cells: bool,
) -> float:
    if not has_cells:
        return 0.0
    score = (
        _WEIGHTS["image_quality"] * image_quality
        + _WEIGHTS["detection_quality"] * detection_quality
        + _WEIGHTS["grouping_quality"] * grouping_quality
        + _WEIGHTS["line_quality"] * line_quality
        + _WEIGHTS["translation_completeness"] * translation_completeness
    )
    return round(min(max(score, 0.0), 1.0), 3)
