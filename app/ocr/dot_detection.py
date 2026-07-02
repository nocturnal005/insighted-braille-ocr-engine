"""Braille dot candidate detection via contour analysis.

Version 1 targets clear dark-dot-on-light images (scans, prints, synthetic
renders). Photographs of embossed paper, where dots appear as shadow and
highlight pairs, are not yet handled reliably — see limitations.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np


@dataclass
class Dot:
    x: float
    y: float
    r: float
    confidence: float


def detect_dots(binary: np.ndarray) -> tuple[list[Dot], float]:
    """Return (dot candidates, overall detection quality in [0, 1])."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, float, float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        radius = math.sqrt(area / math.pi)
        candidates.append((cx, cy, radius, circularity))

    if not candidates:
        return [], 0.0

    r_med = median(c[2] for c in candidates)
    dots: list[Dot] = []
    for cx, cy, radius, circularity in candidates:
        # Discard blobs far from the typical dot size (noise, smudges, text).
        if radius > 2.5 * r_med or radius < 0.4 * r_med:
            continue
        size_consistency = math.exp(-abs(radius - r_med) / max(r_med, 1e-6))
        confidence = min(
            1.0, 0.6 * min(circularity / 0.8, 1.0) + 0.4 * size_consistency
        )
        dots.append(Dot(x=cx, y=cy, r=radius, confidence=confidence))

    if not dots:
        return [], 0.0

    detection_quality = float(sum(d.confidence for d in dots) / len(dots))
    return dots, detection_quality
