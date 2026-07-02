"""Image preprocessing: denoise, contrast enhancement, thresholding, deskew.

Produces a binary image with candidate Braille dots as white foreground,
plus a heuristic image-quality score in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    binary: np.ndarray  # uint8, dots white (255) on black (0)
    gray: np.ndarray  # contrast-enhanced grayscale
    quality: float  # heuristic image quality in [0, 1]
    deskewed: bool


def _threshold(enhanced: np.ndarray) -> np.ndarray:
    min_dim = min(enhanced.shape)
    if min_dim < 12:
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary
    block_size = 35
    if block_size >= min_dim:
        block_size = max(3, (min_dim // 2) * 2 + 1)
    return cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        10,
    )


def _deskew(binary: np.ndarray, enhanced: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 30:
        return binary, enhanced, False
    angle = cv2.minAreaRect(points)[2]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    if not (0.7 <= abs(angle) <= 15):
        return binary, enhanced, False
    h, w = binary.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    binary = cv2.warpAffine(
        binary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    enhanced = cv2.warpAffine(
        enhanced, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255
    )
    return binary, enhanced, True


def _quality_score(gray: np.ndarray, enhanced: np.ndarray, binary: np.ndarray) -> float:
    foreground = binary > 0
    if not foreground.any():
        return 0.0

    # Sharpness: mean absolute Laplacian around the detected foreground.
    mask = cv2.dilate(binary, np.ones((5, 5), np.uint8)) > 0
    laplacian = np.abs(cv2.Laplacian(enhanced, cv2.CV_64F))
    sharpness = float(np.clip(laplacian[mask].mean() / 40.0, 0.0, 1.0))

    # Contrast: separation between foreground and background intensity.
    fg_mean = float(gray[foreground].mean())
    bg_mean = float(gray[~foreground].mean())
    contrast = float(np.clip(abs(bg_mean - fg_mean) / 200.0, 0.0, 1.0))

    return 0.5 * sharpness + 0.5 * contrast


def preprocess(gray: np.ndarray) -> PreprocessResult:
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    binary = _threshold(enhanced)

    # Foreground (dots) should be the minority; invert if the polarity flipped.
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)

    binary, enhanced, deskewed = _deskew(binary, enhanced)

    quality = _quality_score(gray, enhanced, binary)
    return PreprocessResult(binary=binary, gray=enhanced, quality=quality, deskewed=deskewed)
