"""Image preprocessing: denoise, contrast enhancement, thresholding, deskew.

Produces candidate binary images with Braille dots as white foreground, plus
a heuristic image-quality score in [0, 1] per candidate.

Two binarisation strategies are produced (Stage 3D-D):

* ``dark``   — the original path for dark printed/synthetic dots on a light
               background (adaptive threshold on CLAHE-enhanced grayscale).
* ``emboss`` — for photographs/scans of embossed paper, where a raised dot
               has no ink at all: it appears only as a highlight/shadow
               crescent pair under directional light. This path flattens
               uneven illumination, finds highlight and shadow blobs
               separately, pairs them along the self-calibrated light
               direction, and reconstructs one clean circular candidate at
               each pair's midpoint — the true dot centre.

Variant selection (in the pipeline) scores both candidates and picks the
better one, so clean scans keep their original behaviour and embossed
photographs get usable, well-centred dots instead of nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MODE_DARK = "dark"
MODE_EMBOSS = "emboss"

# Pixel-scale assumptions: dots roughly 6-14 px across (typical phone photos
# and our samples). Other scales degrade to a low score rather than garbage.
_ILLUMINATION_SIGMA = 25.0  # much larger than a dot: models lighting only
_SPECKLE_OPEN_KERNEL = 3  # drops single-pixel paper-noise speckles


@dataclass
class BinaryVariant:
    mode: str  # MODE_DARK or MODE_EMBOSS
    binary: np.ndarray  # uint8, dots white (255) on black (0)
    quality: float  # heuristic image quality in [0, 1]
    deskewed: bool


@dataclass
class PreprocessResult:
    binary: np.ndarray  # primary (dark-path) binary, kept for compatibility
    gray: np.ndarray  # contrast-enhanced grayscale
    quality: float  # dark-path quality in [0, 1]
    deskewed: bool
    variants: list[BinaryVariant] = field(default_factory=list)


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


def _emboss_quality_score(gray: np.ndarray, enhanced: np.ndarray, binary: np.ndarray) -> float:
    """Quality for the emboss variant.

    The dark-path contrast measure (foreground mean vs background mean) is
    meaningless for embossed dots: a highlight/shadow pair averages back to
    the paper level. Contrast here is the mean *absolute* deviation of
    foreground pixels from the background level, which is high whenever the
    crescents are pronounced, regardless of polarity.
    """
    foreground = binary > 0
    if not foreground.any():
        return 0.0

    mask = cv2.dilate(binary, np.ones((5, 5), np.uint8)) > 0
    laplacian = np.abs(cv2.Laplacian(enhanced, cv2.CV_64F))
    sharpness = float(np.clip(laplacian[mask].mean() / 40.0, 0.0, 1.0))

    bg_mean = float(gray[~foreground].mean())
    deviation = float(np.abs(gray[foreground].astype(np.float32) - bg_mean).mean())
    contrast = float(np.clip(deviation / 60.0, 0.0, 1.0))

    return 0.5 * sharpness + 0.5 * contrast


def _flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Remove large-scale lighting variation, keeping dot-scale detail.

    Embossed photographs are usually lit from one side, so the paper itself
    drifts from bright to dark across the page. Subtracting a heavily blurred
    copy (the lighting field) re-centres every region on the same mid-grey,
    which lets a single global threshold work for the whole page.
    """
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=_ILLUMINATION_SIGMA)
    flattened = gray.astype(np.int16) - background.astype(np.int16) + 128
    return np.clip(flattened, 0, 255).astype(np.uint8)


def _side_blobs(side: np.ndarray) -> list[tuple[float, float, float]]:
    """Blob centroids on one relief side: list of (x, y, radius).

    Each side (highlight or shadow) of an embossed dot is a small crescent.
    A tiny open removes single-pixel paper noise; anything dot-crescent
    sized survives.
    """
    if float(side.max()) < 12:  # near-flat side: nothing but noise
        return []
    _, binary = cv2.threshold(side, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_SPECKLE_OPEN_KERNEL, _SPECKLE_OPEN_KERNEL)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    if float((binary > 0).mean()) > 0.20:  # texture, not dots
        return []

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    blobs: list[tuple[float, float, float]] = []
    for i in range(1, count):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        radius = float(np.sqrt(area / np.pi))
        blobs.append((float(centroids[i][0]), float(centroids[i][1]), radius))
    return blobs


def _emboss_binary(gray: np.ndarray) -> np.ndarray:
    """Reconstruct embossed dots by pairing highlight and shadow blobs.

    A raised dot under directional light shows a bright crescent on the lit
    side and a dark crescent on the shadow side; the dot centre sits at the
    midpoint of the pair. Thresholding the relief directly yields half-moon
    fragments that break circularity checks, so instead:

    1. flatten illumination and split the signed deviation from mid-grey
       into a highlight side and a shadow side;
    2. find small blobs on each side;
    3. estimate the dominant light direction as the median highlight->shadow
       offset (self-calibrating: works for any light angle);
    4. keep highlight/shadow pairs consistent with that direction and close
       enough to belong to one dot;
    5. paint a filled disc at each pair midpoint.

    The painted binary contains one clean circular candidate per physical
    dot, at the true dot centre, which the ordinary contour detector then
    handles exactly like a printed dot. Pages with too few pairs return an
    empty image so variant selection falls back to the dark path.
    """
    flat = _flatten_illumination(gray)
    flat = cv2.medianBlur(flat, 3)

    signed = flat.astype(np.int16) - 128
    bright = np.clip(signed, 0, 255).astype(np.uint8)
    dark = np.clip(-signed, 0, 255).astype(np.uint8)

    bright_blobs = _side_blobs(bright)
    dark_blobs = _side_blobs(dark)
    if len(bright_blobs) < 3 or len(dark_blobs) < 3:
        return np.zeros_like(gray)

    typical_r = float(
        np.median([b[2] for b in bright_blobs] + [d[2] for d in dark_blobs])
    )
    max_pair_distance = max(4.0, 3.5 * typical_r)

    dark_points = np.array([[d[0], d[1]] for d in dark_blobs])

    # First pass: nearest shadow for every highlight -> dominant light vector.
    offsets: list[tuple[float, float]] = []
    for bx, by, _ in bright_blobs:
        distances = np.sqrt(((dark_points - [bx, by]) ** 2).sum(axis=1))
        j = int(distances.argmin())
        if distances[j] <= max_pair_distance:
            offsets.append((dark_points[j][0] - bx, dark_points[j][1] - by))
    if len(offsets) < 3:
        return np.zeros_like(gray)
    median_dx = float(np.median([o[0] for o in offsets]))
    median_dy = float(np.median([o[1] for o in offsets]))
    direction_norm = float(np.hypot(median_dx, median_dy))
    if direction_norm < 0.5:
        return np.zeros_like(gray)

    # Second pass: one-to-one greedy matching by ascending distance, keeping
    # only pairs whose offset agrees with the dominant light direction. The
    # uniqueness constraint matters on tightly spaced pages: a highlight can
    # otherwise also claim the shadow of the *next* dot along the light axis,
    # painting a phantom dot between the two real ones. True pairs are
    # mutually nearest, so they win the greedy pass; phantoms find their
    # shadow already taken.
    candidate_pairs: list[tuple[float, int, int]] = []
    for i, (bx, by, _) in enumerate(bright_blobs):
        deltas = dark_points - [bx, by]
        distances = np.sqrt((deltas**2).sum(axis=1))
        for j in np.where(distances <= max_pair_distance)[0]:
            distance = float(distances[j])
            if distance < 1e-6:
                continue
            cosine = (
                float(deltas[j][0]) * median_dx + float(deltas[j][1]) * median_dy
            ) / (distance * direction_norm)
            if cosine >= 0.5:
                candidate_pairs.append((distance, i, int(j)))

    candidate_pairs.sort()
    canvas = np.zeros_like(gray)
    dot_radius = max(2, int(round(1.6 * typical_r)))
    used_bright: set[int] = set()
    used_dark: set[int] = set()
    paired = 0
    for _, i, j in candidate_pairs:
        if i in used_bright or j in used_dark:
            continue
        used_bright.add(i)
        used_dark.add(j)
        bx, by, _ = bright_blobs[i]
        cx = int(round((bx + float(dark_points[j][0])) / 2))
        cy = int(round((by + float(dark_points[j][1])) / 2))
        cv2.circle(canvas, (cx, cy), dot_radius, 255, thickness=-1)
        paired += 1

    if paired < 3:
        return np.zeros_like(gray)
    return canvas


def preprocess(gray: np.ndarray) -> PreprocessResult:
    # --- Dark-dot path (original behaviour, unchanged) ----------------------
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    binary = _threshold(enhanced)

    # Foreground (dots) should be the minority; invert if the polarity flipped.
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)

    enhanced_unrotated = enhanced.copy()
    binary, enhanced, deskewed = _deskew(binary, enhanced)
    dark_quality = _quality_score(gray, enhanced, binary)

    variants = [
        BinaryVariant(
            mode=MODE_DARK, binary=binary, quality=dark_quality, deskewed=deskewed
        )
    ]

    # --- Embossed-relief path (Stage 3D-D) ----------------------------------
    emboss = _emboss_binary(gray)
    if (emboss > 0).any():
        # Quality is measured before deskew so binary, enhanced, and gray all
        # stay pixel-aligned; deskew only helps the geometric stages after.
        emboss_quality = _emboss_quality_score(gray, enhanced_unrotated, emboss)
        emboss, _, emboss_deskewed = _deskew(emboss, enhanced_unrotated)
        variants.append(
            BinaryVariant(
                mode=MODE_EMBOSS,
                binary=emboss,
                quality=emboss_quality,
                deskewed=emboss_deskewed,
            )
        )

    return PreprocessResult(
        binary=binary,
        gray=enhanced,
        quality=dark_quality,
        deskewed=deskewed,
        variants=variants,
    )
