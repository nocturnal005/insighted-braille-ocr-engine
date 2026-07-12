"""Grid-slot dot evidence scoring (Stage 3D-M1).

The blob pipeline decides each cell's dot pattern from whichever blobs
survived thresholding — so every faint dot the detector missed becomes a
misread cell. This module inverts the question: once the grid is fitted we
know the exact position of all six possible dots in every cell, so we ask
"is there dot evidence at this exact spot?" — a far easier decision that
works for dots too faint to survive blob detection.

Evidence is a mode-specific matched filter:

* **emboss** — a raised dot under directional light is a bright/dark
  crescent pair oriented the same way across the whole page. The global
  light direction is estimated from the already-detected (strongest) dots,
  and the filter is an offset Gaussian pair along that axis: the response
  at a true dot centre is strongly positive however faint the relief.
* **dark** — printed/synthetic dots are intensity dips: a centre-surround
  difference-of-Gaussians.

Decisions are made against the page's own background statistics, sampled at
grid positions that are guaranteed empty (the corridor between cell
columns), so the threshold self-calibrates to each capture.

Safety: refinement NEVER fabricates structure — it only re-scores cells on
the already-fitted grid (plus interior gap cells of the same lines). It
fails closed: when the evidence map cannot even confirm the majority of the
dots blob detection already found, the page is returned unchanged with a
flag rather than "improved" by an unreliable signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.models.responses import Flag
from app.ocr.cell_grouping import CellCandidate, GroupingResult, LineGrid
from app.ocr.dot_detection import DetectionOutcome
from app.ocr.flags import (
    CATEGORY_UNCLEAR_BRAILLE_CELL,
    dedupe_flags,
    make_flag,
)

# Minimum cells on the page for background statistics to mean anything.
_MIN_CELLS = 8
# Minimum guaranteed-empty background samples for a usable threshold.
_MIN_BACKGROUND_SAMPLES = 12
# Decision threshold: background mean + this many background std-devs.
_THRESHOLD_SIGMA = 3.0
# Fail-closed sanity: the evidence map must confirm at least this fraction
# of the dots blob detection already found, or refinement is not applied.
_MIN_DETECTED_CONFIRMED = 0.5
# Candidate light directions for the emboss matched filter.
_N_ANGLES = 16


@dataclass
class RefinementOutcome:
    """Result of grid-evidence re-scoring. ``applied`` False = unchanged."""

    applied: bool = False
    reason: str = ""  # why not applied (safe, fixed vocabulary)
    lines: list[list[CellCandidate]] = field(default_factory=list)
    total_cells: int = 0
    cells_changed: int = 0
    cells_recovered: int = 0  # gap cells where evidence found a pattern
    cells_dropped: int = 0  # blob cells where evidence found nothing
    dots_added: int = 0
    dots_removed: int = 0
    low_margin_cells: int = 0
    flags: list[Flag] = field(default_factory=list)


def _flatten(gray: np.ndarray, radius: float) -> np.ndarray:
    """Illumination-corrected signed image (float32, ~0 = paper level)."""
    sigma = max(15.0, 4.0 * radius)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    return gray.astype(np.float32) - background.astype(np.float32)


def _gaussian_kernel(size: int, sigma: float, cx: float, cy: float) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    g = np.exp(-(((xs - cx) ** 2) + ((ys - cy) ** 2)) / (2.0 * sigma * sigma))
    return g / max(float(g.sum()), 1e-6)


def _sample(image: np.ndarray, x: float, y: float) -> float:
    h, w = image.shape
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < w and 0 <= yi < h:
        return float(image[yi, xi])
    return 0.0


def _light_direction(
    signed: np.ndarray, calibration: list[tuple[float, float]], radius: float
) -> tuple[float, float] | None:
    """Global light direction as a unit vector, from the strongest dots.

    For each candidate angle, measure (bright side − dark side) across the
    calibration dots; the true axis maximises the mean response. Returns
    None when no direction separates (flat/diffuse light or wrong mode).
    """
    if not calibration:
        return None
    delta = max(1.5, 0.8 * radius)
    best: tuple[float, float, float] | None = None  # (score, dx, dy)
    for k in range(_N_ANGLES):
        angle = 2.0 * np.pi * k / _N_ANGLES
        dx, dy = float(np.cos(angle)), float(np.sin(angle))
        score = 0.0
        for x, y in calibration:
            score += _sample(signed, x - dx * delta, y - dy * delta) - _sample(
                signed, x + dx * delta, y + dy * delta
            )
        score /= len(calibration)
        if best is None or score > best[0]:
            best = (score, dx, dy)
    if best is None or best[0] <= 1.0:  # < 1 grey level of relief: no signal
        return None
    return best[1], best[2]


def build_evidence_map(
    aligned_gray: np.ndarray,
    mode: str,
    radius: float,
    calibration: list[tuple[float, float]],
) -> np.ndarray | None:
    """Evidence response map: strongly positive at dot centres, ~0 elsewhere."""
    radius = max(2.0, radius)
    signed = _flatten(aligned_gray, radius)
    sigma = max(1.0, 0.6 * radius)
    size = int(2 * np.ceil(3 * sigma + radius) + 1)
    center = (size - 1) / 2.0

    if mode == "emboss":
        direction = _light_direction(signed, calibration, radius)
        if direction is None:
            return None
        dx, dy = direction
        delta = max(1.5, 0.8 * radius)
        bright = _gaussian_kernel(size, sigma, center - dx * delta, center - dy * delta)
        dark = _gaussian_kernel(size, sigma, center + dx * delta, center + dy * delta)
        kernel = bright - dark
    else:
        # Dark dots: centre-surround (positive response where a dip of
        # dot size sits below the local paper level). signed is negated so
        # a dark dot gives a positive response like the emboss path.
        signed = -signed
        center_k = _gaussian_kernel(size, sigma, center, center)
        surround = _gaussian_kernel(size, max(sigma * 3.0, 2.0), center, center)
        kernel = center_k - surround

    return cv2.filter2D(signed, cv2.CV_32F, kernel)


def _window_max(evidence: np.ndarray, x: float, y: float, half: int) -> float:
    """Max evidence in a small window: tolerates ~1-2 px of grid error."""
    h, w = evidence.shape
    xi, yi = int(round(x)), int(round(y))
    x1, x2 = max(0, xi - half), min(w, xi + half + 1)
    y1, y2 = max(0, yi - half), min(h, yi + half + 1)
    if x1 >= x2 or y1 >= y2:
        return 0.0
    return float(evidence[y1:y2, x1:x2].max())


def refine_grouping(
    detection: DetectionOutcome, grouping: GroupingResult
) -> RefinementOutcome:
    """Re-score every fitted cell's dot pattern against image evidence."""
    grid = grouping.grid
    if grid is None or detection.aligned_gray is None:
        return RefinementOutcome(reason="no_grid_or_image")
    if detection.mode != "emboss":
        # The dark path's blob detection is already reliable on the captures
        # it wins (clean scans / synthetic renders) — re-scoring has nothing
        # to add there and must never risk altering a correct controlled
        # result. Evidence re-scoring targets embossed photographs, where
        # faint relief is exactly what blob detection loses.
        return RefinementOutcome(reason="dark_mode_not_refined")
    if grouping.total_cells < _MIN_CELLS:
        return RefinementOutcome(reason="too_few_cells")

    calibration = [(d.x, d.y) for d in detection.dots]
    evidence = build_evidence_map(
        detection.aligned_gray, detection.mode, detection.median_radius, calibration
    )
    if evidence is None:
        return RefinementOutcome(reason="no_light_direction")

    radius = max(2.0, detection.median_radius)
    # Window: large enough to absorb residual grid error after anchoring,
    # small enough never to reach a neighbouring slot (pitch/2 away).
    window = max(1, min(3, int(round(0.25 * grid.u_v))))
    slope = grid.skew_slope

    def image_xy(gx: float, gy: float) -> tuple[float, float]:
        # Grid space is the shear-corrected frame: undo it for sampling.
        return gx, gy + slope * gx

    line_grids = {lg.line_number: lg for lg in grid.lines}

    # --- Local anchoring -----------------------------------------------------
    # The global line fit drifts by several px along a real page (pitch and
    # origin error accumulate over 30 cells), which is enough to move a slot
    # centre off its dot entirely. Anchor each cell's slot grid to its own
    # detected dots, and interpolate the correction across cells that have
    # none (gaps, or cells whose dots were all missed by blob detection).
    dot_xy = (
        np.array([[d.x, d.y] for d in detection.dots], dtype=np.float32)
        if detection.dots
        else np.zeros((0, 2), dtype=np.float32)
    )
    snap_limit_sq = (0.6 * grid.u_v) ** 2
    # anchors[line_number] = sorted list of (cell_idx, dx, dy)
    anchors: dict[int, list[tuple[int, float, float]]] = {}
    for line in grouping.lines:
        for cell in line:
            if cell.slot_centers is None or not len(dot_xy):
                continue
            offsets: list[tuple[float, float]] = []
            for dot_number in cell.dots:
                sx, sy = image_xy(*cell.slot_centers[dot_number - 1])
                d2 = ((dot_xy - (sx, sy)) ** 2).sum(axis=1)
                j = int(d2.argmin())
                if d2[j] <= snap_limit_sq:
                    offsets.append(
                        (float(dot_xy[j][0]) - sx, float(dot_xy[j][1]) - sy)
                    )
            if offsets:
                anchors.setdefault(cell.line_number, []).append(
                    (
                        cell.grid_index,
                        float(np.mean([o[0] for o in offsets])),
                        float(np.mean([o[1] for o in offsets])),
                    )
                )
    for entries in anchors.values():
        entries.sort()

    def local_offset(line_number: int, cell_idx: int) -> tuple[float, float]:
        """Interpolated anchoring correction for one cell."""
        entries = anchors.get(line_number)
        if not entries:
            return 0.0, 0.0
        # Exact or interpolate between the nearest anchors either side.
        before = None
        after = None
        for idx, dx, dy in entries:
            if idx == cell_idx:
                return dx, dy
            if idx < cell_idx:
                before = (idx, dx, dy)
            elif after is None:
                after = (idx, dx, dy)
                break
        if before and after:
            span = after[0] - before[0]
            t = (cell_idx - before[0]) / span if span else 0.0
            return (
                before[1] + t * (after[1] - before[1]),
                before[2] + t * (after[2] - before[2]),
            )
        nearest = before or after
        return nearest[1], nearest[2]

    def anchored_slots(
        lg: LineGrid, cell_idx: int
    ) -> tuple[tuple[float, float], ...]:
        dx, dy = local_offset(lg.line_number, cell_idx)
        return tuple(
            (x + dx, y + dy)
            for x, y in (
                image_xy(gx, gy) for gx, gy in grid.slot_centers(lg, cell_idx)
            )
        )

    # --- Background statistics from guaranteed-empty corridor positions ----
    # The corridor between the second column of one cell and the first column
    # of the next never contains a dot; sampled with the same local anchoring
    # as the slots so grid drift cannot leak dot energy into the background.
    background: list[float] = []
    corridor_off = grid.u_h + (grid.advance - grid.u_h) / 2.0
    for lg in grid.lines:
        for cell_idx in range(lg.first_cell_idx, lg.last_cell_idx):
            dx, dy = local_offset(lg.line_number, cell_idx)
            cx = lg.origin_x + cell_idx * grid.advance + corridor_off
            for row in (0, 1, 2):
                x, y = image_xy(cx, lg.y_top + row * grid.u_v)
                background.append(
                    _window_max(evidence, x + dx, y + dy, window)
                )
    if len(background) < _MIN_BACKGROUND_SAMPLES:
        return RefinementOutcome(reason="too_few_background_samples")
    bg = np.asarray(background, dtype=np.float32)
    # Robust statistics: on annotated/dirty pages the corridor contains ink
    # marks whose scores would inflate a plain std and push the threshold
    # into the true-dot band. Median + MAD ignores that tail.
    bg_median = float(np.median(bg))
    bg_mad = float(np.median(np.abs(bg - bg_median)))
    bg_std = 1.4826 * bg_mad
    if bg_std < 1e-3:
        bg_std = float(bg.std())  # constant-ish corridor: fall back
    if bg_std < 1e-3:
        return RefinementOutcome(reason="degenerate_background")
    threshold = bg_median + _THRESHOLD_SIGMA * bg_std

    # --- Fail-closed sanity: evidence must confirm the blob dots -----------
    confirmed = 0
    total_known = 0
    for line in grouping.lines:
        lg = line_grids.get(line[0].line_number) if line else None
        if lg is None:
            continue
        for cell in line:
            slots = anchored_slots(lg, cell.grid_index)
            for dot_number in cell.dots:
                x, y = slots[dot_number - 1]
                total_known += 1
                if _window_max(evidence, x, y, window) >= threshold:
                    confirmed += 1
    if total_known == 0 or confirmed / total_known < _MIN_DETECTED_CONFIRMED:
        return RefinementOutcome(reason="evidence_contradicts_detection")

    # --- Re-score every grid cell (including interior gaps) ----------------
    outcome = RefinementOutcome(applied=True)
    blob_patterns: dict[tuple[int, int], tuple[int, ...]] = {}
    blob_confidence: dict[tuple[int, int], float] = {}
    for line in grouping.lines:
        for cell in line:
            blob_patterns[(cell.line_number, cell.grid_index)] = cell.dots
            blob_confidence[(cell.line_number, cell.grid_index)] = cell.confidence

    new_lines: list[list[CellCandidate]] = []
    for lg in sorted(line_grids.values(), key=lambda g: g.line_number):
        new_line: list[CellCandidate] = []
        for cell_idx in range(lg.first_cell_idx, lg.last_cell_idx + 1):
            slots = anchored_slots(lg, cell_idx)
            scores = [_window_max(evidence, x, y, window) for x, y in slots]
            pattern = tuple(
                n for n, s in enumerate(scores, start=1) if s >= threshold
            )
            key = (lg.line_number, cell_idx)
            blob = blob_patterns.get(key)

            if not pattern:
                if blob:
                    outcome.cells_dropped += 1
                    outcome.cells_changed += 1
                    outcome.dots_removed += len(blob)
                continue

            on = [scores[n - 1] for n in pattern]
            margin = (min(on) - threshold) / (_THRESHOLD_SIGMA * bg_std)
            confidence = float(np.clip(0.55 + 0.15 * margin, 0.0, 0.99))
            if margin < 0.5:
                outcome.low_margin_cells += 1

            if blob is None:
                outcome.cells_recovered += 1
                outcome.cells_changed += 1
                outcome.dots_added += len(pattern)
            elif blob != pattern:
                outcome.cells_changed += 1
                outcome.dots_added += len(set(pattern) - set(blob))
                outcome.dots_removed += len(set(blob) - set(pattern))
            else:
                # Same pattern: keep the (usually higher) blob confidence.
                confidence = max(confidence, blob_confidence.get(key, 0.0))

            xs = [x for x, _ in slots]
            ys = [y for _, y in slots]
            new_line.append(
                CellCandidate(
                    line_number=lg.line_number,
                    grid_index=cell_idx,
                    dots=pattern,
                    bbox=(
                        int(min(xs) - radius),
                        int(min(ys) - radius),
                        int(round(max(xs) + radius)),
                        int(round(max(ys) + radius)),
                    ),
                    confidence=round(confidence, 3),
                    slot_centers=grid.slot_centers(lg, cell_idx),
                )
            )
        if new_line:
            new_lines.append(new_line)

    outcome.lines = new_lines
    outcome.total_cells = sum(len(line) for line in new_lines)

    if outcome.cells_changed:
        outcome.flags.append(
            make_flag(
                text="",
                reason=(
                    f"{outcome.cells_changed} Braille cell(s) were re-read "
                    "directly from the fitted dot grid (faint-dot evidence "
                    "scoring); the draft reflects the re-read cells and "
                    "should be checked against the page."
                ),
                category=CATEGORY_UNCLEAR_BRAILLE_CELL,
                severity="low",
            )
        )
    if outcome.low_margin_cells:
        outcome.flags.append(
            make_flag(
                text="",
                reason=(
                    f"{outcome.low_margin_cells} re-read cell(s) had dot "
                    "evidence only marginally above the page background; "
                    "treat these cells with particular caution."
                ),
                category=CATEGORY_UNCLEAR_BRAILLE_CELL,
                severity="medium",
            )
        )
    outcome.flags = dedupe_flags(outcome.flags)
    return outcome
