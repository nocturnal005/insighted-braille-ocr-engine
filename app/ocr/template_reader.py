"""Self-calibrated template-matching page reader (Stage 3D-N1).

Real embossed-paper photographs defeat blob detection: at the downscale the
rest of the pipeline is calibrated for, a raised dot is a faint low-contrast
crescent that thresholding either drops or splits, so cell patterns come out
mostly wrong even when row structure is recovered. This module reads such
pages a different way and at full resolution:

1. **Self-calibrate a dot template.** The strongest highlight peaks on the
   page are averaged (after a median-consistency filter) into one canonical
   dot appearance — this capture's exact relief shape and light direction.
2. **Match it everywhere.** Normalised cross-correlation against that template
   locates every dot, however faint, because a faint dot and a strong dot
   share the same *shape*; ink annotations and paper texture do not correlate
   with the embossed crescent and are rejected.
3. **Fit the lattice deterministically.** Dots are peeled into Braille lines
   top-down, each line fitted to the regular 3-row dot lattice by a bounded
   slope/offset search (no randomness — same image always reads the same).
4. **Snap to cells.** Columns are paired into 2-column cells at the measured
   dot pitch; each cell's raised slots become dot numbers 1-6.

The output is an ordinary ``GroupingResult`` of dot-pattern cells plus the
fitted ``PageGrid`` — identical in shape to what ``group_dots`` produces — so
Unicode conversion and Grade 1/2 back-translation downstream are unchanged.
The letter/contraction decode is deliberately NOT done here; this module only
locates dots. It never runs on clean scans or synthetic renders (the pipeline
gates it behind a real-photo readability signal) and it fails closed: when the
matched dots do not form a self-consistent, readable lattice it returns
``None`` and the standard result stands.

Safety: this path produces a draft for QTVI / specialist verification like
every other path. It is capped at the embossed-photo confidence ceiling and
carries an explicit "experimental reader" flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.models.responses import Flag
from app.ocr.cell_grouping import (
    CellCandidate,
    GroupingResult,
    LineGrid,
    PageGrid,
)
from app.ocr.dot_detection import Dot, DetectionOutcome, spacing_regularity
from app.ocr.flags import (
    CATEGORY_LINE_ORDER_UNCERTAINTY,
    CATEGORY_LOW_OCR_CONFIDENCE,
    make_flag,
)
from app.ocr.preprocessing import MODE_EMBOSS

# Full resolution matters: the whole point is to read at the scale the blob
# pipeline downscales away. Only genuinely huge captures are shrunk, purely to
# bound compute; the pitch estimate adapts to whatever scale survives.
_MAX_LONG_SIDE = 4200

# Template self-calibration.
_SEED_PEAK_SIGMA = 6.0  # highlight peaks this many robust-std above paper
_MAX_SEED_PEAKS = 300
_MIN_SEED_PEAKS = 40
_TEMPLATE_CONSISTENCY_PCT = 40  # drop the least template-like 40% of seeds

# Detection.
_NCC_THRESHOLD = 0.60  # real embossed dots plateau well above this
_NMS_FACTOR = 0.55  # suppress peaks closer than this fraction of the pitch

# Lattice fit (deterministic bounded search — no RANSAC randomness).
_SLOPE_LIMIT = 0.06  # max plausible per-row tilt (dy/dx) after any deskew
_SLOPE_STEPS = 25
_OFFSET_STEPS = 30
_ROW_INLIER_FRAC = 0.33  # dot counts as on-row within this fraction of pitch

# Acceptance gates (fail-closed). Below any of these, read_page returns None.
_MIN_DOTS = 24
_MIN_CELLS = 12
_MIN_LETTER_FRACTION = 0.45  # fraction of cells that back-translate to letters
_MIN_ROW_LETTER_CELLS = 2  # a kept line needs at least this many letter cells

_ROTATIONS = (0, 180, 90, 270)


@dataclass
class TemplateReadResult:
    """A full-resolution template read, shaped for the pipeline."""

    detection: DetectionOutcome
    grouping: GroupingResult
    rotation_applied: int  # degrees counter-clockwise applied to the input
    letter_fraction: float  # self-consistency: cells that read as letters
    flags: list[Flag] = field(default_factory=list)


# --- scale -------------------------------------------------------------------


def _limit_scale(gray: np.ndarray) -> np.ndarray:
    long_side = max(gray.shape[:2])
    if long_side <= _MAX_LONG_SIDE:
        return gray
    scale = _MAX_LONG_SIDE / long_side
    h, w = gray.shape[:2]
    return cv2.resize(
        gray,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _highpass(gray: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    return cv2.GaussianBlur(gray.astype(np.float32) - blur, (0, 0), 3.0)


# --- pitch + template --------------------------------------------------------


def _seed_peaks(hp: np.ndarray) -> np.ndarray:
    """Strongest highlight local maxima as (y, x) rows, brightest first."""
    med = float(np.median(hp))
    mad = float(np.median(np.abs(hp - med))) * 1.4826 + 1e-6
    dilated = cv2.dilate(hp, np.ones((15, 15), np.uint8))
    peaks = (hp >= dilated - 1e-6) & (hp > med + _SEED_PEAK_SIGMA * mad)
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(-hp[ys, xs])[:_MAX_SEED_PEAKS]
    return np.stack([ys[order], xs[order]], axis=1)


def _nn_median(points: np.ndarray) -> float | None:
    """Median nearest-neighbour distance, biased to the in-cell dot pitch.

    Most Braille dots have a same-cell neighbour one pitch away, so the lower
    half of the nearest-neighbour distribution concentrates at the dot pitch;
    its median is a robust pitch estimate that ignores the larger cell-advance
    and inter-line gaps in the upper half.
    """
    if len(points) < 4:
        return None
    pts = points.astype(np.float64)
    diffs = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diffs**2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest = np.sort(dist.min(axis=1))
    lower = nearest[: max(1, len(nearest) // 2)]
    value = float(np.median(lower))
    return value if value > 1e-3 else None


def _build_template(gray: np.ndarray, hp: np.ndarray, seeds: np.ndarray,
                    radius: int) -> np.ndarray | None:
    """Average the seed dot patches into one zero-mean, unit-norm template."""
    g = gray.astype(np.float32)
    h, w = g.shape
    patches = []
    for y, x in seeds:
        if radius <= y < h - radius and radius <= x < w - radius:
            patch = g[y - radius:y + radius + 1, x - radius:x + radius + 1].copy()
            patch -= patch.mean()
            norm = np.linalg.norm(patch)
            if norm > 0:
                patches.append(patch / norm)
    if len(patches) < 30:
        return None
    stack = np.stack(patches)
    med = np.median(stack, axis=0)
    med -= med.mean()
    med /= np.linalg.norm(med) + 1e-9
    consistency = stack.reshape(len(stack), -1) @ med.ravel()
    keep = consistency > np.percentile(consistency, _TEMPLATE_CONSISTENCY_PCT)
    template = stack[keep].mean(axis=0) if keep.any() else stack.mean(axis=0)
    template -= template.mean()
    template /= np.linalg.norm(template) + 1e-9
    return template.astype(np.float32)


def _match_dots(gray: np.ndarray, template: np.ndarray, pitch: float) -> list[
    tuple[float, float, float]
]:
    """NCC match + non-max suppression → list of (x, y, score)."""
    ncc = cv2.matchTemplate(gray.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    dilated = cv2.dilate(ncc, np.ones((15, 15), np.uint8))
    peaks = (ncc >= dilated - 1e-6) & (ncc > _NCC_THRESHOLD)
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return []
    off = template.shape[0] // 2
    scores = ncc[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]
    taken = np.zeros(len(ys), dtype=bool)
    radius2 = (_NMS_FACTOR * pitch) ** 2
    out: list[tuple[float, float, float]] = []
    for i in range(len(ys)):
        if taken[i]:
            continue
        d2 = (ys - ys[i]) ** 2 + (xs - xs[i]) ** 2
        taken |= d2 < radius2
        out.append((float(xs[i] + off), float(ys[i] + off), float(scores[i])))
    return out


# --- lattice + cells ---------------------------------------------------------


def _fit_row_lattice(
    band: list[tuple[float, float, float]], pitch: float
) -> tuple[float, float] | None:
    """Fit ``y = a + slope*x + row*pitch`` (row in 0..2) to a band of dots.

    Deterministic bounded search over slope and vertical offset, then a
    least-squares refine on the inliers. Returns ``(a, slope)`` or None.
    """
    if len(band) < 3:
        return None
    xs = np.array([p[0] for p in band])
    ys = np.array([p[1] for p in band])
    tol = _ROW_INLIER_FRAC * pitch
    best: tuple[int, float, float, float] | None = None  # (inliers,-resid,a,slope)
    for slope in np.linspace(-_SLOPE_LIMIT, _SLOPE_LIMIT, _SLOPE_STEPS):
        r = ys - slope * xs
        r0 = float(r.min())
        for a in np.linspace(r0, r0 + 2.0 * pitch, _OFFSET_STEPS):
            row = np.round((r - a) / pitch)
            resid = np.abs(r - a - row * pitch)
            ok = (resid < tol) & (row >= 0) & (row <= 2)
            n = int(ok.sum())
            if n == 0:
                continue
            score = (n, -float(resid[ok].mean()), float(a), float(slope))
            if best is None or score[:2] > best[:2]:
                best = score
    if best is None:
        return None
    _, _, a, slope = best
    r = ys - slope * xs
    row = np.round((r - a) / pitch)
    ok = (np.abs(r - a - row * pitch) < tol) & (row >= 0) & (row <= 2)
    if ok.sum() >= 3:
        A = np.stack([np.ones(int(ok.sum())), xs[ok]], axis=1)
        target = ys[ok] - row[ok] * pitch
        sol, *_ = np.linalg.lstsq(A, target, rcond=None)
        a, slope = float(sol[0]), float(sol[1])
    return a, slope


def _split_rows(
    dots: list[tuple[float, float, float]], pitch: float
) -> list[tuple[float, float, list[tuple[float, float, float]]]]:
    """Peel dots into Braille lines top-down. Returns (a, slope, dots) each."""
    remaining = list(dots)
    rows: list[tuple[float, float, list[tuple[float, float, float]]]] = []
    guard = 0
    while len(remaining) >= 3 and guard < 80:
        guard += 1
        remaining.sort(key=lambda p: p[1])
        y0 = remaining[0][1]
        band = [p for p in remaining if p[1] < y0 + 3.2 * pitch]
        fit = _fit_row_lattice(band, pitch)
        if fit is None:
            remaining = remaining[3:]
            continue
        a, slope = fit
        tol = _ROW_INLIER_FRAC * pitch
        inliers, rest = [], []
        for p in remaining:
            r = p[1] - (a + slope * p[0])
            row = round(r / pitch)
            if 0 <= row <= 2 and abs(r - row * pitch) < tol:
                inliers.append(p)
            else:
                rest.append(p)
        if len(inliers) < 3:
            remaining = remaining[3:]
            continue
        rows.append((a, slope, inliers))
        remaining = rest
    rows.sort(key=lambda r: r[0])
    return rows


def _cluster_columns(xs: list[float], pitch: float) -> list[float]:
    """Merge dot x-positions within ~half a pitch into column centres."""
    ordered = sorted(xs)
    clusters = [[ordered[0]]]
    for x in ordered[1:]:
        if x - clusters[-1][-1] < 0.55 * pitch:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [sum(c) / len(c) for c in clusters]


def _estimate_spacing(
    rows: list[tuple[float, float, list[tuple[float, float, float]]]], pitch: float
) -> tuple[float, float]:
    """Measure within-cell column pitch and cell advance from the data.

    Consecutive column-centre gaps form distinct populations: the within-cell
    gap (~one dot pitch) and the between-cell gap (cell advance minus one
    pitch). Estimating both from the page instead of assuming a fixed multiple
    of the dot pitch keeps cell pairing correct when the dot-pitch estimate is
    slightly off — the two populations sit close together, so a fixed threshold
    misclassifies half of them and merges adjacent cells.
    """
    gaps: list[float] = []
    for _a, _slope, dots in rows:
        centres = _cluster_columns([x for x, _y, _s in dots], pitch)
        gaps.extend(b - a for a, b in zip(centres, centres[1:]) if b - a > 0)
    if not gaps:
        return pitch, 2.5 * pitch
    within = [g for g in gaps if 0.6 * pitch <= g <= 1.3 * pitch]
    u_h = float(np.median(within)) if within else pitch
    # Normalise each between-cell population to a full advance before averaging.
    advance_estimates = [g + u_h for g in gaps if 1.3 * u_h < g <= 2.0 * u_h] + [
        g for g in gaps if 2.0 * u_h < g <= 3.2 * u_h
    ]
    advance = float(np.median(advance_estimates)) if advance_estimates else 2.5 * u_h
    # Guard against a degenerate estimate collapsing the cell width.
    advance = max(advance, 1.8 * u_h)
    return u_h, advance


def _line_cells(
    line_number: int,
    a: float,
    slope: float,
    dots: list[tuple[float, float, float]],
    pitch: float,
    u_h: float,
    advance: float,
) -> tuple[list[CellCandidate], LineGrid] | None:
    """Snap one fitted row of dots into 2-column Braille cells."""
    placed = []  # (x, row, score)
    for x, y, score in dots:
        r = y - (a + slope * x)
        row = int(round(r / pitch))
        if 0 <= row <= 2:
            placed.append((x, row, score))
    if not placed:
        return None
    col_centres = _cluster_columns([p[0] for p in placed], u_h)
    if not col_centres:
        return None

    # Pair adjacent columns into cells. Within a cell the two columns are ~one
    # dot pitch apart; the gap to the next cell's left column is the advance
    # minus one pitch — a larger value. Split at the midpoint of the measured
    # advance so a slightly-off pitch estimate cannot merge adjacent cells.
    pair_threshold = 0.5 * advance
    cells: list[tuple[float, float | None]] = []
    i = 0
    while i < len(col_centres):
        if i + 1 < len(col_centres) and (
            col_centres[i + 1] - col_centres[i]
        ) < pair_threshold:
            cells.append((col_centres[i], col_centres[i + 1]))
            i += 2
        else:
            cells.append((col_centres[i], None))
            i += 1

    origin_x = cells[0][0]
    candidates: list[tuple[int, tuple[int, ...], float, tuple[int, int, int, int]]] = []
    for c1, c2 in cells:
        pattern: set[int] = set()
        scores: list[float] = []
        xs_in: list[float] = []
        ys_in: list[float] = []
        for x, row, score in placed:
            if abs(x - c1) < 0.5 * u_h:
                pattern.add(row + 1)
            elif c2 is not None and abs(x - c2) < 0.5 * u_h:
                pattern.add(row + 4)
            else:
                continue
            scores.append(score)
            xs_in.append(x)
            ys_in.append(a + slope * x + row * pitch)
        if not pattern:
            continue
        grid_index = int(round((c1 - origin_x) / advance))
        half = 0.5 * pitch
        x1 = int(min(xs_in) - half)
        y1 = int(min(ys_in) - half)
        x2 = int(round(max(xs_in) + half))
        y2 = int(round(max(ys_in) + half))
        candidates.append((grid_index, tuple(sorted(pattern)),
                           float(np.mean(scores)), (x1, y1, x2, y2)))

    # Collapse any cells that rounded to the same grid index (rare overlap).
    by_index: dict[int, tuple[tuple[int, ...], float, tuple[int, int, int, int]]] = {}
    for grid_index, pattern, conf, bbox in candidates:
        if grid_index not in by_index or conf > by_index[grid_index][1]:
            by_index[grid_index] = (pattern, conf, bbox)
    if not by_index:
        return None

    _drop_erasure_runs(by_index)
    if not by_index:
        return None

    line_cells = [
        CellCandidate(
            line_number=line_number,
            grid_index=idx,
            dots=pattern,
            bbox=bbox,
            confidence=round(min(1.0, conf), 3),
        )
        for idx, (pattern, conf, bbox) in sorted(by_index.items())
    ]
    line_grid = LineGrid(
        line_number=line_number,
        y_top=float(a),
        origin_x=float(origin_x),
        first_cell_idx=min(by_index),
        last_cell_idx=max(by_index),
    )
    return line_cells, line_grid


_FULL_CELL = (1, 2, 3, 4, 5, 6)


def _drop_erasure_runs(
    by_index: dict[int, tuple[tuple[int, ...], float, tuple[int, int, int, int]]],
) -> None:
    """Remove runs of >=2 adjacent all-six-dot cells (pencil erasure blocks).

    A single all-six cell can be a legitimate Grade 2 sign (``for``), so it is
    kept; but two or more in a row on consecutive grid positions is a pupil's
    scribbled-out erasure, never text. Dropping the run — rather than emitting
    it as unreadable cells — is faithful to the page (this is exactly how the
    teacher transcriptions mark such blocks) and mutates ``by_index`` in place.
    """
    full = sorted(
        i for i, (pattern, _c, _b) in by_index.items() if pattern == _FULL_CELL
    )
    run: list[int] = []
    to_drop: list[int] = []
    for idx in full + [None]:  # sentinel flushes the final run
        if run and (idx is None or idx != run[-1] + 1):
            if len(run) >= 2:
                to_drop.extend(run)
            run = []
        if idx is not None:
            run.append(idx)
    for idx in to_drop:
        by_index.pop(idx, None)


def _letter_fraction(grouping: GroupingResult) -> float:
    """Fraction of cells the Grade 1 fallback reads as a letter (self-check).

    A wrong orientation or a noise fit is dominated by patterns that map to
    nothing; a true page is dominated by letters. Uses the same fallback
    translator the rest of the pipeline uses, so the signal is consistent.
    Never raises: any failure means "no evidence", scored 0.
    """
    try:
        from app.ocr.braille_decode import dots_to_unicode_char
        from app.translation.fallback_translator import back_translate_unicode_lines

        total = letters = 0
        for line in grouping.lines:
            for cell in line:
                total += 1
                unicode_char = dots_to_unicode_char(cell.dots)
                decoded = back_translate_unicode_lines([unicode_char])
                text = decoded.text.strip()
                if len(text) == 1 and text.isalpha():
                    letters += 1
        return letters / total if total else 0.0
    except Exception:
        return 0.0


def _read_one_orientation(gray: np.ndarray) -> TemplateReadResult | None:
    """Full template read at a single orientation, or None (fail-closed)."""
    hp = _highpass(gray)
    seeds = _seed_peaks(hp)
    if len(seeds) < _MIN_SEED_PEAKS:
        return None
    coarse_pitch = _nn_median(seeds)
    if coarse_pitch is None or coarse_pitch < 6.0:
        return None
    radius = int(round(min(40, max(8, 0.67 * coarse_pitch))))
    template = _build_template(gray, hp, seeds, radius)
    if template is None:
        return None

    dots_xy = _match_dots(gray, template, coarse_pitch)
    if len(dots_xy) < _MIN_DOTS:
        return None
    pitch = _nn_median(np.array([[y, x] for x, y, _ in dots_xy])) or coarse_pitch
    if pitch < 6.0:
        return None

    rows = _split_rows(dots_xy, pitch)
    if not rows:
        return None

    # Measure the within-cell column pitch and the cell advance from the page
    # rather than assuming a fixed multiple of the dot pitch.
    u_h, advance = _estimate_spacing(rows, pitch)

    lines_out: list[list[CellCandidate]] = []
    grid = PageGrid(u_v=pitch, u_h=u_h, advance=advance, skew_slope=0.0)
    line_number = 0
    for a, slope, row_dots in rows:
        line_number += 1
        built = _line_cells(line_number, a, slope, row_dots, pitch, u_h, advance)
        if built is None:
            line_number -= 1
            continue
        cells, line_grid = built
        letter_cells = _count_letter_cells(cells)
        if letter_cells < _MIN_ROW_LETTER_CELLS:
            line_number -= 1
            continue
        lines_out.append(cells)
        grid.lines.append(line_grid)

    # Renumber kept lines 1..n so downstream line numbers stay contiguous.
    for new_number, (cells, line_grid) in enumerate(zip(lines_out, grid.lines), 1):
        for cell in cells:
            cell.line_number = new_number
        line_grid.line_number = new_number

    total_cells = sum(len(line) for line in lines_out)
    if total_cells < _MIN_CELLS:
        return None

    used_dots = _dots_from_grid(lines_out, grid)
    detection = DetectionOutcome(
        dots=used_dots,
        quality=round(min(1.0, float(np.mean([d.confidence for d in used_dots]))), 3),
        mode=MODE_EMBOSS,
        image_quality=0.6,
        spacing_regularity=spacing_regularity(used_dots),
        raw_candidates=len(dots_xy),
        median_radius=max(1.0, 0.2 * pitch),
        aligned_gray=None,  # M1 evidence refinement no-ops on the template path
    )
    grouping = GroupingResult(
        lines=lines_out,
        quality=_grid_quality(rows, pitch),
        line_quality=0.7,
        flags=[],
        total_cells=total_cells,
        recovered_via_fallback=False,
        grid=grid if grid.lines else None,
    )
    letter_fraction = _letter_fraction(grouping)
    return TemplateReadResult(
        detection=detection,
        grouping=grouping,
        rotation_applied=0,
        letter_fraction=letter_fraction,
    )


def _count_letter_cells(cells: list[CellCandidate]) -> int:
    try:
        from app.ocr.braille_decode import dots_to_unicode_char
        from app.translation.fallback_translator import back_translate_unicode_lines

        count = 0
        for cell in cells:
            if len(cell.dots) < 2:
                continue
            decoded = back_translate_unicode_lines(
                [dots_to_unicode_char(cell.dots)]
            ).text.strip()
            if len(decoded) == 1 and decoded.isalpha():
                count += 1
        return count
    except Exception:
        return 0


def _dots_from_grid(lines: list[list[CellCandidate]], grid: PageGrid) -> list[Dot]:
    """Reconstruct a Dot at each raised slot's fitted grid position.

    Distinct per-slot positions (not the cell centre) so nearest-neighbour
    spacing regularity reflects the real dot lattice.
    """
    dots: list[Dot] = []
    r = max(1.0, 0.2 * grid.u_v)
    by_line = {lg.line_number: lg for lg in grid.lines}
    for line in lines:
        for cell in line:
            line_grid = by_line.get(cell.line_number)
            if line_grid is None:
                continue
            for dot in cell.dots:
                col, row = (dot - 1) // 3, (dot - 1) % 3
                x = line_grid.origin_x + cell.grid_index * grid.advance + col * grid.u_h
                y = line_grid.y_top + row * grid.u_v
                dots.append(
                    Dot(x=float(x), y=float(y), r=r, confidence=cell.confidence,
                        area=float(np.pi * r * r), bbox=cell.bbox)
                )
    return dots


def _grid_quality(rows, pitch: float) -> float:
    """Mean lattice inlier tightness across rows, mapped to [0, 1]."""
    residuals: list[float] = []
    for a, slope, dots in rows:
        for x, y, _ in dots:
            r = y - (a + slope * x)
            row = round(r / pitch)
            residuals.append(abs(r - row * pitch) / max(pitch, 1e-6))
    if not residuals:
        return 0.0
    return float(np.clip(1.0 - 2.0 * float(np.mean(residuals)), 0.0, 1.0))


def _rotate(gray: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return gray
    return np.ascontiguousarray(np.rot90(gray, k=degrees // 90))


def read_page(gray: np.ndarray) -> TemplateReadResult | None:
    """Read an embossed page by self-calibrated template matching.

    Tries the four right-angle orientations and keeps the most readable one.
    Returns None (the standard pipeline result then stands) whenever no
    orientation yields a self-consistent, readable lattice.
    """
    if gray is None or gray.ndim != 2 or min(gray.shape) < 40:
        return None
    scaled = _limit_scale(gray)

    best: TemplateReadResult | None = None
    for degrees in _ROTATIONS:
        try:
            result = _read_one_orientation(_rotate(scaled, degrees))
        except Exception:
            result = None
        if result is None:
            continue
        result.rotation_applied = degrees
        key = (result.letter_fraction, result.grouping.total_cells)
        if best is None or key > (best.letter_fraction, best.grouping.total_cells):
            best = result

    if best is None or best.letter_fraction < _MIN_LETTER_FRACTION:
        return None

    flags = [
        make_flag(
            text="",
            reason=(
                "This draft was produced by the experimental full-resolution "
                "template reader for embossed photographs; cell and line "
                "structure is inferred and must be checked carefully by a "
                "specialist."
            ),
            category=CATEGORY_LOW_OCR_CONFIDENCE,
            severity="medium",
        )
    ]
    if best.rotation_applied:
        flags.append(
            make_flag(
                text="",
                reason=(
                    "The page appeared rotated; the draft was produced after "
                    f"automatic rotation by {best.rotation_applied} degrees. Cell "
                    "positions refer to the rotated image and line order should "
                    "be checked."
                ),
                category=CATEGORY_LINE_ORDER_UNCERTAINTY,
                severity="medium",
            )
        )
    # Flags travel on the result only; grouping.flags stays empty so the
    # pipeline's separate ``flags.extend(grouping.flags)`` cannot double-add.
    best.flags = flags
    return best
