"""Group detected dots into 6-dot Braille cells on a 2-column x 3-row grid.

The approach fits a regular grid: it estimates the dot pitch (the distance
between adjacent dot positions) vertically and horizontally, the cell
advance (distance between cell origins), then assigns each dot to a
(line, cell, dot-position) slot. Both possible anchor alignments for each
line are tried (the leftmost detected column may be column 1 or column 2 of
its cell, e.g. when a line starts with a capital sign), and the alignment
with the lowest total grid residual wins.

Stage 3D-D robustness for embossed photographs:

* Residual skew correction — image-level deskew only triggers above ~0.7
  degrees; smaller camera tilt still smears dot rows together. The median
  per-row slope of the detected dot centres is measured and sheared out
  analytically before clustering, so mild skew no longer merges rows.
  (Cell bounding boxes are therefore reported in this corrected space,
  consistent with the image-level deskew that already rewarps the page.)
* Pitch-refined re-clustering — the first row clustering uses a threshold
  derived from dot radius; once the true vertical dot pitch is measured,
  rows are re-clustered with a pitch-based threshold, which handles pages
  with unusually tight or wide dot spacing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import median

from app.models.responses import Flag
from app.ocr.dot_detection import Dot
from app.ocr.flags import (
    CATEGORY_LINE_ORDER_UNCERTAINTY,
    CATEGORY_LOW_IMAGE_QUALITY,
    CATEGORY_LOW_OCR_CONFIDENCE,
    CATEGORY_UNCLEAR_BRAILLE_CELL,
    dedupe_flags,
    make_flag,
)


@dataclass
class CellCandidate:
    line_number: int  # 1-based, top to bottom
    grid_index: int  # 0-based grid position within the line (spaces occupy indices)
    dots: tuple[int, ...]  # sorted dot numbers 1-6
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


@dataclass
class GroupingResult:
    lines: list[list[CellCandidate]] = field(default_factory=list)
    quality: float = 0.0  # grid-fit quality in [0, 1]
    line_quality: float = 0.0  # line/row-order certainty in [0, 1]
    flags: list[Flag] = field(default_factory=list)
    total_cells: int = 0


def _cluster_1d(items: list, key, threshold: float) -> list[list]:
    """Cluster items along one axis; items closer than threshold share a cluster."""
    ordered = sorted(items, key=key)
    clusters: list[list] = [[ordered[0]]]
    for item in ordered[1:]:
        if key(item) - key(clusters[-1][-1]) <= threshold:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def _estimate_unit(gaps: list[float], fallback: float) -> float:
    """Estimate the base dot pitch from a list of gaps (smallest gap cluster)."""
    positive = [g for g in gaps if g > 1e-6]
    if not positive:
        return fallback
    smallest = min(positive)
    close = [g for g in positive if g <= 1.6 * smallest]
    return median(close)


def _fit_line_columns(
    col_centers: list[float], u_h: float, advance: float
) -> tuple[list[tuple[int, int, float]], float]:
    """Assign each column to (cell_index, column_in_cell) via grid fitting.

    Tries both anchor alignments (leftmost column = column 1 or column 2)
    and returns the assignment with the lowest total residual, plus the
    mean normalised residual for quality scoring.
    """
    x0 = col_centers[0]
    best_assignment: list[tuple[int, int, float]] | None = None
    best_total = float("inf")

    for anchor_offset in (0.0, u_h):
        total = 0.0
        assignment: list[tuple[int, int, float]] = []
        for cx in col_centers:
            rel = cx - x0 + anchor_offset
            cell_1 = round(rel / advance)
            resid_1 = abs(rel - cell_1 * advance)
            cell_2 = round((rel - u_h) / advance)
            resid_2 = abs(rel - u_h - cell_2 * advance)
            if resid_1 <= resid_2:
                cell_idx, col_in_cell, resid = int(cell_1), 0, resid_1
            else:
                cell_idx, col_in_cell, resid = int(cell_2), 1, resid_2
            if cell_idx < 0:
                cell_idx = 0
                resid += u_h  # penalise impossible placements
            assignment.append((cell_idx, col_in_cell, resid))
            total += resid
        if total < best_total - 1e-9:
            best_total = total
            best_assignment = assignment

    assert best_assignment is not None
    residuals = [r / max(u_h, 1e-6) for _, _, r in best_assignment]
    mean_residual = sum(residuals) / len(residuals)
    return best_assignment, mean_residual


def _row_structure(
    dots: list[Dot], threshold: float, fallback_unit: float
) -> tuple[list[list[Dot]], list[float], float]:
    """Cluster dots into horizontal rows and estimate the vertical dot pitch."""
    rows = _cluster_1d(dots, key=lambda d: d.y, threshold=threshold)
    row_centers = [sum(d.y for d in row) / len(row) for row in rows]
    row_gaps = [b - a for a, b in zip(row_centers, row_centers[1:])]
    u_v = _estimate_unit(row_gaps, fallback=fallback_unit)
    return rows, row_centers, u_v


def _estimate_residual_skew(rows: list[list[Dot]], u_v: float) -> float:
    """Median least-squares slope (dy/dx) across rows with 3+ dots.

    A perfectly level page gives slope 0; a mildly tilted photograph gives a
    consistent small slope on every row. Rows with fewer than 3 dots carry
    too little signal and are ignored — as are rows whose vertical spread
    exceeds a dot pitch: those are *merged* row clusters (several physical
    dot rows chained together), and a regression through them measures the
    text pattern, not the page tilt.
    """
    slopes: list[float] = []
    for row in rows:
        if len(row) < 3:
            continue
        ys = [d.y for d in row]
        if max(ys) - min(ys) > 0.8 * u_v:
            continue
        xs = [d.x for d in row]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom < 1e-6:
            continue
        slopes.append(sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom)
    return median(slopes) if slopes else 0.0


def _max_row_spread(rows: list[list[Dot]]) -> float:
    return max(max(d.y for d in row) - min(d.y for d in row) for row in rows)


def _line_lifts(
    line_groups: list[list[int]], row_centers: list[float], u_v: float
) -> list[int]:
    """Rows (0-2) each line's topmost cluster sits *below* its true origin.

    Anchoring each line's topmost detected row cluster as physical row 0 fails
    when a whole line uses only the middle/lower rows (e.g. a line of comma
    cells): the line is read one row too high. This locates each line's origin
    on the page's regular line ladder instead, using a reference line that
    provably contains all three physical rows (three clusters spanning ~two dot
    pitches, so its topmost cluster is certainly row 0). A line whose topmost
    cluster falls a whole dot pitch or more below its predicted origin is
    lifted by that many rows.

    Returns all zeros — a no-op — whenever the ladder cannot be trusted (too
    few lines, no regular pitch, or no full-height reference line). Lines that
    already carry a row-0 dot sit on the ladder and get lift 0, so pages
    without single-row lines (clean scans, embossed samples, Grade 1) are
    unaffected.
    """
    lifts = [0] * len(line_groups)
    if len(line_groups) < 3:
        return lifts
    tops = [row_centers[group[0]] for group in line_groups]
    gaps = [b - a for a, b in zip(tops, tops[1:])]
    plausible = [g for g in gaps if 2.4 * u_v <= g <= 8.0 * u_v]
    if not plausible:
        return lifts
    line_pitch = median(plausible)
    if not (3.5 <= line_pitch / u_v <= 6.5):  # standard Braille interline
        return lifts

    reference = None
    for index, group in enumerate(line_groups):
        span = row_centers[group[-1]] - row_centers[group[0]]
        if len(group) >= 3 and span >= 1.6 * u_v:
            reference = index
            break
    if reference is None:
        return lifts

    for index in range(len(line_groups)):
        predicted_origin = tops[reference] + (index - reference) * line_pitch
        lift = round((tops[index] - predicted_origin) / u_v)
        lifts[index] = min(2, max(0, lift))
    return lifts


# Shear-correct when the measured tilt is between ~0.25 and ~10 degrees:
# below that it is measurement noise, above it the page is unusable anyway.
_SKEW_MIN_SLOPE = 0.004
_SKEW_MAX_SLOPE = 0.18


def group_dots(dots: list[Dot]) -> GroupingResult:
    if not dots:
        return GroupingResult(
            flags=[
                make_flag(
                    text="",
                    reason="No Braille dot candidates were detected in the image.",
                    category=CATEGORY_LOW_OCR_CONFIDENCE,
                    severity="high",
                )
            ]
        )

    flags: list[Flag] = []
    r_med = median(d.r for d in dots)
    cluster_threshold = max(2.0, 1.5 * r_med)
    fallback_unit = 3.0 * r_med

    # --- Vertical structure: dot rows, then Braille lines -------------------
    rows, row_centers, u_v = _row_structure(dots, cluster_threshold, fallback_unit)

    # Residual skew correction from the dot geometry itself (Stage 3D-D).
    slope = _estimate_residual_skew(rows, u_v)
    if _SKEW_MIN_SLOPE < abs(slope) <= _SKEW_MAX_SLOPE:
        dots = [replace(d, y=d.y - slope * d.x) for d in dots]
        rows, row_centers, u_v = _row_structure(dots, cluster_threshold, fallback_unit)
        if abs(slope) > 0.05:  # ~3 degrees: correction applied but worth flagging
            flags.append(
                make_flag(
                    text="",
                    reason=(
                        "The page appears noticeably tilted; line order was "
                        "reconstructed after skew correction and should be "
                        "checked."
                    ),
                    category=CATEGORY_LINE_ORDER_UNCERTAINTY,
                    severity="low",
                )
            )

    # Re-cluster rows with a threshold derived from the measured dot pitch
    # when it differs materially from the radius-based first guess (tight or
    # wide dot spacing).
    # (row_threshold diverges from cluster_threshold here: columns keep the
    # radius-based threshold — x jitter is unaffected by row problems.)
    row_threshold = cluster_threshold
    refined_threshold = max(2.0, 0.45 * u_v)
    ratio = refined_threshold / row_threshold
    if ratio < 0.8 or ratio > 1.25:
        rows, row_centers, u_v = _row_structure(dots, refined_threshold, fallback_unit)
        row_threshold = refined_threshold

    # Row-collapse rescue: a single physical dot row is vertically tight —
    # it can never span most of a dot pitch. When jittery dot centres (e.g.
    # crescents from a tightly spaced embossed page) chain neighbouring rows
    # into one cluster, re-cluster with a tighter threshold. A retry that
    # fragments rows down to jitter scale (u_v collapsing towards the dot
    # radius) is rejected: that means the rows are genuinely inseparable.
    collapse_retries = 0
    while (
        _max_row_spread(rows) > 0.8 * u_v
        and collapse_retries < 2
        and row_threshold > 2.0
    ):
        row_threshold = max(2.0, 0.5 * row_threshold)
        retry = _row_structure(dots, row_threshold, fallback_unit)
        collapse_retries += 1
        if retry[2] < 1.8 * r_med:
            break
        rows, row_centers, u_v = retry

    if _max_row_spread(rows) > 0.8 * u_v:
        # Rows remain merged: any dot-to-slot assignment would be a guess.
        # Fail safely — empty result, honest flags — rather than emit
        # confidently-wrong text for a specialist to untangle.
        return GroupingResult(
            flags=dedupe_flags(
                flags
                + [
                    make_flag(
                        text="",
                        reason=(
                            "Braille dot rows could not be separated reliably "
                            "(dots too tightly spaced or too blurred for this "
                            "image resolution); no draft could be produced."
                        ),
                        category=CATEGORY_LINE_ORDER_UNCERTAINTY,
                        severity="high",
                    ),
                    make_flag(
                        text="",
                        reason=(
                            "Try a higher-resolution, flatter photograph with "
                            "the Braille area filling more of the frame."
                        ),
                        category=CATEGORY_LOW_IMAGE_QUALITY,
                        severity="medium",
                    ),
                ]
            )
        )

    if collapse_retries:
        flags.append(
            make_flag(
                text="",
                reason=(
                    "Detected dot rows overlapped and had to be re-clustered "
                    "with a tighter threshold; row and line structure is "
                    "uncertain."
                ),
                category=CATEGORY_LINE_ORDER_UNCERTAINTY,
                severity="medium",
            )
        )

    line_groups: list[list[int]] = [[0]]
    for i in range(1, len(rows)):
        if row_centers[i] - row_centers[i - 1] > 2.4 * u_v:
            line_groups.append([i])
        else:
            line_groups[-1].append(i)

    # --- Assign row indices (0-2) within each line, collect columns ---------
    # A collapse rescue means the row structure was ambiguous: line quality
    # starts below 1 so the final confidence reflects that uncertainty.
    line_quality = 1.0 - 0.15 * collapse_retries
    # Anchor each line to the page line ladder so a line that uses only the
    # middle/lower rows (e.g. all comma cells) is not read one row too high.
    lifts = _line_lifts(line_groups, row_centers, u_v)
    if any(lifts):
        flags.append(
            make_flag(
                text="",
                reason=(
                    "One or more Braille lines use only the lower rows of the "
                    "cell; their row position was inferred from the page line "
                    "spacing and should be checked."
                ),
                category=CATEGORY_UNCLEAR_BRAILLE_CELL,
                severity="low",
            )
        )
    per_line: list[list[tuple[Dot, int]]] = []  # (dot, row_index_in_cell)
    for group_index, group in enumerate(line_groups):
        y0 = row_centers[group[0]]
        lift = lifts[group_index]
        line_dots: list[tuple[Dot, int]] = []
        for row_i in group:
            row_index = int(round((row_centers[row_i] - y0) / u_v)) + lift
            if row_index > 2:
                line_quality = max(0.0, line_quality - 0.2)
                flags.append(
                    make_flag(
                        text="",
                        reason=(
                            "Detected dot rows did not fit the expected 3-row "
                            "Braille line structure; line order may be wrong."
                        ),
                        category=CATEGORY_LINE_ORDER_UNCERTAINTY,
                        severity="medium",
                    )
                )
                row_index = 2
            for dot in rows[row_i]:
                line_dots.append((dot, row_index))
        per_line.append(line_dots)

    # --- Horizontal structure: columns, dot pitch, cell advance -------------
    line_columns: list[tuple[list[list[tuple[Dot, int]]], list[float]]] = []
    all_col_gaps: list[float] = []
    for line_dots in per_line:
        columns = _cluster_1d(line_dots, key=lambda t: t[0].x, threshold=cluster_threshold)
        centers = [sum(t[0].x for t in col) / len(col) for col in columns]
        all_col_gaps.extend(b - a for a, b in zip(centers, centers[1:]))
        line_columns.append((columns, centers))

    # Horizontal dot pitch matches the vertical pitch in standard Braille;
    # prefer measured gaps near u_v, fall back to u_v itself.
    band = [g for g in all_col_gaps if 0.7 * u_v <= g <= 1.4 * u_v]
    u_h = median(band) if band else u_v
    # The cell advance appears in the gap data as two distinct populations:
    # column-2 -> column-1 gaps of ~(advance - u_h), and full-advance gaps
    # where a cell only has one occupied column. Mixing them in one median
    # biases the advance, and even a small bias accumulates into wrong cell
    # assignments by the end of a long line — so normalise each population
    # to "advance" before taking the median.
    advance_estimates = [
        g + u_h for g in all_col_gaps if 1.2 * u_h < g <= 2.0 * u_h
    ] + [g for g in all_col_gaps if 2.0 * u_h < g <= 3.0 * u_h]
    advance = median(advance_estimates) if advance_estimates else 2.5 * u_h

    # --- Grid assignment -----------------------------------------------------
    lines_out: list[list[CellCandidate]] = []
    residual_means: list[float] = []
    uncertain_columns = 0

    for line_number, (columns, centers) in enumerate(line_columns, start=1):
        if not columns:
            continue
        assignment, mean_residual = _fit_line_columns(centers, u_h, advance)
        residual_means.append(mean_residual)

        cell_slots: dict[int, dict[int, Dot]] = {}
        for (cell_idx, col_in_cell, resid), column in zip(assignment, columns):
            if resid / max(u_h, 1e-6) > 0.35:
                uncertain_columns += 1
            for dot, row_index in column:
                dot_number = col_in_cell * 3 + row_index + 1
                slot = cell_slots.setdefault(cell_idx, {})
                if dot_number in slot:
                    uncertain_columns += 1
                    if dot.confidence > slot[dot_number].confidence:
                        slot[dot_number] = dot
                else:
                    slot[dot_number] = dot

        line_cells: list[CellCandidate] = []
        for cell_idx in sorted(cell_slots):
            slot = cell_slots[cell_idx]
            cell_dots = list(slot.values())
            x1 = min(d.x - d.r for d in cell_dots)
            y1 = min(d.y - d.r for d in cell_dots)
            x2 = max(d.x + d.r for d in cell_dots)
            y2 = max(d.y + d.r for d in cell_dots)
            confidence = sum(d.confidence for d in cell_dots) / len(cell_dots)
            line_cells.append(
                CellCandidate(
                    line_number=line_number,
                    grid_index=cell_idx,
                    dots=tuple(sorted(slot)),
                    bbox=(int(x1), int(y1), int(round(x2)), int(round(y2))),
                    confidence=round(confidence, 3),
                )
            )
        lines_out.append(line_cells)

    if uncertain_columns:
        flags.append(
            make_flag(
                text="",
                reason=(
                    f"{uncertain_columns} Braille cell position(s) did not align "
                    "cleanly with the detected cell grid; some cells may be "
                    "misread or merged."
                ),
                category=CATEGORY_UNCLEAR_BRAILLE_CELL,
                severity="medium",
            )
        )

    mean_residual = (
        sum(residual_means) / len(residual_means) if residual_means else 1.0
    )
    quality = max(0.0, min(1.0, 1.0 - 1.5 * mean_residual))
    total_cells = sum(len(line) for line in lines_out)

    return GroupingResult(
        lines=lines_out,
        quality=quality,
        line_quality=line_quality,
        flags=dedupe_flags(flags),
        total_cells=total_cells,
    )
