"""Group detected dots into 6-dot Braille cells on a 2-column x 3-row grid.

The approach fits a regular grid: it estimates the dot pitch (the distance
between adjacent dot positions) vertically and horizontally, the cell
advance (distance between cell origins), then assigns each dot to a
(line, cell, dot-position) slot. Both possible anchor alignments for each
line are tried (the leftmost detected column may be column 1 or column 2 of
its cell, e.g. when a line starts with a capital sign), and the alignment
with the lowest total grid residual wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from app.models.responses import Flag
from app.ocr.dot_detection import Dot
from app.ocr.flags import (
    CATEGORY_LINE_ORDER_UNCERTAINTY,
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

    # --- Vertical structure: dot rows, then Braille lines -------------------
    rows = _cluster_1d(dots, key=lambda d: d.y, threshold=cluster_threshold)
    row_centers = [sum(d.y for d in row) / len(row) for row in rows]
    row_gaps = [b - a for a, b in zip(row_centers, row_centers[1:])]
    u_v = _estimate_unit(row_gaps, fallback=3.0 * r_med)

    line_groups: list[list[int]] = [[0]]
    for i in range(1, len(rows)):
        if row_centers[i] - row_centers[i - 1] > 2.4 * u_v:
            line_groups.append([i])
        else:
            line_groups[-1].append(i)

    # --- Assign row indices (0-2) within each line, collect columns ---------
    line_quality = 1.0
    per_line: list[list[tuple[Dot, int]]] = []  # (dot, row_index_in_cell)
    for group in line_groups:
        y0 = row_centers[group[0]]
        line_dots: list[tuple[Dot, int]] = []
        for row_i in group:
            row_index = int(round((row_centers[row_i] - y0) / u_v))
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
    advance_band = [g for g in all_col_gaps if 1.2 * u_h < g <= 3.0 * u_h]
    advance = (median(advance_band) + u_h) if advance_band else 2.5 * u_h

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
