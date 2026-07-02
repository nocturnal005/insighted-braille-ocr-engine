"""Order cells into lines and insert word spacing from grid gaps.

A gap of one empty grid cell between detected cells is a normal Braille word
space. Larger runs of blank cells are kept (capped) and flagged as word
spacing uncertainty.
"""

from __future__ import annotations

from app.models.responses import Flag, RawCell
from app.ocr.cell_grouping import CellCandidate, GroupingResult
from app.ocr.flags import CATEGORY_WORD_SPACING_UNCERTAINTY, dedupe_flags, make_flag

# Token lines: each entry is a CellCandidate or None (None = one space).
TokenLine = list["CellCandidate | None"]

_MAX_CONSECUTIVE_SPACES = 5


def reconstruct_lines(
    grouping: GroupingResult,
) -> tuple[list[TokenLine], list[RawCell], list[Flag]]:
    token_lines: list[TokenLine] = []
    raw_cells: list[RawCell] = []
    flags: list[Flag] = []

    for line in grouping.lines:
        tokens: TokenLine = []
        previous_index: int | None = None
        for cell in sorted(line, key=lambda c: c.grid_index):
            if previous_index is not None:
                gap = cell.grid_index - previous_index - 1
                if gap >= 3:
                    flags.append(
                        make_flag(
                            text="",
                            reason=(
                                f"A run of {gap} blank cells was detected on line "
                                f"{cell.line_number}; word spacing may be uncertain."
                            ),
                            category=CATEGORY_WORD_SPACING_UNCERTAINTY,
                            severity="low",
                        )
                    )
                tokens.extend([None] * min(gap, _MAX_CONSECUTIVE_SPACES))
            tokens.append(cell)
            previous_index = cell.grid_index
            raw_cells.append(
                RawCell(
                    line=cell.line_number,
                    cellIndex=cell.grid_index + 1,
                    dots=list(cell.dots),
                    bbox=list(cell.bbox),
                    confidence=cell.confidence,
                )
            )
        token_lines.append(tokens)

    return token_lines, raw_cells, dedupe_flags(flags)
