"""Cell-level metrics comparing OCR ``rawBraille`` to expected Braille cells.

Stage 3D-G3. These metrics score the *visual* pipeline (dot detection, cell
grouping, line reconstruction, rawBraille) against Braille cells decoded from a
BRF transport encoding - deliberately NOT against English text. No Grade 1 or
Grade 2 back-translation is involved and English CER/WER is never computed
here, so a Grade 2 page (whose contractions the engine does not interpret) can
still be scored fairly on whether the dots were read correctly.

``rawBraille`` strings are Unicode Braille (U+2800 block) with lines joined by
``"\\n"`` and single/blank cells rendered as spaces.
"""

from __future__ import annotations

from app.evaluation.metrics import character_error_rate, levenshtein_distance

_SPACE = " "
_NEWLINE = "\n"


def _cells(text: str) -> list[str]:
    """Flatten to the sequence of cell glyphs, ignoring spaces and line breaks."""
    return [char for char in text if char not in (_SPACE, _NEWLINE)]


def _content_lines(text: str) -> list[str]:
    """Split into lines, dropping trailing blank lines (never leading content)."""
    lines = text.split(_NEWLINE)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def rawbraille_cer(expected: str, predicted: str) -> float:
    """Character error rate over the raw Braille strings (space-sensitive)."""
    return character_error_rate(expected, predicted)


def cell_error_rate(expected: str, predicted: str) -> float:
    """Edit distance over the cell sequence, ignoring spaces and line breaks.

    This isolates whether the *dots* were read correctly, independent of word
    spacing and line-break reconstruction.
    """
    reference = _cells(expected)
    hypothesis = _cells(predicted)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def line_metrics(expected: str, predicted: str) -> dict:
    """Line-count and per-line exact-match statistics for line reconstruction."""
    expected_lines = _content_lines(expected)
    predicted_lines = _content_lines(predicted)
    compared = min(len(expected_lines), len(predicted_lines))
    exact = sum(
        1 for i in range(compared) if expected_lines[i] == predicted_lines[i]
    )
    denominator = max(len(expected_lines), 1)
    return {
        "expected_lines": len(expected_lines),
        "predicted_lines": len(predicted_lines),
        "line_count_mismatch": abs(len(expected_lines) - len(predicted_lines)),
        "exact_line_matches": exact,
        "line_reconstruction_accuracy": exact / denominator,
    }


def cell_counts(expected: str, predicted: str) -> tuple[int, int]:
    """(expected cell count, predicted cell count), ignoring spaces/newlines."""
    return len(_cells(expected)), len(_cells(predicted))


def sample_metrics(expected: str, predicted: str) -> dict:
    """All cell-level metrics for one sample. Never returns any Braille text."""
    expected_cells, predicted_cells = cell_counts(expected, predicted)
    lines = line_metrics(expected, predicted)
    raw_cer = rawbraille_cer(expected, predicted)
    cer = cell_error_rate(expected, predicted)
    return {
        "rawbraille_cer": raw_cer,
        "cell_error_rate": cer,
        "expected_cells": expected_cells,
        "predicted_cells": predicted_cells,
        "cell_count_mismatch": abs(expected_cells - predicted_cells),
        "exact_sample_match": raw_cer == 0.0,
        **lines,
    }
