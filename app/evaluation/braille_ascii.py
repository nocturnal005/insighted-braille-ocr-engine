"""Braille ASCII (BRF transport encoding) codec for cell-level validation.

A BRF (Braille Ready Format) file stores one Braille cell per byte using the
standard 64-character *Braille ASCII* transport code (also called North
American Braille ASCII / the ISO 11548-1 mapping). This is NOT a language
translation: each printable byte in ``0x20``-``0x5F`` maps to exactly one
6-dot cell, and the space byte maps to the empty cell. Decoding a BRF byte to
its cell therefore reads the *visual* Braille of the page - the same thing the
OCR pipeline reconstructs from an image - without interpreting any Grade 1 or
Grade 2 contraction meaning.

Stage 3D-G3 uses this to build expected ``rawBraille`` from UKAAF Grade 2 BRF
files so the visual pipeline (dot detection, cell grouping, line
reconstruction, rawBraille) can be scored at the cell level. English text is
never produced here; Grade 2 contraction support is out of scope.

The table is validated by ``verify_table()`` (and the test-suite): it must be a
bijection over all 64 possible cells and must agree with the engine's
independently-authored Grade 1 letter map and sign constants.
"""

from __future__ import annotations

from app.ocr.braille_decode import dots_to_unicode_char
from app.translation.braille_maps import (
    CAPITAL_SIGN,
    LETTER_TO_DOTS,
    NUMBER_SIGN,
)

# Canonical Braille ASCII table: printable byte -> dots (1-6). Uppercase
# letters are used, as in BRF. The space byte maps to the empty cell and is
# handled separately (it is not a dot pattern). Digits map to the "lowered"
# cells (a-j shifted down one row), matching the Braille ASCII standard - they
# are the transport encoding, not UEB numbers (UEB numbers are number-sign +
# letters, e.g. "#DB").
_BRF_DOTS: dict[str, tuple[int, ...]] = {
    "!": (2, 3, 4, 6),
    '"': (5,),
    "#": (3, 4, 5, 6),
    "$": (1, 2, 4, 6),
    "%": (1, 4, 6),
    "&": (1, 2, 3, 4, 6),
    "'": (3,),
    "(": (1, 2, 3, 5, 6),
    ")": (2, 3, 4, 5, 6),
    "*": (1, 6),
    "+": (3, 4, 6),
    ",": (6,),
    "-": (3, 6),
    ".": (4, 6),
    "/": (3, 4),
    "0": (3, 5, 6),
    "1": (2,),
    "2": (2, 3),
    "3": (2, 5),
    "4": (2, 5, 6),
    "5": (2, 6),
    "6": (2, 3, 5),
    "7": (2, 3, 5, 6),
    "8": (2, 3, 6),
    "9": (3, 5),
    ":": (1, 5, 6),
    ";": (5, 6),
    "<": (1, 2, 6),
    "=": (1, 2, 3, 4, 5, 6),
    ">": (3, 4, 5),
    "?": (1, 4, 5, 6),
    "@": (4,),
    "A": (1,),
    "B": (1, 2),
    "C": (1, 4),
    "D": (1, 4, 5),
    "E": (1, 5),
    "F": (1, 2, 4),
    "G": (1, 2, 4, 5),
    "H": (1, 2, 5),
    "I": (2, 4),
    "J": (2, 4, 5),
    "K": (1, 3),
    "L": (1, 2, 3),
    "M": (1, 3, 4),
    "N": (1, 3, 4, 5),
    "O": (1, 3, 5),
    "P": (1, 2, 3, 4),
    "Q": (1, 2, 3, 4, 5),
    "R": (1, 2, 3, 5),
    "S": (2, 3, 4),
    "T": (2, 3, 4, 5),
    "U": (1, 3, 6),
    "V": (1, 2, 3, 6),
    "W": (2, 4, 5, 6),
    "X": (1, 3, 4, 6),
    "Y": (1, 3, 4, 5, 6),
    "Z": (1, 3, 5, 6),
    "[": (2, 4, 6),
    "\\": (1, 2, 5, 6),
    "]": (1, 2, 4, 5, 6),
    "^": (4, 5),
    "_": (4, 5, 6),
}

SPACE = " "

# Frozenset view of the table (space -> empty cell included for completeness).
BRF_CHAR_TO_DOTS: dict[str, frozenset[int]] = {
    char: frozenset(dots) for char, dots in _BRF_DOTS.items()
}
BRF_CHAR_TO_DOTS[SPACE] = frozenset()


def verify_table() -> None:
    """Assert the Braille ASCII table is internally and engine-consistent.

    Raises ``AssertionError`` if any invariant fails - callers (generation
    scripts, tests) run this before trusting the table.
    """
    # 1. Bijection over all 64 possible cells (dots drawn from {1..6}).
    all_cells = {frozenset(BRF_CHAR_TO_DOTS[c]) for c in BRF_CHAR_TO_DOTS}
    assert len(BRF_CHAR_TO_DOTS) == 64, f"expected 64 chars, got {len(BRF_CHAR_TO_DOTS)}"
    assert len(all_cells) == 64, "Braille ASCII table is not a bijection (duplicate cells)"
    for dots in all_cells:
        assert dots <= {1, 2, 3, 4, 5, 6}, f"cell {sorted(dots)} uses invalid dot numbers"

    # 2. Letters A-Z must match the engine's independently-authored map.
    for letter, dots in LETTER_TO_DOTS.items():
        assert BRF_CHAR_TO_DOTS[letter.upper()] == dots, (
            f"letter {letter!r} disagrees with engine LETTER_TO_DOTS"
        )

    # 3. Sign cells must match engine constants (number sign, capital sign).
    assert BRF_CHAR_TO_DOTS["#"] == NUMBER_SIGN, "'#' cell != engine NUMBER_SIGN"
    assert BRF_CHAR_TO_DOTS[","] == CAPITAL_SIGN, "',' cell != engine CAPITAL_SIGN"


def brf_char_to_dots(char: str) -> frozenset[int]:
    """Dots for one BRF byte. Unknown bytes raise ``KeyError`` (never guessed)."""
    return BRF_CHAR_TO_DOTS[char]


def brf_line_to_cells(line: str) -> list[frozenset[int] | None]:
    """One BRF line -> cells; the space byte becomes ``None`` (a blank cell)."""
    cells: list[frozenset[int] | None] = []
    for char in line:
        if char == SPACE:
            cells.append(None)
        else:
            cells.append(brf_char_to_dots(char))
    return cells


def brf_line_to_unicode(line: str) -> str:
    """One BRF line -> Unicode Braille string (space byte -> space char)."""
    return "".join(
        SPACE if char == SPACE else dots_to_unicode_char(brf_char_to_dots(char))
        for char in line
    )


# --------------------------------------------------------------------------
# BRF normalisation for controlled cell-level validation (Stage 3D-G3)
#
# The engine's rawBraille never carries leading indentation or trailing
# spaces and caps long blank runs (see line_reconstruction). Expected
# rawBraille is normalised to the same conventions so a low error rate
# reflects the dot reading, not spacing artefacts. Every rule below is
# documented; nothing is translated, invented, or silently dropped.
# --------------------------------------------------------------------------

FORM_FEED = "\x0c"
# Mirrors line_reconstruction._MAX_CONSECUTIVE_SPACES: the pipeline caps a run
# of blank cells at this many spaces, so expected output caps to match.
MAX_SPACE_RUN = 5


def normalise_brf_text(text: str) -> str:
    """Normalise line endings only: CRLF / CR -> LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def brf_pages(text: str) -> list[str]:
    """Split a BRF into embossable pages on the form-feed byte."""
    return normalise_brf_text(text).split(FORM_FEED)


def brf_first_page_lines(text: str) -> list[str]:
    """Lines of the first embossable page, trailing spaces and outer blank
    lines removed. One page is the natural, standard-sized render unit."""
    page = brf_pages(text)[0]
    lines = [line.rstrip() for line in page.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def content_lines(lines: list[str]) -> list[str]:
    """Keep only non-blank lines, each stripped of leading/trailing spaces.

    Leading indentation is stripped because the pipeline does not represent
    leading whitespace; internal word spacing is preserved. Blank separator
    lines are dropped because the pipeline emits a line only where cells are
    detected. This keeps rendered rows and expected lines aligned 1:1.
    """
    return [line.strip() for line in lines if line.strip()]


def _cap_space_runs(text: str) -> str:
    """Collapse any run of more than MAX_SPACE_RUN spaces down to the cap."""
    out: list[str] = []
    run = 0
    for char in text:
        if char == SPACE:
            run += 1
            if run <= MAX_SPACE_RUN:
                out.append(char)
        else:
            run = 0
            out.append(char)
    return "".join(out)


def expected_rawbraille(lines: list[str]) -> str:
    """Build expected rawBraille from content lines (Unicode Braille, capped
    blank runs, lines joined by newline). No English is produced."""
    return "\n".join(_cap_space_runs(brf_line_to_unicode(line)) for line in lines)
