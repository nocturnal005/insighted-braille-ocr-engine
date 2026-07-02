"""Built-in Grade 1 (uncontracted) UEB back-translator.

Used when Liblouis is not installed or fails. Handles letters, the capital
sign, the number sign with digits, and a small set of punctuation. Unknown
cells become '?' and are flagged. Grade 2 contractions are NOT interpreted;
a clear uncertainty flag records this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.responses import Flag
from app.ocr.flags import (
    CATEGORY_POSSIBLE_CAPITALISATION_ISSUE,
    CATEGORY_POSSIBLE_CONTRACTION_ISSUE,
    CATEGORY_POSSIBLE_NUMBER_SIGN_ISSUE,
    CATEGORY_UNCLEAR_BRAILLE_CELL,
    dedupe_flags,
    make_flag,
)
from app.translation.braille_maps import (
    CAPITAL_SIGN,
    DIGIT_FROM_LETTER,
    LETTER_FROM_DOTS,
    NUMBER_SIGN,
    PUNCTUATION_FROM_DOTS,
)

_BRAILLE_BASE = 0x2800
_UNKNOWN_PLACEHOLDER = "?"


@dataclass
class TranslationOutcome:
    text: str
    completeness: float  # fraction of cells decoded to a known meaning
    flags: list[Flag] = field(default_factory=list)


def _dots_from_char(char: str) -> frozenset[int] | None:
    code = ord(char)
    if not (_BRAILLE_BASE <= code <= _BRAILLE_BASE + 0xFF):
        return None
    mask = code - _BRAILLE_BASE
    if mask >= 64:  # uses dots 7/8; outside the 6-dot pipeline
        return None
    return frozenset(d for d in range(1, 7) if mask & (1 << (d - 1)))


def back_translate_unicode_lines(lines: list[str]) -> TranslationOutcome:
    output_lines: list[str] = []
    flags: list[Flag] = []
    total_cells = 0
    decoded_cells = 0

    for line in lines:
        buffer: list[str] = []
        capitalise_next = False
        numeric_mode = False

        for char in line:
            if char == " ":
                if capitalise_next:
                    flags.append(
                        make_flag(
                            text="",
                            reason="A capital sign was not followed by a letter.",
                            category=CATEGORY_POSSIBLE_CAPITALISATION_ISSUE,
                            severity="medium",
                        )
                    )
                    capitalise_next = False
                numeric_mode = False
                buffer.append(" ")
                continue

            dots = _dots_from_char(char)
            if dots is None:
                buffer.append(char)
                continue

            total_cells += 1

            if dots == NUMBER_SIGN:
                numeric_mode = True
                decoded_cells += 1
                continue

            if dots == CAPITAL_SIGN:
                if capitalise_next:
                    flags.append(
                        make_flag(
                            text="",
                            reason=(
                                "A double capital sign (word capitals) was detected; "
                                "full word-capitalisation is not yet supported."
                            ),
                            category=CATEGORY_POSSIBLE_CAPITALISATION_ISSUE,
                            severity="low",
                        )
                    )
                capitalise_next = True
                decoded_cells += 1
                continue

            letter = LETTER_FROM_DOTS.get(dots)

            if numeric_mode:
                if letter is not None and letter in DIGIT_FROM_LETTER:
                    buffer.append(DIGIT_FROM_LETTER[letter])
                    decoded_cells += 1
                    continue
                flags.append(
                    make_flag(
                        text=letter or char,
                        reason=(
                            "A number sign was followed by a cell that is not a "
                            "digit; numeric interpretation may be wrong."
                        ),
                        category=CATEGORY_POSSIBLE_NUMBER_SIGN_ISSUE,
                        severity="medium",
                    )
                )
                numeric_mode = False

            if letter is not None:
                buffer.append(letter.upper() if capitalise_next else letter)
                capitalise_next = False
                decoded_cells += 1
                continue

            if capitalise_next:
                flags.append(
                    make_flag(
                        text=char,
                        reason="A capital sign was not followed by a letter.",
                        category=CATEGORY_POSSIBLE_CAPITALISATION_ISSUE,
                        severity="medium",
                    )
                )
                capitalise_next = False

            punctuation = PUNCTUATION_FROM_DOTS.get(dots)
            if punctuation is not None:
                buffer.append(punctuation)
                decoded_cells += 1
                continue

            buffer.append(_UNKNOWN_PLACEHOLDER)
            flags.append(
                make_flag(
                    text=char,
                    reason=(
                        "An unrecognised Braille cell pattern was replaced with "
                        f"'{_UNKNOWN_PLACEHOLDER}'."
                    ),
                    category=CATEGORY_UNCLEAR_BRAILLE_CELL,
                    severity="medium",
                )
            )

        if capitalise_next:
            flags.append(
                make_flag(
                    text="",
                    reason="A capital sign at the end of a line was not followed by a letter.",
                    category=CATEGORY_POSSIBLE_CAPITALISATION_ISSUE,
                    severity="medium",
                )
            )
        output_lines.append("".join(buffer))

    if decoded_cells:
        flags.append(
            make_flag(
                text="",
                reason=(
                    "Back-translation used the built-in Grade 1 (uncontracted) UEB "
                    "table. Grade 2 contractions, if present, were not interpreted."
                ),
                category=CATEGORY_POSSIBLE_CONTRACTION_ISSUE,
                severity="low",
            )
        )

    completeness = decoded_cells / total_cells if total_cells else 0.0
    return TranslationOutcome(
        text="\n".join(output_lines),
        completeness=completeness,
        flags=dedupe_flags(flags),
    )
