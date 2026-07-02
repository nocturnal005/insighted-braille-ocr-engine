"""Shared Grade 1 (uncontracted) UEB dot-pattern maps.

Used by the fallback back-translator, the Unicode decoder, and the synthetic
sample generator, so forward and reverse mappings can never drift apart.
"""

from __future__ import annotations

LETTER_TO_DOTS: dict[str, frozenset[int]] = {
    "a": frozenset({1}),
    "b": frozenset({1, 2}),
    "c": frozenset({1, 4}),
    "d": frozenset({1, 4, 5}),
    "e": frozenset({1, 5}),
    "f": frozenset({1, 2, 4}),
    "g": frozenset({1, 2, 4, 5}),
    "h": frozenset({1, 2, 5}),
    "i": frozenset({2, 4}),
    "j": frozenset({2, 4, 5}),
    "k": frozenset({1, 3}),
    "l": frozenset({1, 2, 3}),
    "m": frozenset({1, 3, 4}),
    "n": frozenset({1, 3, 4, 5}),
    "o": frozenset({1, 3, 5}),
    "p": frozenset({1, 2, 3, 4}),
    "q": frozenset({1, 2, 3, 4, 5}),
    "r": frozenset({1, 2, 3, 5}),
    "s": frozenset({2, 3, 4}),
    "t": frozenset({2, 3, 4, 5}),
    "u": frozenset({1, 3, 6}),
    "v": frozenset({1, 2, 3, 6}),
    "w": frozenset({2, 4, 5, 6}),
    "x": frozenset({1, 3, 4, 6}),
    "y": frozenset({1, 3, 4, 5, 6}),
    "z": frozenset({1, 3, 5, 6}),
}

PUNCTUATION_TO_DOTS: dict[str, frozenset[int]] = {
    ",": frozenset({2}),
    ".": frozenset({2, 5, 6}),
    "'": frozenset({3}),
    "-": frozenset({3, 6}),
}

CAPITAL_SIGN: frozenset[int] = frozenset({6})
NUMBER_SIGN: frozenset[int] = frozenset({3, 4, 5, 6})

DIGIT_TO_LETTER: dict[str, str] = {
    "1": "a",
    "2": "b",
    "3": "c",
    "4": "d",
    "5": "e",
    "6": "f",
    "7": "g",
    "8": "h",
    "9": "i",
    "0": "j",
}

LETTER_FROM_DOTS: dict[frozenset[int], str] = {
    dots: letter for letter, dots in LETTER_TO_DOTS.items()
}
PUNCTUATION_FROM_DOTS: dict[frozenset[int], str] = {
    dots: char for char, dots in PUNCTUATION_TO_DOTS.items()
}
DIGIT_FROM_LETTER: dict[str, str] = {
    letter: digit for digit, letter in DIGIT_TO_LETTER.items()
}
