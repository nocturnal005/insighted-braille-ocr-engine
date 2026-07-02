"""Convert detected dot patterns into Unicode Braille (U+2800 block)."""

from __future__ import annotations

from app.ocr.line_reconstruction import TokenLine

_BRAILLE_BASE = 0x2800


def dots_to_unicode_char(dots: tuple[int, ...] | frozenset[int]) -> str:
    mask = 0
    for dot in dots:
        if 1 <= dot <= 6:
            mask |= 1 << (dot - 1)
    return chr(_BRAILLE_BASE + mask)


def token_lines_to_unicode(token_lines: list[TokenLine]) -> list[str]:
    """Render token lines as Unicode Braille strings (None tokens become spaces)."""
    lines: list[str] = []
    for tokens in token_lines:
        chars = [
            " " if cell is None else dots_to_unicode_char(cell.dots) for cell in tokens
        ]
        lines.append("".join(chars))
    return lines
