"""Optional Liblouis back-translation adapter.

Liblouis is NOT image OCR: it only back-translates already-detected Unicode
Braille into text. The python 'louis' bindings and Liblouis tables are
optional; when unavailable this adapter returns None and the pipeline uses
the built-in Grade 1 fallback translator plus a clear uncertainty flag.
This module must never raise.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


def liblouis_available() -> bool:
    try:
        import louis  # noqa: F401  # type: ignore

        return True
    except Exception:
        return False


def liblouis_back_translate(unicode_braille: str, table: str) -> str | None:
    """Back-translate Unicode Braille via Liblouis. Returns None when unavailable."""
    try:
        import louis  # type: ignore
    except Exception:
        return None

    try:
        translated_lines = []
        for line in unicode_braille.split("\n"):
            if not line.strip():
                translated_lines.append("")
                continue
            translated_lines.append(louis.backTranslateString([table], line))
        return "\n".join(translated_lines)
    except Exception as exc:  # table missing, binding error, etc.
        logger.warning("liblouis_back_translation_failed type=%s", type(exc).__name__)
        return None
