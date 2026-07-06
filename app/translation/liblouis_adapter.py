"""Optional Liblouis back-translation adapter.

Liblouis is NOT image OCR: it only back-translates already-detected Unicode
Braille into text. The python 'louis' bindings and Liblouis tables are
optional; when unavailable this adapter returns None and the pipeline uses
the built-in Grade 1 fallback translator plus a clear uncertainty flag.
This module must never raise.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

_dll_dir_registered = False


def _ensure_dll_dir() -> None:
    """Register the DLL directory from config once, before ``import louis``."""
    global _dll_dir_registered
    if _dll_dir_registered:
        return
    _dll_dir_registered = True

    from app.core.config import get_settings

    settings = get_settings()
    dll_dir = settings.liblouis_dll_dir
    if dll_dir and Path(dll_dir).is_dir():
        try:
            # add_dll_directory rejects relative paths; the config may use one.
            os.add_dll_directory(str(Path(dll_dir).resolve()))
        except (OSError, AttributeError):
            pass


def _resolve_table(table: str) -> str:
    """Resolve a bare table name to an absolute path when LIBLOUIS_TABLE_PATH
    is configured and the table exists there. Returns the original name when
    no configured path matches (lets Liblouis use its own search)."""
    from app.core.config import get_settings

    table_path = get_settings().liblouis_table_path
    if not table_path:
        return table
    candidate = Path(table_path) / table
    if candidate.is_file():
        return str(candidate.resolve())
    return table


def liblouis_available() -> bool:
    try:
        _ensure_dll_dir()
        import louis  # noqa: F401  # type: ignore

        return True
    except Exception:
        return False


def liblouis_back_translate(unicode_braille: str, table: str) -> str | None:
    """Back-translate Unicode Braille via Liblouis. Returns None when unavailable."""
    try:
        _ensure_dll_dir()
        import louis  # type: ignore
    except Exception:
        return None

    resolved = _resolve_table(table)
    try:
        translated_lines = []
        for line in unicode_braille.split("\n"):
            if not line.strip():
                translated_lines.append("")
                continue
            translated_lines.append(louis.backTranslateString([resolved], line))
        return "\n".join(translated_lines)
    except Exception as exc:
        logger.warning("liblouis_back_translation_failed type=%s", type(exc).__name__)
        return None


def is_grade2_table(table: str) -> bool:
    """Heuristic: does the configured table name indicate Grade 2 / contracted?"""
    lower = table.lower()
    return "g2" in lower or "grade2" in lower or "contracted" in lower
