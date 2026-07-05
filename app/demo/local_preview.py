"""Local preview CLI: run the OCR pipeline on one image and show the draft.

Usage:
    python -m app.demo.local_preview samples\\images\\sample_01_hello_world.png
    python -m app.demo.local_preview path\\to\\photo.jpg --json

Runs the unchanged OCR pipeline in-process (no server needed) on a single
PNG/JPEG file and prints a human-readable report: the draft-only banner,
confidence, uncertainty flags, the detected Braille cells (rawBraille), and
the draft back-translation. `--json` prints the full /ocr contract response
as JSON instead (the banner then goes to stderr so stdout stays valid JSON).

This tool intentionally displays draft text and Braille content on the local
terminal — that is the point of a human preview. It never writes files, and
its output must not be committed, logged, or treated as evidence of
accuracy. Only use material allowed by the collection protocols (see
docs/stage_3d_g6_real_capture_collection_protocol.md sections 1-2): never
real pupil work or identifying material.
"""

from __future__ import annotations

import argparse
import base64
import sys
import uuid
from pathlib import Path
from time import perf_counter

from app.core.config import DRAFT_ONLY_WARNING
from app.models.requests import OcrRequest
from app.models.responses import OcrResponse
from app.ocr.pipeline import run_ocr

MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

_BANNER_RULE = "=" * 72


def _ensure_utf8_output() -> None:
    """Braille output is U+2800-block text; legacy Windows console encodings
    cannot represent it. Reconfigure to UTF-8 with replacement so the preview
    degrades to '?' instead of crashing when the console cannot render it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _banner() -> str:
    return "\n".join(
        [
            _BANNER_RULE,
            "DRAFT-ONLY BRAILLE OCR PREVIEW — NOT VERIFIED OUTPUT",
            DRAFT_ONLY_WARNING,
            _BANNER_RULE,
        ]
    )


def _build_request(image_path: Path, mime: str) -> OcrRequest:
    data_url = (
        f"data:{mime};base64,"
        + base64.b64encode(image_path.read_bytes()).decode("ascii")
    )
    return OcrRequest(
        taskId="preview-" + uuid.uuid4().hex[:12],
        title="Local preview",
        fileName=image_path.name,
        mimeType=mime,
        dataUrl=data_url,
    )


def _format_report(image_path: Path, response: OcrResponse, duration_ms: int) -> str:
    lines: list[str] = [_banner(), ""]
    lines.append(f"Image:      {image_path.name} ({image_path.stat().st_size:,} bytes)")
    lines.append(f"Request id: {response.providerRequestId}")
    lines.append(f"Duration:   {duration_ms} ms")
    cell_count = len(response.rawCells)
    line_count = len((response.rawBraille or "").splitlines())
    lines.append(f"Detected:   {cell_count} cells across {line_count} lines")
    lines.append(
        f"Confidence: {response.confidence:.3f}  "
        "(internal heuristic, not a calibrated probability)"
    )

    lines.append("")
    lines.append(f"Uncertainty flags ({len(response.flags)}):")
    if response.flags:
        for flag in response.flags:
            where = f" [{flag.text}]" if flag.text else ""
            lines.append(f"  - ({flag.severity}) {flag.category}{where}: {flag.reason}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Detected Braille cells (rawBraille):")
    if response.rawBraille:
        for braille_line in response.rawBraille.splitlines():
            lines.append(f"  {braille_line}")
    else:
        lines.append("  (none — the page could not be read; see flags above)")

    lines.append("")
    lines.append("Draft back-translation (UNVERIFIED DRAFT):")
    if response.draftText:
        for text_line in response.draftText.splitlines():
            lines.append(f"  {text_line}")
    else:
        lines.append("  (empty — no draft was produced; see flags above)")

    lines.append("")
    lines.append(
        "Reminder: this draft must be verified by a QTVI or Braille-literate "
        "specialist before any use in teacher feedback or export."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_output()
    parser = argparse.ArgumentParser(
        prog="python -m app.demo.local_preview",
        description=(
            "Preview the draft-only OCR result for one local Braille image. "
            "Output is an unverified draft requiring specialist verification."
        ),
    )
    parser.add_argument("image", help="Path to a PNG or JPEG Braille image")
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the full /ocr contract response as JSON instead of the "
            "human-readable report (banner goes to stderr)"
        ),
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"image file not found: {image_path}")
    mime = MIME_BY_EXTENSION.get(image_path.suffix.lower())
    if mime is None:
        supported = ", ".join(sorted(MIME_BY_EXTENSION))
        parser.error(
            f"unsupported file extension '{image_path.suffix}' "
            f"(supported: {supported})"
        )

    request = _build_request(image_path, mime)
    started = perf_counter()
    response = run_ocr(request)
    duration_ms = int((perf_counter() - started) * 1000)

    if args.json:
        print(_banner(), file=sys.stderr)
        print(response.model_dump_json(indent=2))
    else:
        print(_format_report(image_path, response, duration_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
