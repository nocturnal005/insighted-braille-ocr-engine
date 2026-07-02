"""Synthetic Braille sample generation.

Renders Grade 1 UEB text as ideal black-dot-on-white images with standard
cell geometry. These are best-case inputs used for tests, the samples/
folder, and the evaluation harness ground truth. Real embossed-paper
photographs are much harder — see limitations.md.

Usage:
    python -m app.evaluation.sample_generator [--out-dir samples]

Writes:
    samples/images/*.png          synthetic Braille page images
    samples/ground_truth/*.txt    matching expected text
    samples/sample_request.json   contract-shaped example request
    samples/sample_response.json  actual engine response for that request
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from app.translation.braille_maps import (
    CAPITAL_SIGN,
    DIGIT_TO_LETTER,
    LETTER_TO_DOTS,
    NUMBER_SIGN,
    PUNCTUATION_TO_DOTS,
)

# Standard-ish Braille geometry, scaled to pixels: dot pitch (unit), cell
# advance 2.5 units, line pitch 5 units.
UNIT = 12
DOT_RADIUS = 4
CELL_ADVANCE = int(2.5 * UNIT)
LINE_PITCH = 5 * UNIT
MARGIN = 48

# Sample texts are synthetic and school-flavoured. Never use real pupil
# names or identifiable data in sample files.
SAMPLES: list[tuple[str, str]] = [
    ("sample_01_hello_world", "hello world"),
    ("sample_02_science", "the cell membrane"),
    ("sample_03_numbers", "add 12 and 34"),
    ("sample_04_capitals", "Year 10 Physics"),
    ("sample_05_multiline", "light travels\nin straight lines"),
]


def text_line_to_cells(line: str) -> list[frozenset[int] | None]:
    """Convert one line of text into dot patterns (None = blank space cell)."""
    cells: list[frozenset[int] | None] = []
    numeric_mode = False
    for char in line:
        if char == " ":
            cells.append(None)
            numeric_mode = False
            continue
        if char.isdigit():
            if not numeric_mode:
                cells.append(NUMBER_SIGN)
                numeric_mode = True
            cells.append(LETTER_TO_DOTS[DIGIT_TO_LETTER[char]])
            continue
        numeric_mode = False
        if char.isalpha():
            lower = char.lower()
            if lower not in LETTER_TO_DOTS:
                raise ValueError(f"Unsupported letter for sample generation: {char!r}")
            if char.isupper():
                cells.append(CAPITAL_SIGN)
            cells.append(LETTER_TO_DOTS[lower])
            continue
        if char in PUNCTUATION_TO_DOTS:
            cells.append(PUNCTUATION_TO_DOTS[char])
            continue
        raise ValueError(f"Unsupported character for sample generation: {char!r}")
    return cells


def render_braille_image(text: str) -> Image.Image:
    """Render text as a synthetic Braille image (black dots on white)."""
    line_cells = [text_line_to_cells(line) for line in text.split("\n")]
    max_cells = max(len(cells) for cells in line_cells)
    width = 2 * MARGIN + max(1, max_cells) * CELL_ADVANCE
    height = 2 * MARGIN + (len(line_cells) - 1) * LINE_PITCH + 2 * UNIT + 2 * DOT_RADIUS

    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)

    for line_index, cells in enumerate(line_cells):
        for cell_index, dots in enumerate(cells):
            if dots is None:
                continue
            origin_x = MARGIN + cell_index * CELL_ADVANCE
            origin_y = MARGIN + line_index * LINE_PITCH
            for dot in sorted(dots):
                column = 0 if dot <= 3 else 1
                row = (dot - 1) % 3
                cx = origin_x + column * UNIT
                cy = origin_y + row * UNIT
                draw.ellipse(
                    [cx - DOT_RADIUS, cy - DOT_RADIUS, cx + DOT_RADIUS, cy + DOT_RADIUS],
                    fill=0,
                )
    return image


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_contract_samples(out_dir: Path) -> None:
    """Write a real request/response pair produced by the actual pipeline."""
    from app.models.requests import OcrRequest
    from app.ocr.pipeline import run_ocr

    image = render_braille_image(SAMPLES[0][1])
    request = OcrRequest(
        taskId="task-demo-001",
        title="Braille homework page 1",
        fileName="sample_01_hello_world.png",
        mimeType="image/png",
        dataUrl=image_to_data_url(image),
        subject="English",
        yearGroup="Year 9",
    )
    response = run_ocr(request)

    (out_dir / "sample_request.json").write_text(
        json.dumps(request.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "sample_response.json").write_text(
        json.dumps(response.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="samples", help="Output directory")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    truth_dir = out_dir / "ground_truth"
    images_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    for name, text in SAMPLES:
        render_braille_image(text).save(images_dir / f"{name}.png")
        (truth_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"wrote {name}")

    _write_contract_samples(out_dir)
    print("wrote sample_request.json and sample_response.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
