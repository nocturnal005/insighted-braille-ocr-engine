"""Synthetic Braille sample generation.

Renders Grade 1 UEB text two ways:

* Ideal black-dot-on-white images with standard cell geometry — the best
  case used for tests, the samples/images folder, and ground truth.
* Embossed-paper-style images (Stage 3D-D): raised dots simulated as a
  height map shaded by directional light, so each dot appears as a
  highlight/shadow pair on a paper-toned background — with configurable
  contrast, noise, uneven illumination, skew, spacing, and dot size. These
  imitate photographs/scans of real embossed Braille without using any
  real pupil material. All texts are synthetic and school-flavoured.

Usage:
    python -m app.evaluation.sample_generator [--out-dir samples]

Writes:
    samples/images/*.png                 ideal synthetic Braille pages
    samples/ground_truth/*.txt           matching expected text
    samples/embossed_images/*.png        embossed-photo-style pages
    samples/embossed_ground_truth/*.txt  matching expected text
    samples/sample_request.json          contract-shaped example request
    samples/sample_response.json         actual engine response for that request
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
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


# --------------------------------------------------------------------------
# Embossed-paper-style rendering (Stage 3D-D)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbossedStyle:
    """Controls for the embossed-photo simulation.

    relief        shading strength: how strongly the raised dot bends light
                  (low relief = faint, low-contrast dots)
    shadow_gain   extra multiplier on the shadow side only (photographs under
                  harsh light show deeper shadows than highlights)
    paper_level   base paper grey level (0-255)
    noise_sigma   Gaussian paper-texture noise
    illumination  peak brightness fall-off across the page (uneven lighting)
    rotation_deg  whole-page rotation (camera skew)
    unit          dot pitch in pixels (spacing variation)
    dot_radius    dot bump radius in pixels (dot size variation)
    blur_sigma    softness of the relief (focus/emboss sharpness)
    margin        page margin in pixels
    """

    relief: float = 22.0
    shadow_gain: float = 1.0
    paper_level: int = 205
    noise_sigma: float = 2.5
    illumination: float = 0.0
    rotation_deg: float = 0.0
    unit: int = UNIT
    dot_radius: int = DOT_RADIUS
    blur_sigma: float = 1.1
    margin: int = MARGIN


def render_embossed_braille_image(
    text: str, style: EmbossedStyle = EmbossedStyle(), seed: int = 0
) -> Image.Image:
    """Render text as an embossed-paper-style photograph simulation.

    A height map of raised dots is shaded by directional light from the
    top-left, so every dot becomes a highlight/shadow crescent pair — the
    signature of real embossed Braille photographs — instead of a dark
    printed disc. Noise, uneven illumination, and rotation are then applied.
    The result is grayscale, like the decoded input the pipeline sees.
    """
    line_cells = [text_line_to_cells(line) for line in text.split("\n")]
    max_cells = max(len(cells) for cells in line_cells)
    unit, radius, margin = style.unit, style.dot_radius, style.margin
    cell_advance = int(2.5 * unit)
    line_pitch = 5 * unit
    width = 2 * margin + max(1, max_cells) * cell_advance
    height = 2 * margin + (len(line_cells) - 1) * line_pitch + 2 * unit + 2 * radius

    # Height map: each raised dot is a soft circular bump.
    bumps = np.zeros((height, width), dtype=np.float32)
    for line_index, cells in enumerate(line_cells):
        for cell_index, dots in enumerate(cells):
            if dots is None:
                continue
            origin_x = margin + cell_index * cell_advance
            origin_y = margin + line_index * line_pitch
            for dot in sorted(dots):
                column = 0 if dot <= 3 else 1
                row = (dot - 1) % 3
                cx = origin_x + column * unit
                cy = origin_y + row * unit
                cv2.circle(bumps, (cx, cy), radius, 1.0, thickness=-1)
    bumps = cv2.GaussianBlur(bumps, (0, 0), sigmaX=max(style.blur_sigma, 0.3))

    # Directional shading: light from the top-left. The slope of the bump
    # towards the light brightens; away from it darkens — producing the
    # highlight/shadow pair around every dot.
    grad_y, grad_x = np.gradient(bumps)
    light_x, light_y = -0.707, -0.707
    shading = style.relief * 8.0 * (grad_x * light_x + grad_y * light_y)
    shading = np.where(shading < 0, shading * style.shadow_gain, shading)

    page = np.full((height, width), float(style.paper_level), dtype=np.float32)
    page += shading

    if style.illumination > 0:
        # Smooth diagonal brightness fall-off, as if lit from one corner.
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        page -= style.illumination * (0.6 * ys + 0.4 * xs)

    if style.noise_sigma > 0:
        rng = np.random.default_rng(seed)
        page += rng.normal(0.0, style.noise_sigma, size=page.shape).astype(np.float32)

    image = np.clip(page, 0, 255).astype(np.uint8)

    if abs(style.rotation_deg) > 1e-3:
        h, w = image.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), style.rotation_deg, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=int(style.paper_level),
        )

    return Image.fromarray(image, mode="L")


# Embossed sample set. Grade 1 UEB only (Grade 2 contractions are NOT
# supported — do not add contracted samples until they are). Texts are
# synthetic; never real pupil work or identifiable data.
EMBOSSED_SAMPLES: list[tuple[str, str, EmbossedStyle]] = [
    ("embossed_01_clean", "the cat sat on the mat", EmbossedStyle()),
    ("embossed_02_low_contrast", "reading by touch", EmbossedStyle(relief=10.0)),
    ("embossed_03_mild_shadow", "shadows on paper", EmbossedStyle(relief=26.0, shadow_gain=1.8)),
    ("embossed_04_mild_skew", "keep the page flat", EmbossedStyle(rotation_deg=2.0)),
    ("embossed_05_noisy_paper", "rough paper texture", EmbossedStyle(noise_sigma=7.0)),
    ("embossed_06_uneven_light", "light from one side", EmbossedStyle(illumination=45.0)),
    ("embossed_07_numbers", "maths test 42", EmbossedStyle()),
    ("embossed_08_multiline", "braille is read\nby touch not sight", EmbossedStyle()),
    ("embossed_09_wide_spacing", "wide dot spacing", EmbossedStyle(unit=16)),
    # unit 10 is tighter than the standard 12 but still resolvable; unit 9
    # sits below the resolution floor and is exercised as a controlled
    # failure in the tests instead of shipping as a sample.
    ("embossed_10_tight_spacing", "tight dot spacing", EmbossedStyle(unit=10, dot_radius=3)),
    ("embossed_11_faint_dots", "faint braille dots", EmbossedStyle(relief=7.0, blur_sigma=1.5)),
    ("embossed_12_rotated_margin", "check the margins", EmbossedStyle(rotation_deg=3.0, margin=80)),
]


def _sample_seed(name: str) -> int:
    """Stable per-sample noise seed so generated files are reproducible."""
    return zlib.crc32(name.encode("utf-8"))


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

    embossed_images_dir = out_dir / "embossed_images"
    embossed_truth_dir = out_dir / "embossed_ground_truth"
    embossed_images_dir.mkdir(parents=True, exist_ok=True)
    embossed_truth_dir.mkdir(parents=True, exist_ok=True)

    for name, text, style in EMBOSSED_SAMPLES:
        image = render_embossed_braille_image(text, style, seed=_sample_seed(name))
        image.save(embossed_images_dir / f"{name}.png")
        (embossed_truth_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"wrote {name}")

    _write_contract_samples(out_dir)
    print("wrote sample_request.json and sample_response.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
