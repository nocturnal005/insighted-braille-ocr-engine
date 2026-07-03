"""Stage 3D-D tests: embossed-paper-style Braille photograph handling.

Covers the embossed sample generator, the emboss preprocessing variant,
variant selection, end-to-end decoding under adverse conditions (low
contrast, skew, noise, uneven light), honest confidence caps, controlled
failure on unusable images, unchanged response shape, and log hygiene.
All images are synthetic — never real pupil material.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from PIL import Image

from app.evaluation.metrics import normalise_text
from app.evaluation.sample_generator import (
    EMBOSSED_SAMPLES,
    EmbossedStyle,
    image_to_data_url,
    render_braille_image,
    render_embossed_braille_image,
)
from app.models.requests import OcrRequest
from app.ocr.confidence import EMBOSS_MODE_CAP, FALLBACK_TRANSLATION_CAP
from app.ocr.image_decode import decode_data_url
from app.ocr.pipeline import _select_variant, run_ocr
from app.ocr.preprocessing import MODE_DARK, MODE_EMBOSS, preprocess
from app.tests.helpers import EXPECTED_RESPONSE_KEYS, make_payload

STYLE_BY_NAME = {name: (text, style) for name, text, style in EMBOSSED_SAMPLES}


def embossed_payload(text: str, style: EmbossedStyle, seed: int = 7, **overrides) -> dict:
    image = render_embossed_braille_image(text, style, seed=seed)
    return make_payload(dataUrl=image_to_data_url(image), **overrides)


def gray_of(image: Image.Image) -> np.ndarray:
    data_url = image_to_data_url(image)
    gray, _ = decode_data_url(
        data_url, "image/png", max_bytes=10_000_000, max_pixels=40_000_000
    )
    return gray


# --- Sample generation -------------------------------------------------------


def test_embossed_render_produces_highlight_and_shadow_relief():
    image = render_embossed_braille_image("hello", EmbossedStyle(), seed=1)
    pixels = np.asarray(image)
    paper = EmbossedStyle().paper_level
    assert image.mode == "L"
    # Raised dots must appear as both brighter AND darker regions than paper.
    assert pixels.max() > paper + 15
    assert pixels.min() < paper - 15


def test_embossed_sample_set_is_grade1_and_synthetic():
    assert len(EMBOSSED_SAMPLES) >= 12
    names = [name for name, _, _ in EMBOSSED_SAMPLES]
    assert len(set(names)) == len(names)
    for _, text, _ in EMBOSSED_SAMPLES:
        # Grade 1 renderable (raises ValueError on unsupported/contracted input)
        render_embossed_braille_image(text, EmbossedStyle(), seed=0)


def test_sample_generator_writes_embossed_pairs(tmp_path):
    from app.evaluation.sample_generator import main

    assert main(["--out-dir", str(tmp_path)]) == 0
    images = sorted(p.name for p in (tmp_path / "embossed_images").glob("*.png"))
    truths = sorted(p.stem for p in (tmp_path / "embossed_ground_truth").glob("*.txt"))
    assert len(images) >= 12
    assert [i.removesuffix(".png") for i in images] == truths


# --- Preprocessing -----------------------------------------------------------


def test_emboss_variant_present_for_embossed_image():
    gray = gray_of(render_embossed_braille_image("hello world", EmbossedStyle(), seed=2))
    variants = preprocess(gray).variants
    modes = {v.mode for v in variants}
    assert MODE_DARK in modes and MODE_EMBOSS in modes
    emboss = next(v for v in variants if v.mode == MODE_EMBOSS)
    assert (emboss.binary > 0).any()


def test_clean_synthetic_image_has_no_emboss_variant():
    # Printed dark dots have no highlight side, so pairing finds nothing and
    # the dark path keeps its original single-variant behaviour.
    gray = gray_of(render_braille_image("hello world"))
    assert {v.mode for v in preprocess(gray).variants} == {MODE_DARK}


def test_emboss_variant_handles_low_contrast():
    _, style = STYLE_BY_NAME["embossed_02_low_contrast"]
    gray = gray_of(render_embossed_braille_image("reading by touch", style, seed=3))
    emboss = [v for v in preprocess(gray).variants if v.mode == MODE_EMBOSS]
    assert emboss and (emboss[0].binary > 0).any()


# --- Detection and variant selection -----------------------------------------


def test_selection_picks_emboss_mode_on_embossed_image():
    gray = gray_of(
        render_embossed_braille_image("the cat sat on the mat", EmbossedStyle(), seed=4)
    )
    detection, grouping = _select_variant(preprocess(gray).variants)
    assert detection.mode == MODE_EMBOSS
    assert len(detection.dots) >= 20
    assert grouping.total_cells >= 10
    # Every dot candidate preserves geometry diagnostics.
    for dot in detection.dots:
        assert dot.area > 0
        x1, y1, x2, y2 = dot.bbox
        assert x2 > x1 and y2 > y1


def test_selection_keeps_dark_mode_on_clean_synthetic():
    gray = gray_of(render_braille_image("hello world"))
    detection, _ = _select_variant(preprocess(gray).variants)
    assert detection.mode == MODE_DARK


# --- End-to-end decoding -----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "embossed_01_clean",
        "embossed_02_low_contrast",
        "embossed_04_mild_skew",
        "embossed_05_noisy_paper",
        "embossed_06_uneven_light",
        "embossed_08_multiline",
    ],
)
def test_embossed_round_trip(name):
    text, style = STYLE_BY_NAME[name]
    response = run_ocr(OcrRequest(**embossed_payload(text, style)))
    assert normalise_text(response.draftText) == normalise_text(text)
    assert 0.0 < response.confidence <= EMBOSS_MODE_CAP
    categories = {flag.category for flag in response.flags}
    assert "low_image_quality" in categories  # embossed-photo mode is flagged


def test_confidence_caps_are_honest():
    # Embossed runs never exceed the emboss cap...
    text, style = STYLE_BY_NAME["embossed_01_clean"]
    embossed = run_ocr(OcrRequest(**embossed_payload(text, style)))
    assert embossed.confidence <= EMBOSS_MODE_CAP
    # ...and even a perfect clean scan stays below 1.0 while the fallback
    # (non-Liblouis) translator is in use.
    clean = run_ocr(OcrRequest(**make_payload("hello world")))
    assert clean.confidence <= FALLBACK_TRANSLATION_CAP


# --- Controlled failure ------------------------------------------------------


def test_unresolvable_tight_spacing_fails_safely():
    # unit 9 with radius-3 dots is below the resolution floor: dot rows
    # cannot be separated. The engine must return an empty draft with an
    # honest high-severity flag — never confidently-wrong text.
    style = EmbossedStyle(unit=9, dot_radius=3)
    response = run_ocr(OcrRequest(**embossed_payload("tight dot spacing", style)))
    assert response.draftText == ""
    assert response.confidence == 0.0
    assert any(flag.severity == "high" for flag in response.flags)


def test_noise_only_image_fails_safely():
    rng = np.random.default_rng(11)
    noise = rng.normal(190, 12, size=(240, 480)).clip(0, 255).astype(np.uint8)
    payload = make_payload(dataUrl=image_to_data_url(Image.fromarray(noise, mode="L")))
    response = run_ocr(OcrRequest(**payload))
    assert response.draftText == "" or response.confidence < 0.5
    assert response.flags


# --- Contract and hygiene ----------------------------------------------------


def test_embossed_response_shape_unchanged():
    text, style = STYLE_BY_NAME["embossed_01_clean"]
    response = run_ocr(OcrRequest(**embossed_payload(text, style)))
    dumped = response.model_dump()
    assert set(dumped) == EXPECTED_RESPONSE_KEYS
    assert dumped["rawCells"], "embossed runs must still return rawCells"
    for cell in dumped["rawCells"]:
        assert set(cell) == {"line", "cellIndex", "dots", "bbox", "confidence"}


def test_embossed_run_logs_no_sensitive_content(caplog):
    text, style = STYLE_BY_NAME["embossed_01_clean"]
    payload = embossed_payload(
        text,
        style,
        taskId="task-SENSITIVE-ID-42",
        title="Pupil Jane Doe homework",
    )
    with caplog.at_level(logging.INFO):
        response = run_ocr(OcrRequest(**payload))
    assert response.draftText  # the run succeeded
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "task-SENSITIVE-ID-42" not in logged  # raw taskId never logged
    assert "Jane Doe" not in logged  # titles never logged
    assert normalise_text(text) not in logged  # transcription text never logged
    assert "base64" not in logged and "data:image" not in logged
