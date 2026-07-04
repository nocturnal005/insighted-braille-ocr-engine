"""Stage 3D-G2 tests: noise robustness and confidence calibration.

Covers: strict-retry recovery on noisy pages (specks filtered, true grid
decoded at reduced confidence with honest flags), safe failure on heavy
noise, dot-size confidence caps near the ~6 px readable floor, monotone
confidence as dots shrink, new flag reasons, preserved behaviour on clean /
skew / low-contrast dark renders, and the unchanged /ocr response shape.
All fixtures are synthetic deterministic renders — never real pupil material
and never local-only UKAAF files.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.evaluation.metrics import character_error_rate, normalise_text
from app.evaluation.sample_generator import image_to_data_url, render_braille_image
from app.models.requests import OcrRequest
from app.ocr.confidence import dot_size_cap, noise_ratio_factor
from app.ocr.dot_detection import DetectionOutcome, Dot, strict_variant
from app.ocr.pipeline import run_ocr
from app.tests.helpers import EXPECTED_RESPONSE_KEYS, make_payload

TEXT = "the cat sat on the mat\nyear 10 physics adds 12"


def scaled_to_dot(image: Image.Image, target_dot_px: float) -> Image.Image:
    """Resize so the ~9 px rendered dots become target_dot_px across."""
    factor = target_dot_px / 9.0
    return image.resize(
        (int(image.width * factor), int(image.height * factor)), Image.LANCZOS
    )


def noised(image: Image.Image, sigma: float, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image, dtype=np.float32)
    arr = arr + rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def low_contrast(image: Image.Image, fg: int = 140, bg: int = 205) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return Image.fromarray((fg + arr * (bg - fg)).astype(np.uint8))


def ocr(image: Image.Image):
    return run_ocr(OcrRequest(**make_payload(dataUrl=image_to_data_url(image))))


def cer_of(response) -> float:
    return character_error_rate(normalise_text(TEXT), normalise_text(response.draftText))


def flag_reasons(response) -> str:
    return " ".join(f.reason.lower() for f in response.flags)


# --- Preserved behaviour ---------------------------------------------------------


def test_clean_dark_sample_still_decodes_high_confidence():
    response = ocr(render_braille_image(TEXT))
    assert cer_of(response) == 0.0
    assert response.confidence >= 0.90


def test_mild_skew_still_decodes():
    image = render_braille_image(TEXT).rotate(
        2, resample=Image.BICUBIC, expand=True, fillcolor=255
    )
    response = ocr(image)
    assert cer_of(response) == 0.0
    assert response.confidence >= 0.90


def test_low_contrast_still_decodes():
    response = ocr(low_contrast(render_braille_image(TEXT)))
    assert cer_of(response) == 0.0
    assert response.confidence >= 0.90


# --- Noise robustness --------------------------------------------------------------


def test_mild_noise_recovers_with_reduced_confidence_and_flags():
    # Before Stage 3D-G2 this failed outright (spurious specks broke cell
    # grouping). The strict retry must now recover the true grid — but the
    # page is noisy, so confidence must be visibly reduced and flagged.
    for seed in (42, 3):
        response = ocr(noised(render_braille_image(TEXT), sigma=8, seed=seed))
        assert cer_of(response) == 0.0, f"seed {seed} did not recover"
        assert 0.55 <= response.confidence <= 0.85
        categories = {f.category for f in response.flags}
        assert "low_image_quality" in categories
        assert "unclear_braille_cell" in categories
        reasons = flag_reasons(response)
        assert "noise" in reasons


def test_heavy_noise_fails_safely_never_confident_garbage():
    response = ocr(noised(render_braille_image(TEXT), sigma=16))
    # Either a safe failure (preferred) or, if it ever decodes, it must be
    # correct or clearly low-confidence — never confidently wrong.
    if response.draftText == "":
        assert response.confidence == 0.0
        assert response.flags
    else:
        assert cer_of(response) == 0.0 or response.confidence < 0.55


# --- Confidence calibration ----------------------------------------------------------


def test_confidence_drops_toward_dot_size_floor():
    base = render_braille_image(TEXT)
    conf = {
        px: ocr(scaled_to_dot(base, px)).confidence for px in (10, 8, 6, 5)
    }
    # Above the floor: high confidence. At/below: visibly reduced, monotone.
    assert conf[10] >= 0.90 and conf[8] >= 0.90
    assert conf[6] < conf[8] - 0.10
    assert conf[5] <= conf[6]
    assert conf[5] < 0.60


def test_near_floor_dot_size_is_flagged():
    base = render_braille_image(TEXT)
    floor_response = ocr(scaled_to_dot(base, 6))
    assert "size floor" in flag_reasons(floor_response)
    clean_response = ocr(scaled_to_dot(base, 10))
    assert "size floor" not in flag_reasons(clean_response)


def test_confidence_never_higher_on_degraded_image():
    base = render_braille_image(TEXT)
    clean_conf = ocr(base).confidence
    for degraded in (
        noised(base, sigma=8),
        scaled_to_dot(base, 6),
        scaled_to_dot(base, 5),
    ):
        assert ocr(degraded).confidence < clean_conf


# --- Contract ---------------------------------------------------------------------


def test_response_contract_unchanged_on_degraded_samples():
    base = render_braille_image(TEXT)
    for image in (noised(base, sigma=8), scaled_to_dot(base, 5)):
        response = ocr(image)
        assert set(response.model_dump().keys()) == EXPECTED_RESPONSE_KEYS


# --- Unit level -----------------------------------------------------------------------


def test_dot_size_cap_boundaries():
    assert dot_size_cap(0.0) is None  # no dots: nothing to cap
    assert dot_size_cap(4.0) is None  # healthy dot size
    assert dot_size_cap(3.2) is None  # floor boundary: cap starts below
    near = dot_size_cap(3.0)
    low = dot_size_cap(2.4)
    bottom = dot_size_cap(1.5)
    assert near is not None and 0.70 <= near <= 0.80
    assert low is not None and low < near
    assert bottom == 0.50  # clamped at the bottom of the range
    # Monotone: smaller dots never get a higher cap.
    radii = [3.1, 2.9, 2.7, 2.5, 2.3, 2.1, 1.9]
    caps = [dot_size_cap(r) for r in radii]
    assert all(a >= b for a, b in zip(caps, caps[1:]))


def test_noise_ratio_factor_boundaries():
    assert noise_ratio_factor(100, 100) == 1.0  # nothing rejected
    assert noise_ratio_factor(95, 100) == 1.0  # within the clean band
    penalised = noise_ratio_factor(70, 100)
    assert 0.6 <= penalised < 1.0
    assert noise_ratio_factor(10, 100) == 0.6  # floored, never zero
    assert noise_ratio_factor(0, 100) == 1.0  # no accepted dots: moot


def _outcome(confidences: list[float], raw: int) -> DetectionOutcome:
    dots = [
        Dot(x=10.0 * i, y=10.0, r=4.0, confidence=c) for i, c in enumerate(confidences)
    ]
    return DetectionOutcome(dots=dots, raw_candidates=raw)


def test_strict_variant_only_offered_with_noise_evidence():
    # raw == accepted: no evidence of rejected marks, no strict candidate.
    assert strict_variant(_outcome([1.0] * 10 + [0.6] * 2, raw=12)) is None
    # Evidence present and low-confidence dots removable: candidate offered.
    strict = strict_variant(_outcome([1.0] * 10 + [0.6] * 2, raw=20))
    assert strict is not None
    assert strict.noise_filtered
    assert len(strict.dots) == 10
    assert strict.raw_candidates == 20
    # Nothing below the threshold: no candidate (nothing to remove).
    assert strict_variant(_outcome([1.0] * 12, raw=20)) is None
    # Too few dots would survive: no candidate.
    assert strict_variant(_outcome([1.0] * 5 + [0.6] * 7, raw=20)) is None
