"""Stage 3D-K2 tests: real-capture row/cell grouping robustness.

K1 found that genuine physical Braille photos with many detected dots could
still stall at L1 (dots detected, no cell grid) because single-linkage row
clustering chains adjacent rows on dense/curved/skewed real captures. K2 adds
a lattice-projection fallback (``_recover_rows_by_lattice``) that recovers the
regular row structure when clustering fails — gated so it (a) never fires on
pages that already group successfully, and (b) rejects random noise instead
of hallucinating a grid.

All fixtures are synthetic: hand-built dot lattices and deterministic renders.
Never real pupil material, never the local-only UKAAF files, never the user's
real sample folder.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from app.evaluation.sample_generator import (
    EmbossedStyle,
    image_to_data_url,
    render_braille_image,
    render_embossed_braille_image,
)
import app.ocr.pipeline as pipeline_module
from app.ocr.cell_grouping import (
    _LATTICE_MIN_SPACING_REGULARITY,
    _estimate_vertical_pitch,
    _recover_rows_by_lattice,
    group_dots,
)
from app.ocr.confidence import EMBOSS_MODE_CAP, LATTICE_RECOVERY_CAP
from app.ocr.dot_detection import Dot, spacing_regularity
from app.models.requests import OcrRequest
from app.ocr.pipeline import run_ocr
from app.tests.helpers import EXPECTED_RESPONSE_KEYS, make_payload


def _lattice_dots(n_lines=3, cells=5, u=10.0, r=3.0, jitter=0.0, seed=0):
    """A regular Braille-like dot lattice: 3 rows/line, 4u inter-line pitch."""
    rng = np.random.default_rng(seed)
    dots: list[Dot] = []
    for line in range(n_lines):
        y_base = line * 4 * u
        for cell in range(cells):
            x_base = cell * 2.5 * u
            for col in range(2):
                for row in range(3):
                    jx = float(rng.normal(0, jitter)) if jitter else 0.0
                    jy = float(rng.normal(0, jitter)) if jitter else 0.0
                    dots.append(
                        Dot(x=x_base + col * u + jx, y=y_base + row * u + jy,
                            r=r, confidence=1.0)
                    )
    return dots


# --- Lattice pitch estimation -------------------------------------------------


def test_estimate_vertical_pitch_recovers_true_pitch():
    dots = _lattice_dots(u=10.0, r=3.0)
    pitch = _estimate_vertical_pitch(dots, r_med=3.0)
    assert pitch is not None
    assert abs(pitch - 10.0) < 1.0  # within a pixel of the true pitch


def test_estimate_vertical_pitch_none_for_too_few_dots():
    assert _estimate_vertical_pitch(_lattice_dots()[:6], r_med=3.0) is None


# --- Lattice row recovery gate ------------------------------------------------


def test_recover_rows_accepts_regular_lattice():
    dots = _lattice_dots(n_lines=3, cells=5, u=10.0, r=3.0)
    result = _recover_rows_by_lattice(dots, r_med=3.0)
    assert result is not None
    rows, centers, u_v = result
    assert len(rows) == 9  # 3 lines x 3 rows
    assert abs(u_v - 10.0) < 1.0
    assert centers == sorted(centers)  # top to bottom


def test_recover_rows_tolerates_jitter():
    dots = _lattice_dots(jitter=1.0, seed=3)
    assert spacing_regularity(dots) >= _LATTICE_MIN_SPACING_REGULARITY
    assert _recover_rows_by_lattice(dots, r_med=3.0) is not None


def test_recover_rows_rejects_random_noise():
    # Random points must NOT be recovered into a grid: a low residual ratio
    # alone cannot tell noise from structure (random points sit ~0.25 pitch
    # from any lattice), so the spacing-regularity gate is what protects us.
    rng = np.random.default_rng(11)
    noise = [
        Dot(x=float(rng.uniform(0, 400)), y=float(rng.uniform(0, 300)),
            r=3.0, confidence=1.0)
        for _ in range(120)
    ]
    assert spacing_regularity(noise) < _LATTICE_MIN_SPACING_REGULARITY
    assert _recover_rows_by_lattice(noise, r_med=3.0) is None


def test_recover_rows_rejects_too_few_dots():
    assert _recover_rows_by_lattice(_lattice_dots()[:6], r_med=3.0) is None


# --- group_dots end to end ----------------------------------------------------


def test_group_dots_noise_yields_no_cells():
    # Dot-rich random noise through the full grouping path must produce no
    # cells (safe failure), never a hallucinated grid.
    rng = np.random.default_rng(5)
    noise = [
        Dot(x=float(rng.uniform(0, 500)), y=float(rng.uniform(0, 400)),
            r=3.0, confidence=1.0)
        for _ in range(200)
    ]
    result = group_dots(noise)
    assert result.total_cells == 0
    assert result.lines == []
    assert result.flags  # honest flags, not silence


def test_group_dots_clean_lattice_still_groups():
    # A clean lattice groups via the normal path (no fallback needed) and the
    # fallback must not disturb it.
    result = group_dots(_lattice_dots(n_lines=2, cells=6, u=12.0, r=3.5))
    assert result.total_cells > 0
    assert result.lines


# --- Pipeline-level robustness (synthetic renders) ----------------------------


def test_controlled_render_unaffected():
    # The clean bundled-style render still decodes cleanly; K2 changes only the
    # failure path, so controlled behaviour is unchanged.
    response = run_ocr(OcrRequest(**make_payload("hello world")))
    assert response.rawBraille
    assert response.draftText
    assert response.confidence > 0.5


def test_blurred_synthetic_no_crash_and_flagged():
    img = render_braille_image("reading by touch")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    response = run_ocr(OcrRequest(**make_payload(dataUrl=image_to_data_url(blurred))))
    # Either recovers or fails safely — but always a valid contract, flags,
    # and never a crash or an overconfident result.
    assert set(response.model_dump()) == EXPECTED_RESPONSE_KEYS
    assert response.flags
    assert 0.0 <= response.confidence <= 0.95


def test_skewed_synthetic_still_decodes():
    img = render_braille_image("skew tolerance test")
    skewed = img.rotate(-3.0, resample=Image.BICUBIC, expand=True, fillcolor=255)
    response = run_ocr(OcrRequest(**make_payload(dataUrl=image_to_data_url(skewed))))
    assert set(response.model_dump()) == EXPECTED_RESPONSE_KEYS
    # A mild skew should still yield a draft (skew correction + grouping).
    assert response.rawBraille


def test_tight_dense_render_no_crash_and_not_overconfident():
    # A tightly-spaced embossed render must not crash and must never be
    # overconfident, whether it groups normally or via the lattice fallback.
    # (The recovery-specific cap is pinned deterministically below and in
    # test_embossed; this is the render-level smoke test.)
    style = EmbossedStyle(unit=9, dot_radius=3)
    response = run_ocr(
        OcrRequest(**make_payload(
            dataUrl=image_to_data_url(render_embossed_braille_image("dense braille", style))
        ))
    )
    assert set(response.model_dump()) == EXPECTED_RESPONSE_KEYS
    if response.draftText:
        assert response.confidence <= EMBOSS_MODE_CAP
        assert response.flags


def test_lattice_recovery_confidence_hard_capped_even_with_perfect_columns(monkeypatch):
    # The core honesty guarantee: a lattice-recovered page must never read as
    # confident even if its columns fit the cell grid perfectly. line_quality
    # carries little weight, so the cap must come from the pipeline-level
    # recovery cap — not from line_quality alone. Force a clean, normally-
    # grouping page to look recovered with perfect grid-fit quality and assert
    # the cap still holds.
    real_select = pipeline_module._select_variant

    def fake_select(variants):
        detection, grouping = real_select(variants)
        if grouping.total_cells > 0:
            grouping.recovered_via_fallback = True
            grouping.quality = 1.0       # pretend the columns fit perfectly
            grouping.line_quality = 1.0  # and the line order is certain
        return detection, grouping

    monkeypatch.setattr(pipeline_module, "_select_variant", fake_select)
    response = run_ocr(OcrRequest(**make_payload("hello world")))
    assert response.confidence <= LATTICE_RECOVERY_CAP


def test_uniform_grid_texture_never_reads_as_confident():
    # A regular non-Braille texture (uniform dot grid) scores high on
    # nearest-neighbour spacing regularity, so it can pass the lattice gate.
    # That is acceptable ONLY because it can never read as confident: a
    # uniform grid does not fit Braille's paired-column cell advance, and any
    # recovered page is hard-capped. Assert it is never confident.
    canvas = np.full((300, 500), 255, dtype=np.uint8)
    for y in range(20, 290, 12):
        for x in range(20, 490, 12):
            canvas[y - 2:y + 2, x - 2:x + 2] = 0
    response = run_ocr(
        OcrRequest(**make_payload(dataUrl=image_to_data_url(Image.fromarray(canvas, "L"))))
    )
    assert set(response.model_dump()) == EXPECTED_RESPONSE_KEYS
    # Never confident: either no draft, or a clearly low-confidence flagged one.
    assert response.confidence < 0.55
    assert response.flags


def test_recovery_cap_below_emboss_cap():
    # A last-ditch lattice recovery is less trustworthy than a clean emboss
    # read, so its cap must be stricter.
    assert LATTICE_RECOVERY_CAP < EMBOSS_MODE_CAP


def test_ocr_contract_unchanged_after_k2():
    response = run_ocr(OcrRequest(**make_payload()))
    assert set(response.model_dump().keys()) == EXPECTED_RESPONSE_KEYS
