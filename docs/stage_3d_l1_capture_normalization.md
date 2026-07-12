# Stage 3D-L1 — Capture normalisation for real phone photos

## Why

The 2026-07-12 real-capture diagnostic (5 self-provided worksheet photos with
teacher transcriptions, run preview-only from outside the repo) showed the
pipeline produced **nothing at all** on real phone captures. Root causes, in
failure order:

1. **Raw phone files rejected at intake.** Many phones emit MPO
   (Multi-Picture Object) JPEGs — a standard JPEG with extra embedded frames.
   Pillow reports the format as `MPO`, and the decoder only accepted
   `PNG`/`JPEG`, so genuine staff photos failed before OCR began.
2. **EXIF orientation ignored.** Phones record physical orientation in EXIF
   instead of rotating pixels; the pipeline never applied it.
3. **No scale normalisation.** Dot detection assumes dots ~6-14 px across
   (preprocessing.py); a 4000-6000 px phone photo renders dots at ~30-40 px,
   flooding detection with texture noise (≈12,000 "dots" per page) and
   collapsing row separation.
4. **No orientation recovery.** Upside-down/sideways pages never decode. A
   180-degree flip is especially dangerous: the Braille lattice is half-turn
   symmetric, so a flipped page still forms a plausible grid and silently
   decodes to garbage.

## What changed

- `app/ocr/image_decode.py`
  - MPO accepted alongside PNG/JPEG (primary frame used).
  - `ImageOps.exif_transpose` applied on open (fail-safe fallback).
- `app/ocr/capture_normalization.py` (new)
  - `normalise_scale`: captures whose long side exceeds 1600 px are
    downscaled to 1400 px (INTER_AREA). Inputs at or below the threshold are
    returned identically — calibrated/synthetic inputs are untouched.
  - `detect_with_normalisation`: wraps preprocessing + variant selection.
    - Upright decode that forms a plausible, dot-accounting cell set is
      returned as-is (zero extra cost on the happy path).
    - **Upside-down disambiguation**: when the upright decode's quick
      Grade 1 readability (fallback-translator completeness, string-only)
      is below 0.5, the 180-degree flip is also decoded and must beat the
      upright readability by a clear margin to win.
    - **Rescue ladder** (only when the upright attempt forms no cells, too
      few cells to be a page, or cells that cannot account for the detected
      dots): retries 90/180/270 rotations, then the same four orientations
      on a bright-region page crop. Attempts are ranked by
      `cells x grid quality x readability`; the base attempt is the
      incumbent, so a genuine small decode survives. Bounded at 7 extra
      detection passes, paid only by images that previously returned
      nothing.
  - Every applied step emits an honest flag (downscaled / rotated /
    cropped); rotated results carry a line-order caution because cell
    bboxes are reported in the rotated frame.
- `app/evaluation/diagnostic_probe.py`: staged replay routes through the
  same normalisation path as `/ocr`; `capture_rescaled`,
  `capture_rotation_applied`, `capture_cropped` added to the safe dict.
- `app/tests/test_capture_normalization.py`: 14 tests — scale/crop unit
  behaviour, all rotations at 1x and 4x end-to-end, upright decode
  unaffected and unflagged, small genuine decode survives the ladder,
  blank image still fails safely.

## Measured effect (local preview diagnostics, not accuracy claims)

Raw, unmodified files from the external sample folder:

| capture | before | after |
| ------- | ------ | ----- |
| all 9 files | rejected at decode (MPO) | decode + detection at sane dot scale |
| sample5 (sideways) | L0 | **L4** via rotation rescue, 236 cells |
| annotated pages 1/2 (upside-down) | L0 | **L4** via 180 rescue, ~225 cells |
| annotated page 4 | L0 | **L4**, 267 cells |
| samples 1-4, page 3 (fabric background) | L0 | L1 — dots detected, rows still unseparated |

Synthetic: 8/8 orientation x scale combinations decode exactly; full test
suite green (245 passed).

## Honest limits

- Word-level accuracy on the real captures is still ~0-3%: where structure
  decodes, per-cell dot classification is mostly wrong. That is the
  fine-tuned-detector workstream (see `tools/finetune/`), not more
  geometry heuristics.
- The busy-fabric background still defeats row separation for 5 of 9
  captures; the bright-region crop is too crude for a page lying on a
  bright textured surface.
- A rescued decode is still an unverified draft. QTVI / Braille-literate
  specialist verification remains mandatory. This engine never claims
  certified Braille accuracy.
