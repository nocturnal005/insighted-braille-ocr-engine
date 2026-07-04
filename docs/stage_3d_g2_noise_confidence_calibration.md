# Stage 3D-G2: Noise Robustness and Confidence Calibration (commit-safe report)

Targeted OCR improvements for the two evidence-backed weaknesses from the
Stage 3D-G1 controlled Grade 1 baseline: spurious-dot failures under image
noise, and over-optimistic confidence near the dot-size floor. No UKAAF
source text, ground truth, draft output, or images appear in this document.

> OCR output remains **draft-only**. QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. Grade 2 contraction support remains out of scope. Nothing in
> this stage changes the /ocr response contract.

## Pre-change evidence (Stage 3D-G1 baseline, reproduced)

- 10 controlled Grade 1 samples: 8 perfect, mild-noise sample failed safely
  (empty draft, confidence 0.000), low-resolution (~6 px dots) sample kept
  confidence 0.933 despite WER 0.571.
- New diagnostic variants exposed a worse failure class: noise combined
  with shadow / low contrast / low resolution produced high-error drafts
  (CER 0.89-0.97) at confidence 0.67-0.71 - confidently wrong output.

## Root causes

1. **Noise:** CLAHE re-amplifies sensor noise; the adaptive threshold turns
   it into small specks; a handful slip the relative size gate and corrupt
   cell grouping. The specks score visibly lower per-dot confidence
   (circularity + size consistency) than true dots.
2. **Confidence:** no blend factor senses absolute dot scale - small dots
   are still round, evenly spaced, and grid-consistent, so near-floor pages
   kept clean-scan confidence.

A post-threshold morphological opening was trialled first and **rejected
with evidence**: it rounds surviving noise clusters so more pass the
circularity gate, turning safe failures into confidently-wrong drafts.

## Changes made

- `app/ocr/dot_detection.py`: `strict_variant()` - when the size filter
  already rejected extra marks (independent noise evidence), offer an
  additional candidate with per-dot confidence < 0.85 dots removed. The
  existing grid-fit variant selection scores it like any other candidate,
  so it only wins when the filtered dots form a clearly better Braille
  grid. Clean pages are unaffected (nothing to remove). New flags:
  moderate background-noise notice, noise-marks-filtered warning, and a
  dot-size-floor warning (dark mode only).
- `app/ocr/confidence.py`: `dot_size_cap()` - confidence cap from 0.80 at
  3.2 px median dot radius down to 0.50 at 2.0 px (dark path only; emboss
  discs are painted reconstructions whose radius says nothing about capture
  resolution). `noise_ratio_factor()` - accepted/raw candidate ratio below
  0.9 scales confidence down (floor 0.6).
- `app/ocr/pipeline.py`: wires strict candidates into variant selection and
  the new cap/factor into confidence (dark mode only).
- `app/ocr/preprocessing.py`: comment documenting the rejected opening
  approach; no behavioural change.
- 12 new tests in `app/tests/test_noise_calibration.py` (112 total).

## Before / after (controlled Grade 1 dataset, engine 0.4.0)

| Metric | Before | After |
| --- | --- | --- |
| failed | 1 | 0 |
| mean CER | 0.108 | 0.008 |
| median CER | 0.000 | 0.000 |
| mean WER | 0.157 | 0.057 |
| mean confidence | 0.843 | 0.889 |
| repeatability | 1.000 | 1.000 |
| mean processing | 37 ms | 36 ms |

Key samples:

| Sample | Before | After |
| --- | --- | --- |
| mild noise (sigma 8) | safe fail, conf 0.000 | CER 0.000, conf 0.775, noise flags |
| low resolution (~6 px) | CER 0.080 / WER 0.571 @ conf 0.933 | same text, conf 0.615 + floor flag |
| clean / skew / contrast / crops (8) | CER 0.000, conf 0.907-0.950 | byte-identical |

Diagnostic variants (local-only): shadow+noise, low-contrast+noise and
low-res+noise all moved from CER 0.89-0.97 garbage at conf ~0.7 to CER
0.000 at conf 0.71-0.79 with noise flags; heavy noise (sigma 12/16) still
fails safely with confidence 0.000; dot-size ladder is now monotone
(0.950 / 0.950 / 0.662 / 0.528 for ~10/8/6/5 px dots).

Regression checks: original dataset unchanged (CER 0.000, conf 0.950);
embossed dataset CER 0.000 on all 12, mean confidence 0.820 -> 0.812
because one faint-dots sample that decodes via the dark path now carries
the noise-evidence penalty (0.820 -> 0.726) - confidence only ever moved
downward.

## Confidence calibration findings

- The failed->recovered noise sample now sits at 0.775, between the clean
  0.950 and the low-confidence band - proportionate to a page that needed
  spurious marks removed.
- Near-floor pages can no longer report clean-scan confidence: the cap
  binds before the blend, and a low_image_quality flag explains why.
- Confidence is monotone non-increasing across every degradation axis
  tested (noise level, dot size), enforced by tests.

## Limitations

- Validated on rendered controlled samples and synthetic fixtures - not on
  real photographed/scanned school material. Real-world accuracy remains
  unproven until physical captures are evaluated; results apply only to
  the evaluated sample sets.
- The strict retry can in principle discard a genuinely faint dot along
  with noise; it is gated (only with noise evidence, only when it wins the
  grid-fit score) and flagged (`unclear_braille_cell`) so reviewers know
  filtering occurred.
- The ~6 px dot floor itself is unchanged - pages below it now score
  honestly rather than decoding better.
- Grade 2 contraction support remains out of scope for this stage.

## Next recommended stage

Real photographed/embossed captures (the standing operational goal), or
cell-level rawBraille evaluation to let the Grade 2 UKAAF triplets score
the visual pipeline fairly.
