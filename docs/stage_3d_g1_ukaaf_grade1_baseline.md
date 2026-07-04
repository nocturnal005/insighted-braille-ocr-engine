# Stage 3D-G1: UKAAF Controlled Grade 1 Baseline (commit-safe report)

First controlled UK/UEB Grade 1 OCR baseline using UKAAF-derived material.
This document contains summary metrics only - no UKAAF source text, no
ground truth, no OCR draft output, no images, and no links that expose
copyrighted content.

> OCR output remains **draft-only**. QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. Nothing in this stage changes that.

## Purpose

Stage 3D-G0 found that the nine UKAAF UEB sample triplets are contracted
(Grade 2) braille, which the engine does not back-translate. The one
Grade 1 source in the pack is the uncontracted UEB Quick Reference BRF.
This stage generated a controlled Grade 1 sample set from that BRF,
created deterministic ground truth, and ran the first controlled UK/UEB
Grade 1 baseline through the existing `real_anonymised` harness.

## Source and method (no content extracts)

- Source: the UKAAF UEB Quick Reference Guide, uncontracted BRF edition
  (ASCII braille, 28 cells x 27 lines format). Confirmed uncontracted: its
  prose lines decode letter-by-letter to readable English.
- Scope: the document body is a symbols reference; most lines contain UEB
  symbol cells outside the engine's Grade 1 decode set (letters, digits,
  capital/number signs, four punctuation marks). Only the in-scope prose
  sections (three header lines, two closing lines) were used. This keeps
  ground truth deterministic - fabricating readings for out-of-scope
  symbols is not acceptable.
- Images: rendered directly from the BRF's own cells (standard NABCC
  mapping) as Braille-only black-dot images, then transformed per
  condition. These are synthetic controlled renders, honestly recorded in
  metadata as generated material - NOT real school photographs.
- Ground truth: produced by an independent NABCC -> Grade 1 decode and
  cross-checked against the engine's fallback translator on the same
  cells; both agreed exactly on all 10 samples (generation aborts on any
  disagreement). Method recorded in the local source-verification note.
- All generated images, ground truth, metadata, and reports are local-only
  and gitignored.

## Samples generated (10)

| Condition | Samples |
| --- | --- |
| Clean, good light (2 sections) | 001, 002 |
| Mild skew (2 deg) / moderate skew (5 deg) | 003, 004 |
| Low contrast (grey-on-grey) | 005 |
| Shadowed (illumination gradient) | 006 |
| Low resolution (~6px dots, at the floor) | 007 |
| Extra-margin crop / close crop | 008, 009 |
| Mild Gaussian noise (sigma 8) | 010 |

## Audit result

10/10 samples evaluable; no unsafe names, no missing ground truth or
metadata, no blocking warnings (advisory "very bright" notes are expected
for sparse black-on-white renders).

## Baseline result (2026-07-04, engine 0.3.0, fallback Grade 1 translator)

- evaluated=10, skipped=0, failed=1
- mean CER 0.108 | median CER 0.000 | mean WER 0.157 | median WER 0.000
- mean confidence 0.843 | repeatability 1.000 | mean processing 37 ms

Per-condition summary:

| Condition | Result |
| --- | --- |
| Clean (both sections) | CER 0.000, confidence 0.950 |
| Mild / moderate skew | CER 0.000, confidence 0.924 / 0.907 |
| Low contrast | CER 0.000, confidence 0.916 |
| Shadowed | CER 0.000, confidence 0.950 |
| Extra margin / close crop | CER 0.000, confidence 0.950 |
| Low resolution (~6px dots) | CER 0.080, WER 0.571, confidence 0.933 |
| Mild noise (sigma 8) | FAILED SAFELY: empty draft, confidence 0.000, low_image_quality + line_order flags |

## Answers to the stage questions

1. **Can the engine process controlled UK/UEB Grade 1 images?** Yes -
   8/10 conditions decode perfectly, including both skews, low contrast,
   shadow, and both crop styles.
2. **Baseline CER/WER:** mean CER 0.108 / median 0.000 (the mean is driven
   entirely by the two stress conditions).
3. **Condition sensitivity:** geometry and illumination transforms were
   handled cleanly; the two real weaknesses are dot size at the ~6px
   resolution floor (mild character errors, WER 0.571) and pixel noise
   (detection picks up spurious dots and the pipeline aborts).
4. **Confidence honesty:** the failed sample correctly scored 0.000, and
   skew/contrast conditions scored slightly lower than clean. One
   calibration observation: the low-resolution sample kept confidence
   0.933 despite WER 0.571 - confidence does not yet feel the
   near-floor degradation. Worth watching in 3D-G2.
5. **Failure modes:** noise-induced spurious dot detections leading to a
   safe abort (empty draft + high-severity flags) - the designed
   fail-safe, preferable to confidently wrong text.
6. **Stage 3D-G2 candidates:** noise-robust dot filtering; confidence
   sensitivity to near-floor dot sizes; optional cell-level (rawBraille)
   evaluation so the Grade 2 UKAAF triplets can fairly score the visual
   pipeline; optional Liblouis Grade 2 support.

## Limitations (read before quoting numbers)

- These are **rendered, controlled samples** - not real school
  photographs, not embossed paper, not phone captures. Real-world
  performance remains unmeasured until photographed/scanned samples are
  evaluated. The results apply only to the evaluated sample set.
- Only in-scope prose sections of one UKAAF document were used (about
  120 characters of distinct text across two sections). This is a small,
  controlled baseline, not a representative corpus.
- The nine UKAAF UEB sample triplets remain Grade 2 and are excluded from
  Grade 1 CER/WER (see the local Grade 2 gap note). They can validate dot
  detection and rawBraille later, and full decoding only if contraction
  support (e.g. Liblouis Grade 2) is added.

## Next recommended stage

Stage 3D-G2: address the two measured weaknesses (noise robustness,
near-floor confidence calibration), and/or add cell-level rawBraille
evaluation to unlock fair visual-pipeline scoring on the Grade 2 UKAAF
triplets. Real photographed/embossed captures remain the standing
operational goal for genuine real-world evidence.
