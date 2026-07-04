# Stage 3D-G3: UKAAF Grade 2 rawBraille / cell-level validation (commit-safe)

Validates whether the visual OCR pipeline can read UKAAF Grade 2 Braille
**cells** — dot detection, cell grouping, line reconstruction, and
`rawBraille` — even though it cannot yet decode Grade 2 contractions into
English. No UKAAF source text, BRF content, rawBraille output, or generated
images appear in this document.

> OCR output remains **draft-only**. QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. **This stage does not add Grade 2 contraction support and makes no
> English Grade 2 transcription-accuracy claim.** The `/ocr` response contract
> is unchanged.

## Why Grade 2 is evaluated only at the cell level

The nine UKAAF UEB sample triplets are contracted **Grade 2** Braille; the
engine back-translates **Grade 1** only. Decoding a Grade 2 page with Grade 1
rules produces systematically wrong English, so English CER/WER would measure a
scoped-out capability, not the vision quality. But a Braille **cell** is the
same regardless of grade, so the visual pipeline can be scored fairly by
comparing its `rawBraille` to the cells of the source BRF — no English needed.

## Ground truth: BRF cells, not English

A BRF file stores one Braille cell per byte using the standard 64-character
Braille-ASCII transport code (a byte→cell mapping, **not** a language
translation). `app/evaluation/braille_ascii.py` decodes each byte to its cell
and to Unicode Braille. The table is validated (`verify_table()` and tests) to:

- be a **bijection** over all 64 possible cells;
- **agree with the engine's** independently-authored Grade 1 letter map,
  number sign, and capital sign.

Expected `rawBraille` is built from the **first page** of each BRF (the natural
standard-sized render unit) with documented normalisation only: CRLF→LF; page
split on form feed; outer blank lines trimmed; blank separator lines dropped and
leading indentation stripped (the pipeline does not represent leading
whitespace); long blank runs capped to match the pipeline. Nothing is
translated, invented, or silently dropped. The text PDFs are **not** used as
rawBraille ground truth.

## Samples

Nine categories, one controlled first-page render each — prose, recipe, press
release, jokes, product instructions, French vocabulary, simple maths,
intermediate maths, computer code — plus four controlled variants (mild skew,
low contrast, mild noise, extra-margin crop) for three representative samples
(prose, simple maths, computer code). 21 samples total. All UKAAF-derived
material, generated images, expected files, and reports are **local-only and
gitignored**.

## rawBraille evaluation method

`app/evaluation/run_rawbraille_evaluation.py` (dataset `ukaaf_grade2_raw`) runs
the pipeline and compares `rawBraille` to the expected cells. Metrics:
cell error rate (space-agnostic, the primary dot-reading measure), rawBraille
CER (space-sensitive), line-count mismatch, cell-count mismatch, exact-sample
match, line-reconstruction accuracy, confidence, flags, processing time, and
rawBraille repeatability. **English CER/WER is never computed.** The runner and
its `--write-report` JSON emit metrics and safe labels only — never expected or
predicted rawBraille, draft text, image data, or unsafe file names.

## Results (engine 0.4.0)

| Metric | Value |
| --- | --- |
| samples evaluated / failed | 21 / 0 |
| mean cell error rate | 0.010 |
| median cell error rate | 0.000 |
| mean rawBraille CER | 0.008 |
| exact sample match rate | 0.571 |
| line-count mismatch rate | 0.000 |
| cell-count mismatch rate | 0.000 |
| mean line-reconstruction accuracy | 0.982 |
| mean confidence | 0.883 |
| rawBraille repeatable across runs | yes (all) |

### Line and cell reconstruction

Line count and cell count matched the source on **every** sample and every
degradation — no rows split, merged, or dropped. All residual error is cell
**substitution**, not insertion/deletion.

### Which sample types are hardest

Prose-style samples with frequent comma/apostrophe punctuation are the hardest
(~2–3% cell error); maths, code, and the press release decode perfectly. A
dot-pattern diagnostic showed every substitution is a middle/lower-row dot read
one row too high (the lone comma cell dominates): a cell whose dots sit only in
the middle/lower rows, with no top-row dot, can be anchored one row too high
relative to the line grid. The synthetic, embossed, and Grade 1 sample sets
contain no such lone low-row cells, so this latent row-anchoring limitation is
newly exposed by richer Grade 2 prose.

### Confidence / flags

Confidence tracks degradation honestly (clean ~0.90–0.94, low contrast ~0.89,
skew ~0.85, noise ~0.70) and is never over-optimistic. Every sample raises
`possible_contraction_issue` (correct — Grade 2 is not interpreted) and
`unclear_braille_cell`, so reviewers are warned about the cells that can be
misread.

## OCR logic changes

**None.** Per this stage's conservative default, no detection, grouping, or
confidence logic was changed. The row-anchoring limitation above is small
(~1% overall, median 0), already flagged on every sample, and a fix would touch
the anchor-hypothesis logic that keeps the original, embossed, and Grade 1
datasets perfect — so it is documented as a finding and left for a future,
narrowly-scoped, regression-guarded stage. The only production-code addition is
an evaluation-only cell renderer helper (`render_cells_image`, a pure refactor
of the existing text renderer) plus the new evaluation modules and tests.

## Regression checks

Original (CER 0.000, conf 0.950), embossed (CER 0.000, conf 0.812), and
controlled Grade 1 (mean CER 0.008) datasets are unchanged. Full test suite
passes (126 tests, 14 new).

## Limitations

- Controlled renders from BRF cells, not photographs/scans of embossed pages;
  real-capture accuracy remains unproven.
- rawBraille CER is space-sensitive; cell error rate is the primary dot-reading
  measure.
- No Grade 2 English transcription is measured or claimed.

## What is needed before Grade 2 English transcription can be measured fairly

1. A verified Grade 2 back-translation path (e.g. Liblouis with the UEB Grade 2
   table — the engine's translation layer already prefers Liblouis when
   present).
2. Only then does English CER/WER on these triplets (text PDFs as ground truth)
   become meaningful.

## Next recommended stage

Either add and validate a Grade 2 back-translation path to unlock English
CER/WER on the UKAAF triplets, or a narrowly-scoped, regression-guarded fix for
the isolated low-row-dot row-anchoring case identified here.
