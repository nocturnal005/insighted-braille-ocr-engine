# Stage 3D-G4: Dot-Row Anchor Robustness (commit-safe)

Investigates and safely fixes the recurring row-lift error identified in Stage
3D-G3, where a whole Braille line using only the middle/lower rows of the cell
was read one row too high. No UKAAF source text, BRF content, rawBraille
output, or generated images appear in this document.

> OCR output remains **draft-only**. QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. This stage adds **no** Grade 2 contraction/back-translation support,
> computes **no** English CER/WER on Grade 2 material, and leaves the `/ocr`
> response contract unchanged.

## What Stage 3D-G3 revealed

G3 scored the visual pipeline on 21 controlled UKAAF Grade 2 renders and found
a mean cell error of 0.010 (median 0.000) with an exact-sample-match rate of
0.571. Every residual error was a cell **substitution** where a middle- or
lower-row-only dot was read one row too high — the comma cell {2} decoding as
{1} dominated. Prose, recipe, jokes, product-instructions, and French samples
carried the error; maths, code, and the press release were perfect.

## The row-lift pattern investigated

The pipeline groups dots into horizontal rows, splits them into Braille lines,
then assigns each row a within-cell index (0, 1, 2) **relative to the topmost
row cluster of that line** (`y0 = row_centers[group[0]]`). This assumes the
topmost detected row of every line is physical row 0 — true only when the line
contains at least one top-row dot.

## Evidence

Instrumenting the grouping on the affected samples showed the failing lines
had a **single row cluster** whose dots all sat in the middle row band, and the
anchor placed that band at row 0 — lifting the whole line up by one row.

A scan of all nine clean samples found the cause is not scattered cells but
**whole single-row lines**: 5 of 214 lines (2.3%) consist entirely of
middle-row-only cells, and those five lines are exactly the five samples that
were imperfect in G3. Maths/code/press-release contain no such lines, which is
why they were already perfect.

## Change made

`app/ocr/cell_grouping.py` gains `_line_lifts()`, which anchors each line to
the **page line ladder** instead of to its own topmost cluster:

1. Estimate the line pitch from the regular spacing of line origins (rejected
   unless it is a standard 3.5–6.5 dot pitches).
2. Pick a **reference line that provably contains all three physical rows**
   (three clusters spanning ~two dot pitches, so its topmost cluster is
   certainly row 0).
3. For each line, predict its origin on the ladder from the reference; if the
   line's topmost cluster sits a whole dot pitch or more below its predicted
   origin, lift its row indices down by that many rows (capped at 2).

When any line is lifted, a low-severity `unclear_braille_cell` flag records
that the row position was **inferred** from the page spacing and should be
checked — preserving the draft-only, verify-first stance.

### Why the change is safe

The correction returns **all-zero lifts** (a complete no-op) whenever the
ladder cannot be trusted: fewer than three lines, no regular line pitch, or no
full-height reference line. Any line that already carries a top-row dot sits on
the ladder and gets lift 0. Because clean scans, embossed samples, and Grade 1
pages have no single-row (top-missing) lines, the correction never fires on
them. This was verified empirically: with the correction on vs off, the grouped
cell output is **byte-identical** on every original, embossed, and Grade 1
image, and the corrected path reproduces the previous grouping exactly on those
datasets.

## Impact on datasets

| Dataset | Before G4 | After G4 |
| --- | --- | --- |
| original | CER 0.000, conf 0.950 | CER 0.000, conf 0.950 (unchanged) |
| embossed | CER 0.000, conf 0.812 | CER 0.000, conf 0.812 (unchanged) |
| controlled Grade 1 | mean CER 0.008 | mean CER 0.008 (unchanged) |
| real_anonymised audit | 10 evaluable | 10 evaluable (runs, unchanged) |
| UKAAF Grade 2 rawBraille | mean cell err 0.010, exact 0.571 | **mean cell err 0.000, exact 1.000** |

UKAAF Grade 2: 21/21 samples, 0 failed, line-count and cell-count mismatch
0.000 (unchanged), line-reconstruction accuracy 1.000, mean confidence ~0.88.
Tests: 133 pass (7 new in `app/tests/test_row_anchor.py`, covering the row-lift
pattern, middle- and lower-row-only lines, the `_line_lifts` ladder logic and
its conservative no-op fallbacks, an unchanged normal page, and the unchanged
`/ocr` contract).

## Limitations

- Validated on controlled renders and synthetic fixtures, not on photographed
  or scanned embossed pages; real-capture behaviour remains unproven.
- The ladder needs a full-height reference line somewhere on the page. A page
  whose lines are *all* single-row (no reference) is left uncorrected by
  design — the absolute row position is genuinely unknowable and a guess would
  be less safe than the honest, flagged uncertainty.
- This is cell-level (rawBraille) reliability only. It is **not** Grade 2
  English transcription accuracy and makes no real-world accuracy claim.

## Recommended next stage

Either add and validate a Grade 2 back-translation path (e.g. Liblouis with the
UEB Grade 2 table) to unlock fair English CER/WER on the UKAAF triplets, or
begin real photographed/embossed capture validation (the standing operational
goal).
