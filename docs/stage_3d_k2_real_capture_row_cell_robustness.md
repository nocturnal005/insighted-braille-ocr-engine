# Stage 3D-K2: Real-capture row and cell grouping robustness

Improves how the engine progresses from **L1 (dot candidates detected)** to
**L2/L3/L4 (rows → cells → rawBraille)** on genuine physical Braille captures,
by adding a lattice-projection fallback to row grouping.

> This is a diagnostic robustness improvement, **not** proof of real-world OCR
> accuracy. OCR output remains **draft-only**: QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. Formal real-capture accuracy remains pending until permission-safe
> physical samples with trusted `.braille` ground truth are collected and
> scored at L6.

## Baseline (K1) and the bottleneck

K1 ran five real physical Braille photos preview-only. Two embossed-paper
samples reached L5; three stalled at **L1** — hundreds of dots detected, but
zero cells formed.

Instrumenting the three stalls showed the cause precisely: the row grouper
clusters dots into rows with **single-linkage 1-D clustering on y**
(`_cluster_1d`). On a dense, mildly curved, or slightly skewed real capture,
a dot near the left edge of one row and a dot near the right edge of the row
above overlap in `y`, so single-linkage **chains adjacent rows across the
frame**. One merged cluster then trips the global gate
(`_max_row_spread(rows) > 0.8 * u_v`), and the existing collapse-retry (which
halves the *global* threshold) over-fragments the clean rows before it can
separate the merged one — so the page fails to an empty result even though the
dots plainly lie on a regular grid.

## The change (one production file: `app/ocr/cell_grouping.py`)

A **lattice-projection fallback**, tried *only* at the point where grouping
would otherwise return the empty safe-failure:

1. `_estimate_vertical_pitch` — a robust vertical dot pitch `u_v` from the
   *within-column* downward nearest-neighbour gaps. It never clusters along
   `y`, so it is immune to the row-chaining that defeats single-linkage, and
   it is local, so it tolerates mild skew.
2. `_lattice_rows` — assigns each dot to row `round((y - y0) / u_v)`, a lattice
   position rather than a proximity cluster, so **adjacent rows can no longer
   chain**. Reports the median per-dot residual as a fraction of the pitch.
3. `_recover_rows_by_lattice` — returns recovered rows **only** when the dots
   genuinely lie on a regular lattice, gated by:
   - `len(dots) >= 12` (a real page, not a fragment);
   - `spacing_regularity(dots) >= 0.60` — **the noise guard**;
   - `u_v >= 2 * r_med` (a plausible Braille pitch);
   - median residual ratio `< 0.30` (dots within ~⅓ pitch of their row);
   - at least 2 occupied rows.

When recovery fires, a medium `line_order_uncertainty` flag is added and the
result is marked `recovered_via_fallback`, which the pipeline treats as a hard
**confidence cap of 0.55** (`LATTICE_RECOVERY_CAP`, below the emboss cap) — see
"Honest confidence" below.

### Honest confidence on recovery (adversarial-review hardening)

An adversarial review of the first cut raised two real risks, both fixed here:

- **Overconfidence.** `line_quality` carries only weight 0.10 in the confidence
  blend, so capping it alone removed at most ~0.05 — and `grouping.quality`
  (weight 0.20, from column fit) was untouched, so a recovery with clean columns
  could still read ~0.9. Fixed with a **pipeline-level hard cap**
  (`LATTICE_RECOVERY_CAP = 0.55`) applied to any `recovered_via_fallback` page,
  regardless of column fit. A last-ditch recovery can no longer read as
  confident.
- **Periodic non-Braille texture.** The noise guard uses nearest-neighbour
  spacing regularity, which a *regular* texture (mesh, halftone, grid paper)
  scores high on — so a texture can pass the lattice gate. This is bounded two
  ways: a uniform texture does not fit Braille's paired-column cell advance, so
  `grouping.quality` (and thus confidence) is low; and the 0.55 recovery cap
  plus the recovery flag ensure any such draft is honestly low-confidence and
  flagged, never confident. Verified: a synthetic uniform dot grid decodes at
  confidence ≤ 0.50 with flags, never as reliable text.

Known, bounded limitations (all held ≤ 0.55 and flagged, never silent): on a
recovered page, very tight interline spacing (< 2.4× the dot pitch) or a
sub-harmonic pitch estimate can mis-assign rows, producing wrong-but-flagged
cells rather than a correct draft. This is acceptable under the draft-only
contract — a specialist verifies every draft — and the honest confidence/flags
are the signal.

### Why a spacing-regularity gate, not just residual

A low residual ratio **cannot** reject noise: uniformly-random points projected
onto any lattice sit a median ~0.25 of the pitch from the nearest lattice line,
which passes a 0.30 residual gate. Measured nearest-neighbour spacing
regularity is the discriminator that separates them — the five real captures
measured 0.72–0.84, random noise measured ~0.51. The floor of **0.60** sits
above that noise level, below every real capture, and just above the pipeline's
existing "inconsistent spacing" flag threshold (0.55). These thresholds are
geometric/statistical criteria, not values fitted to the sample set.

### Why this cannot regress controlled performance

The fallback runs **only** on the failure path — after the existing row-spread
gate has already decided the page could not be grouped. Any image that
currently groups successfully (controlled UKAAF renders, clean scans, the
embossed synthetic set) never reaches it, so their output is byte-for-byte
unchanged. The change can only turn a would-be empty failure into a recovered,
honestly-flagged draft, or leave the failure as-is.

## Diagnostic result (local-only, preview-only, `D:\braille samples`)

Five real physical Braille photos, run via `--input` (not copied into the repo,
never committed, no ground truth, **not scored**):

| Sample | Baseline stage (G2) | After K2 (G2) | Confidence | Quality |
| ------ | ------------------- | ------------- | ---------- | ------- |
| withheld_001 | L5 | L5 (held, normal path) | 0.81 | readable |
| withheld_002 | L5 | L5 (held, normal path) | 0.80 | borderline |
| withheld_003 | **L1** | **L5** (recovered) | 0.55 (capped) | retake |
| withheld_004 | **L1** | **L5** (recovered) | 0.55 (capped) | borderline |
| withheld_005 | **L1** | **L5** (recovered) | 0.55 (capped) | retake |

All three previously-stalled captures now reach rawBraille (and a Grade 2 draft
with Liblouis enabled); the two prior L5 samples held with unchanged confidence
(they group via the normal path — the fallback never touches them). The three
recovered samples are hard-capped at the recovery ceiling (0.55) and flagged
`retake_recommended` / `borderline_candidate` — the improvement is reaching a
draft at all, **not** high confidence.

**Real-capture robustness improved diagnostically.** This is not proof of
real-world OCR accuracy: no sample was scored against `.braille` ground truth,
and the confidence/quality flags remain the honest signal.

## Genuinely-bad input still fails safely

The noise guard is load-bearing and tested: random dot-rich noise
(`spacing_regularity ~0.46-0.51`) is rejected by `_recover_rows_by_lattice` and
by `group_dots` end-to-end (0 cells, honest flags) — it is never hallucinated
into a grid.

## Tests

New synthetic-only suite `app/tests/test_row_cell_robustness.py` (13 tests):
lattice pitch estimation, lattice-recovery accept/reject gates (regular lattice,
jitter tolerance, **noise rejection**, too-few-dots), `group_dots` noise → no
cells, clean lattice unaffected, blurred/skewed/tight synthetic renders
(no crash, valid contract, honest confidence), and the `/ocr` contract lock.

Two existing tests updated to the new (better) behaviour: the tightly-spaced
synthetic that used to fail outright now recovers with confidence honestly
capped (≤ 0.82) and a high-severity size-floor flag. The safe-failure guarantee
for noise/garbage is preserved by `test_noise_only_image_fails_safely` and the
new noise-rejection tests.

## Validation summary

- `pytest`: **231 passed, 3 skipped** (default); **234 passed** (Liblouis
  Grade 2 configured).
- Controlled UKAAF Grade 2: **cellER 0.000**; supplementary English **CER/WER
  0.000** with Liblouis Grade 2 — no regression.
- `/ocr` request/response contract unchanged; Grade 1 fallback unchanged;
  Liblouis still optional; no new dependencies (numpy already used repo-wide).
- No real images, `.braille`, metadata, or reports committed; diagnostic output
  is local-only, gitignored, and content-free.
