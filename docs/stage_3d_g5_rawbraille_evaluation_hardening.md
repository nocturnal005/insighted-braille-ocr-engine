# Stage 3D-G5: rawBraille evaluation hardening and real-capture readiness (commit-safe)

Hardens the rawBraille evaluation framework built in Stage 3D-G3 (and validated
at exact-match 1.000 after the Stage 3D-G4 anchor fix) and prepares the engine
for future validation on real photographed or scanned Braille — without adding
any new OCR capability. No UKAAF source text, BRF content, rawBraille output,
or generated images appear in this document.

> OCR output remains **draft-only**. QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. This stage adds **no** Grade 2 contraction/back-translation support
> (Liblouis is not added), computes **no** English CER/WER on Grade 2 material,
> and leaves the `/ocr` response contract unchanged.

## Why this stage exists after G4

G4 closed the last controlled-render defect: all 21 UKAAF Grade 2 renders now
decode with cell error 0.000. The next meaningful evidence must come from
**real captures** — photographs/scans of physical embossed Braille. Before any
physical sample is handled, the evaluation framework must make it impossible to
(a) confuse controlled-render results with real-capture results, (b) ask for
English Grade 2 scoring, or (c) let sensitive material slip into the repo. That
is what this stage adds.

## Controlled-render vs real-capture validation

- **Controlled render** (`capture_type=controlled_render`): images generated
  locally from BRF cells with known geometry. They prove the geometry pipeline
  (detection → grouping → line reconstruction → rawBraille) is correct, and they
  catch regressions — but they say **nothing** about camera focus, paper
  texture, lighting, dot deformation, or page curvature.
- **Real capture** (`capture_type=real_capture`): photographs/scans of physical
  Braille. Only these can support any statement about real-world usefulness.

Every rawBraille dataset is now registered with an explicit
`capture_type` (`app/evaluation/rawbraille_dataset.DATASETS`), and every
console run and JSON report opens with a banner stating which kind it is.
Controlled-render reports state explicitly that they do **not** demonstrate
real-world capture accuracy.

## What cell-level (rawBraille) metrics can and cannot prove

Can prove: dots were detected, grouped into the right cells, in the right rows
and columns, on the right lines — i.e. the *visual reading* of the page matches
the source cells (cell error rate, line/cell-count mismatch, exact match).

Cannot prove: English transcription accuracy (Grade 2 contractions are not
interpreted — deliberately), certified Braille accuracy, or real-world OCR
accuracy while the dataset is controlled renders. Reports carry
`english_cer_wer_computed: false` and `grade2_english_transcription:
"out_of_scope"` as machine-readable statements of this.

## How to run the hardened evaluation

```bash
# Controlled UKAAF Grade 2 renders (local-only samples required):
python -m app.evaluation.run_rawbraille_evaluation --dataset ukaaf_grade2_raw

# Future real captures (empty intake until safe samples are added):
python -m app.evaluation.run_rawbraille_evaluation --dataset real_capture_grade2_raw

# Optional sanitized JSON report (metrics + safe labels only):
python -m app.evaluation.run_rawbraille_evaluation --dataset ukaaf_grade2_raw \
    --write-report reports/ukaaf_g3_rawbraille/ukaaf-grade2-rawbraille-report.json
```

Reports (schema 1.1) include: a dataset descriptor (name, category,
capture type, source type, grade mode, evaluation mode), a `run_id`, counts
(samples/evaluated/skipped/failed), mean & median cell error, exact-match rate,
line-/cell-count mismatch rates, a confidence summary (mean/median/min/max),
an uncertainty-flag summary, and the standing draft-only note.

## How to audit a future real-capture dataset safely

```bash
python -m app.evaluation.audit_rawbraille_dataset --dataset real_capture_grade2_raw
# optionally with a manifest:
python -m app.evaluation.audit_rawbraille_dataset --dataset real_capture_grade2_raw \
    --manifest path/to/manifest.json
```

The audit reports (never modifies files) and exits 2 on blocking issues:
images without expected `.braille` files, `.txt` English transcripts in the
expected folder (out of scope), unsafe sample ids (pupil/school/assessment
patterns — the id itself is withheld from output), missing or unapproved
permission metadata, `contains_real_pupil_data` / live-assessment markers,
capture-type mismatches, invalid manifest entries, and sample folders not
covered by `.gitignore`. An empty intake folder is a normal, clearly-reported
state — not an error.

Intake locations (local-only, gitignored, `.gitkeep` placeholders committed):

```
samples/real_rawbraille_images/    *.png / *.jpg
samples/real_rawbraille_expected/  <stem>.braille   (cells only, never English)
samples/real_rawbraille_metadata/  <stem>.json
```

## Manifest schema (Stage 3D-G5)

`app/evaluation/rawbraille_manifest.py` defines a minimal, dependency-free
manifest: one JSON entry per sample with required fields `sample_id`,
`image_path`, `expected_rawbraille_path`, `dataset_category`, `capture_type`,
`source_type`, `consent_or_safety_note`, `grade_mode`, `evaluation_mode`.
Validation enforces safe (anonymised) sample ids and — structurally — that
`evaluation_mode` can only be `rawbraille_cell_level`: there is no English
evaluation mode to request.

## Files that must remain local-only (gitignored)

UKAAF source PDFs/BRFs/DOC(X), generated Grade 2 renders, expected rawBraille
files, generated evaluation reports (`reports/`), debug images and local
diagnostics, external downloaded resources (`_external_sources/`), all real
capture samples/expected/metadata, real pupil or school data, `.env`/keys,
`.venv`/caches. The test suite asserts the coverage.

## Why Grade 2 English CER/WER remains out of scope

The engine back-translates Grade 1 only. Scoring Grade 2 pages against English
text would measure a deliberately unimplemented capability and produce
misleading numbers. It becomes meaningful only after a verified Grade 2
back-translation path (e.g. Liblouis with the UEB Grade 2 table) is added and
validated — a separate, explicit future stage.

## Results (engine 0.4.0, unchanged OCR logic)

Controlled UKAAF Grade 2: 21/21 evaluated, 0 failed, mean cell error 0.000,
exact-match 1.000 — identical to G4 (this stage changes evaluation tooling
only). Original (CER 0.000, conf 0.950), embossed (CER 0.000, conf 0.812), and
controlled Grade 1 (mean CER 0.008) datasets unchanged. 153 tests pass
(20 new in `app/tests/test_rawbraille_hardening.py`).

## Limitations

- Real-capture readiness is *framework* readiness: no physical samples have
  been evaluated yet, and no real-world accuracy is claimed.
- The audit's filename screening is pattern-based; a human must still confirm
  anonymisation and consent before any real sample is added.
- Grade 2 English transcription remains out of scope.

## Recommended next stage

Acquire a small, safe, approved set of real photographed/scanned Braille
samples (Braille-only crops, anonymised ids, explicit consent), run the
readiness audit, then perform the first real-capture rawBraille baseline —
reported strictly separately from the controlled-render baseline. Grade 2
back-translation (Liblouis) remains a later, separate stage.
