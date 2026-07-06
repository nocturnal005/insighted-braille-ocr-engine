# Stage 3D-J1: Real-capture OCR diagnostic and Grade 2 preview

A local diagnostic pathway for real physical Braille captures: for each
photographed or scanned page, it shows how far the existing OCR pipeline
gets — dot detection, row grouping, cell formation, rawBraille output, and
(when Liblouis Grade 2 is configured) draft translation — and why it
stopped where it stopped.

> **This is a diagnostic, not certified accuracy.** Stage labels and
> quality classifications describe pipeline progress on one image; they
> are not accuracy measurements and must never be quoted as accuracy
> evidence. Cell-level scores exist only for samples that pass the
> Stage 3D-G6 gating with `.braille` ground truth. OCR output remains
> **draft-only**: QTVI/Braille-literate specialist verification is
> mandatory in InsightEd AI before any teacher feedback or export.

No OCR logic changed in this stage. The `/ocr` request/response contract
is unchanged. Liblouis remains optional; the Grade 1 fallback is
unchanged. No new dependencies were added.

## What was added

| Piece | Where | Purpose |
| ----- | ----- | ------- |
| Diagnostic probe | `app/evaluation/diagnostic_probe.py` | Stage ladder L0–L6 per image, reusing production pipeline functions read-only |
| Capture-quality preflight | `app/evaluation/capture_quality.py` | readable / borderline / retake / unusable triage with safe reasons |
| Diagnostic CLI | `app/evaluation/run_real_capture_diagnostic.py` | Folder scan → leak-free local JSON/Markdown report |
| Tests | `app/tests/test_real_capture_diagnostic.py` | Synthetic-only fixtures; ladder, quality, leakage, gating |

## Sample rules (unchanged from Stage 3D-G6)

Real-capture samples must follow the
[Stage 3D-G6 collection protocol](stage_3d_g6_real_capture_collection_protocol.md):
self-authored or public-domain physical material only, never pupil work or
school assessment content, anonymous mechanical file names, Braille-only
crops, EXIF stripped, explicit `permission_status: "approved_for_testing"`
metadata, and `.braille` (cell-level) ground truth — never English
transcripts.

**External/public image packs are diagnostic-only.** They may be run
through this CLI to observe pipeline behaviour, but they are not approved
test data, must never enter the intake folders, must never be committed,
and must never be quoted as accuracy or capability evidence.

Metadata marked `contains_real_pupil_data` or
`contains_live_assessment_material` **blocks the sample completely**: the
CLI refuses to OCR it at all and exits with code 2.

## Running the diagnostic

From the repo root (paths in the engine are repo-root-relative):

```powershell
cd D:\insighted-braille-ocr-engine
.venv\Scripts\activate

# Default: scan the Stage 3D-G6 real-capture intake
python -m app.evaluation.run_real_capture_diagnostic `
    --report reports\real_capture_diagnostic\run.json

# Any local folder of PNG/JPEG candidates (preview-only unless gated)
python -m app.evaluation.run_real_capture_diagnostic `
    --input C:\path\to\candidates --report reports\real_capture_diagnostic\run.json
```

With Grade 2 translation enabled (optional, Stage 3D-I1):

```powershell
$env:LIBLOUIS_TABLE = "en-ueb-g2.ctb"
$env:LIBLOUIS_DLL_DIR = "_external_sources\liblouis\bin"
$env:LIBLOUIS_TABLE_PATH = "_external_sources\liblouis\share\liblouis\tables"
python -m app.evaluation.run_real_capture_diagnostic --report reports\real_capture_diagnostic\run_g2.json
```

An empty intake prints `verdict: BLOCKED (no real-capture candidates
present …)` — that is the honest, expected state until physical samples
are collected. The CLI never invents evidence.

Report paths inside the repository must be gitignored (`reports/` is);
the CLI refuses to write a committable report. Reports contain counts,
scores, stage labels, flag categories, and fixed reason strings only —
never rawBraille content, draft text, ground truth, or unsafe file names
(unsafely named files appear as `withheld_<hash>`).

## Interpreting the stage ladder

Each image is labelled with the **highest stage reached** plus a
**failure point** saying where processing stopped:

| Stage | Meaning | Typical failure points at this rung |
| ----- | ------- | ----------------------------------- |
| L0 | Safe rejection or no detectable Braille content | `decode_rejected`, `unsupported_file`, `no_dot_candidates`, `dots_rejected_by_filters` |
| L1 | Dot candidates detected (accepted by shape/size filters) | `row_separation_failed`, `no_cells_formed` |
| L2 | Rows (Braille lines) separated | `no_cells_formed` |
| L3 | Cells formed on the 2×3 grid | — |
| L4 | Non-empty rawBraille produced | `translation_failed` |
| L5 | Grade 2 draft produced via Liblouis (Grade 2 table configured **and** Liblouis actually translated) | — |
| L6 | Scored against gated `.braille` ground truth | — |

Reading the ladder for a failing capture:

- **L0 / `no_dot_candidates`** — the engine saw nothing dot-like. Usually
  lighting, contrast, or resolution; check the capture-quality reasons.
- **L0 / `dots_rejected_by_filters`** — marks were found but none looked
  like Braille dots (too small, wrong shape). Often resolution or noise.
- **L1 / `row_separation_failed`** — dots exist but their rows could not
  be separated; usually skew, curvature, or spacing damage.
- **L4 without L5** — the visual pipeline succeeded; only Grade 2
  translation is missing (Liblouis not configured or not installed).
  **Liblouis never fixes image problems** — it only translates cells that
  the visual pipeline already read. A capture failing at L0–L3 will not
  be helped by configuring Grade 2.
- **L5 drafts are unverified.** Every Grade 2 draft carries the
  `possible_contraction_issue` flag; treat it as a draft for specialist
  verification, exactly like every other output of this engine.

The capture-quality classification (`readable_candidate`,
`borderline_candidate`, `retake_recommended`, `unusable`) is retake
triage: when it says retake, a better capture will help more than any
pipeline tuning.

## Preparing the next real-capture baseline

1. Produce allowed physical samples (G6 protocol §1) and photograph/scan
   them per G6 §3–4.
2. Sanity-check each candidate with this diagnostic CLI (or the Stage
   3D-H1 preview) — aim for `readable_candidate` at L4+.
3. Complete intake: anonymous names, `.braille` ground truth from the
   source (never from OCR output), metadata JSON with explicit
   permission (`docs/templates/real_capture_rawbraille_metadata.template.json`).
4. Audit to READY: `python -m app.evaluation.audit_rawbraille_dataset --dataset real_capture_grade2_raw`.
5. Run the formal evaluation: `python -m app.evaluation.run_rawbraille_evaluation --dataset real_capture_grade2_raw`
   — reported strictly separately from controlled-render baselines.

## What this stage deliberately does not do

- No `/ocr` contract or pipeline changes — the probe replays existing
  stage functions read-only.
- No real-capture accuracy claims: with an empty intake, formal
  evaluation is **BLOCKED**, and the stage says so.
- No committed samples, reports, or OCR text — all diagnostic output is
  local-only and content-free.
- No new dependencies; Liblouis remains optional.
