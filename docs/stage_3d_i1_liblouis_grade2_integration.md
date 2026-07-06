# Stage 3D-I1: Liblouis Grade 2 back-translation integration

Integrates Liblouis as an optional, locally installed back-translation
engine for Grade 2 (contracted) UEB, and extends the rawBraille evaluation
harness to compute supplementary English CER/WER when Liblouis Grade 2 is
available.

This stage changes **no OCR visual pipeline logic** — dot detection, cell
grouping, and line reconstruction are untouched. It adds a back-translation
path that interprets Grade 2 contractions via Liblouis, honest flags when
that path is used, and a gated English evaluation layer on top of the
existing cell-level rawBraille harness.

> OCR output remains **draft-only**. QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. Liblouis back-translation is a tool — it does not certify the
> result.

## What changed

### 1. Liblouis adapter hardening

`app/translation/liblouis_adapter.py` now:

- discovers the Liblouis DLL and table files from two new optional config
  settings (`LIBLOUIS_DLL_DIR`, `LIBLOUIS_TABLE_PATH`) — no manual env var
  setup or system-wide install required;
- resolves bare table names (like `en-ueb-g2.ctb`) to absolute paths in the
  configured table directory, working around Liblouis's DLL-load-time table
  path caching on Windows;
- exposes `is_grade2_table(table)` — a heuristic for the pipeline's
  grade-aware flag.

### 2. Pipeline: Grade 2 flag

When Liblouis is used with a Grade 2 table, the pipeline appends an honest
`possible_contraction_issue` flag:

> Back-translation used Liblouis with a Grade 2 (contracted) UEB table.
> Contractions were interpreted but the result is an unverified draft.

The `FALLBACK_TRANSLATION_CAP` (0.95) does **not** apply when Liblouis is
active — Liblouis Grade 2 translation is table-driven and more capable than
the built-in fallback. The emboss cap (0.82) still applies to embossed-photo
runs regardless of translation method.

### 3. Evaluation: supplementary English CER/WER

The rawBraille evaluation harness (`run_rawbraille_evaluation`) now detects
whether Liblouis Grade 2 is configured and available. When it is:

- the expected rawBraille for each sample is back-translated through
  Liblouis to derive a reference English text;
- the pipeline's `draftText` (which also uses Liblouis Grade 2) is compared
  against that reference;
- English CER and WER appear as supplementary columns in the console output
  and as `english_summary` in the JSON report;
- the report's `english_cer_wer_computed` field is `true` and
  `grade2_english_transcription` is `supplementary_via_liblouis`.

When Liblouis Grade 2 is **not** available (the default), English scoring is
skipped and the report carries the same `english_cer_wer_computed: false` as
before — no behaviour change.

**Caveat**: English reference text is derived from the expected rawBraille
cells via Liblouis, not from a separately authored transcript. English
CER/WER therefore reflects both visual pipeline accuracy (cell detection)
and Liblouis translation quality. The report and console output state this
explicitly.

### 4. Config additions

| Variable             | Default  | Meaning |
| -------------------- | -------- | ------- |
| `LIBLOUIS_DLL_DIR`   | *(empty)* | Directory containing `liblouis.dll` (Windows). Leave empty when on PATH. |
| `LIBLOUIS_TABLE_PATH`| *(empty)* | Directory containing `.ctb`/`.utb` table files. Leave empty for default. |

Existing settings are unchanged. To enable Grade 2:

```env
LIBLOUIS_TABLE=en-ueb-g2.ctb
LIBLOUIS_DLL_DIR=_external_sources/liblouis/bin
LIBLOUIS_TABLE_PATH=_external_sources/liblouis/share/liblouis/tables
```

### 5. Local Liblouis install (gitignored)

Liblouis 3.38.0 win64 binaries and tables live under
`_external_sources/liblouis/` (gitignored with all other external sources).
The Python `louis` ctypes bindings are installed into `.venv/` (also
gitignored). No new entries in `requirements.txt` — Liblouis remains
optional.

## What this stage does NOT do

- No visual pipeline changes (preprocessing, dot detection, cell grouping,
  line reconstruction are identical).
- No contract changes (`/ocr` request and response shapes unchanged).
- No new Python package dependencies in `requirements.txt`.
- No deployment or infrastructure changes.
- No accuracy claims — English CER/WER is supplementary measurement, not a
  certification.
- The built-in Grade 1 fallback translator is unchanged and still used when
  Liblouis is absent.

## Running with Grade 2

```powershell
cd D:\insighted-braille-ocr-engine
.venv\Scripts\activate

# Set the three Liblouis env vars
$env:LIBLOUIS_TABLE = "en-ueb-g2.ctb"
$env:LIBLOUIS_DLL_DIR = "_external_sources\liblouis\bin"
$env:LIBLOUIS_TABLE_PATH = "_external_sources\liblouis\share\liblouis\tables"

# Pipeline now interprets Grade 2 contractions
python -m app.demo.local_preview samples\images\sample_01_hello_world.png

# rawBraille evaluation with supplementary English CER/WER
python -m app.evaluation.run_rawbraille_evaluation --dataset ukaaf_grade2_raw
```

## Tests

```powershell
# Without Liblouis (default): 2 DLL-dependent tests skip cleanly
python -m pytest app/tests/test_liblouis_grade2.py -v

# With Liblouis: all 10 pass
$env:LIBLOUIS_DLL_DIR = "_external_sources\liblouis\bin"
$env:LIBLOUIS_TABLE_PATH = "_external_sources\liblouis\share\liblouis\tables"
python -m pytest app/tests/test_liblouis_grade2.py -v
```

## Recommended next stage

Collect real-capture samples (Stage 3D-G6 protocol) and evaluate with
Liblouis Grade 2 enabled to get the first end-to-end English CER/WER on
real photographed Braille. This is the standing operational goal: measure
correction burden on real material, not just controlled renders.
