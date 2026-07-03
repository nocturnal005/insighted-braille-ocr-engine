# insighted-braille-ocr-engine

Standalone specialist Braille OCR engine for **InsightEd AI**.

> **Draft-only OCR.** Every output of this service is an unverified draft
> transcription. It must be checked by a **QTVI or Braille-literate
> specialist** before any use in teacher feedback or export. This engine
> never claims certified Braille accuracy. *AI drafts, humans verify.*

## What this is

A Python FastAPI service that accepts a Braille page image (PNG/JPEG as a
base64 data URL), runs a geometric OCR pipeline (preprocess → dot detection →
cell grouping → line reconstruction → Unicode Braille → back-translation),
and returns a draft transcription with a confidence score, raw cell data,
and explicit uncertainty flags — in the exact JSON contract expected by the
InsightEd AI `external_braille_ocr` adapter.

Version 1 targets a **safe, tested, contract-compatible API**, not perfect
Braille recognition. See [limitations.md](limitations.md) before relying on
any output.

## Endpoints

| Method | Path       | Purpose                                              |
| ------ | ---------- | ---------------------------------------------------- |
| GET    | `/health`  | Liveness probe                                       |
| GET    | `/version` | Service name, version, and the draft-only warning    |
| POST   | `/ocr`     | Draft Braille OCR (optional API key, see below)      |

`/ocr` **always returns HTTP 200 with valid contract JSON** for a
well-formed request body. If OCR fails (unsupported type, bad base64,
unreadable image, no dots found), the response has an empty `draftText`,
`confidence` 0, and clear uncertainty flags — never an exception, never
fabricated text. Malformed request bodies get a standard 422 validation
response; a wrong/missing API key (when configured) gets 401.

### Request (POST /ocr)

```json
{
  "taskId": "task-demo-001",
  "title": "Braille homework page 1",
  "fileName": "braille-page-1.png",
  "mimeType": "image/png",
  "dataUrl": "data:image/png;base64,....",
  "subject": "Science",
  "yearGroup": "Year 9"
}
```

Supported `mimeType` values: `image/png`, `image/jpeg`, `image/jpg`.
**PDF is not supported in v1** — a PDF returns a controlled response with a
flag explaining that PDF OCR is not yet available.

### Response

```json
{
  "draftText": "hello world",
  "confidence": 0.94,
  "rawBraille": "⠓⠑⠇⠇⠕ ⠺⠕⠗⠇⠙",
  "rawCells": [
    {
      "line": 1,
      "cellIndex": 1,
      "dots": [1, 2, 5],
      "bbox": [120, 40, 150, 90],
      "confidence": 0.91
    }
  ],
  "providerRequestId": "ocr_6f0f4a…",
  "flags": [
    {
      "text": "",
      "reason": "Back-translation used the built-in Grade 1 (uncontracted) UEB table…",
      "category": "possible_contraction_issue",
      "severity": "low"
    }
  ],
  "pageResults": [
    { "pageNumber": 1, "text": "hello world", "confidence": 0.94, "flags": [] }
  ]
}
```

Real captured examples live in
[samples/sample_request.json](samples/sample_request.json) and
[samples/sample_response.json](samples/sample_response.json).

Flag categories: `low_image_quality`, `low_ocr_confidence`,
`unclear_braille_cell`, `possible_contraction_issue`,
`possible_number_sign_issue`, `possible_capitalisation_issue`,
`line_order_uncertainty`, `word_spacing_uncertainty`,
`subject_specific_term`. Severities: `low`, `medium`, `high`.
`flags[].text` is empty for whole-image flags and holds the affected
cell/letter for cell-level flags.

## Quickstart (Windows)

```powershell
cd D:\insighted-braille-ocr-engine

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the tests
python -m pytest

# Run the service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then:

- `GET http://localhost:8000/health`
- `GET http://localhost:8000/version`
- `POST http://localhost:8000/ocr` with the request JSON above
  (try `samples/sample_request.json` as a ready-made body)

PowerShell example:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ocr `
  -ContentType "application/json" `
  -InFile samples\sample_request.json
```

## Configuration

Copy `.env.example` to `.env` and adjust. Key settings:

| Variable             | Default          | Meaning                                          |
| -------------------- | ---------------- | ------------------------------------------------ |
| `OCR_ENGINE_API_KEY` | *(empty)*        | When set, `/ocr` requires the API key (see below)|
| `MAX_IMAGE_MB`       | `10`             | Maximum decoded image size                       |
| `MAX_IMAGE_PIXELS`   | `40000000`       | Decompression-bomb guard                         |
| `LOG_LEVEL`          | `INFO`           | Logging level (metadata only — see below)        |
| `LIBLOUIS_ENABLED`   | `true`           | Attempt Liblouis back-translation if installed   |
| `LIBLOUIS_TABLE`     | `en-ueb-g1.ctb`  | Liblouis table to use                            |

**Authentication:** when `OCR_ENGINE_API_KEY` is set, `POST /ocr` accepts
the key in **either** header form:

```
X-API-Key: <key>
Authorization: Bearer <key>
```

InsightEd AI's `external_braille_ocr` adapter sends
`Authorization: Bearer <key>`, so no change to the main app is needed.
`/health` and `/version` remain open for probes. When the variable is empty,
authentication is disabled (local development only).

**Logging safety:** the service logs request metadata only (request ids,
byte counts, dot/cell counts, confidence, durations). Task ids appear in
logs only as a short non-reversible hash (`task_ref`). It never logs image
data, base64 payloads, transcription text, file names, task titles, raw
task ids, pupil data, or API keys.

## OCR pipeline

1. **Image intake** — safe base64 data URL decode; MIME, size, and pixel
   validation; PDF and unsupported types rejected with controlled flags.
2. **Preprocessing** — two binarisation strategies (Stage 3D-D):
   * *dark path*: denoise, CLAHE contrast enhancement, adaptive
     thresholding, polarity check, optional deskew (dark printed dots);
   * *emboss path*: illumination flattening, then highlight/shadow blob
     pairing along a self-calibrated light direction, reconstructing one
     clean candidate at each raised dot's true centre (embossed
     photographs). A heuristic quality score accompanies each.
3. **Dot detection & variant selection** — contour analysis with
   circularity and size filters and per-dot confidence (centre, radius,
   area, bounding box); both preprocessing variants are detected and
   grouped, and the one whose dots actually form a Braille grid wins.
4. **Cell grouping** — grid fitting of the 2×3 Braille cell structure:
   estimates dot pitch and cell advance, resolves the column-anchor
   ambiguity, corrects residual skew measured from the dot geometry,
   rescues collapsed dot rows (or fails safely when they cannot be
   separated), and assigns dots to numbered positions 1–6.
5. **Line reconstruction** — orders lines and cells, derives word spacing
   from blank grid cells, emits `rawCells`.
6. **Braille decoding** — dot patterns → Unicode Braille (`rawBraille`).
7. **Back-translation** — Liblouis if installed; otherwise a built-in
   Grade 1 (uncontracted) UEB fallback translator plus an explicit flag.
8. **Confidence & flags** — blends image quality, detection certainty,
   dot-spacing regularity, grid fit, line certainty, and translation
   completeness into `confidence`, plus the uncertainty flags listed above.
   Honesty caps: embossed-photo runs never exceed **0.82** and fallback
   (non-Liblouis) translation never exceeds **0.95** — uncertain conditions
   must never read as near-certainty.

Liblouis is **not** image OCR — it only back-translates the already-detected
Braille cells. If it is not installed, nothing crashes; the response still
carries `rawBraille`/`rawCells` and a clear uncertainty flag.

## Samples and evaluation

Regenerate the bundled synthetic samples (and the contract example files):

```powershell
python -m app.evaluation.sample_generator
```

Run the evaluation harness (CER, WER, repeatability, timing, failures,
flag-category summary, confidence-vs-error summary):

```powershell
# bundled datasets (shorthand)
python -m app.evaluation.run_evaluation --dataset original
python -m app.evaluation.run_evaluation --dataset embossed

# or explicit directories (unchanged)
python -m app.evaluation.run_evaluation --images ./samples/images --truth ./samples/ground_truth
python -m app.evaluation.run_evaluation --images ./samples/embossed_images --truth ./samples/embossed_ground_truth
```

Ground truth is a `.txt` file per image with the same file stem. The
bundled samples are synthetic renders — the `images/` set is ideal
black-dot-on-white, the `embossed_images/` set simulates embossed-paper
photographs (highlight/shadow dots, low contrast, noise, uneven light,
skew, spacing variation). They prove the pipeline round-trips under those
conditions, not real-world accuracy. The harness prints metrics only, never
transcription text. Do not use pupil-identifying names in sample files.

### Stage 3D-D: embossed-photo handling

Stage 3D-D adds embossed-paper-photograph support: raised dots that appear
only as highlight/shadow crescent pairs under directional light are now
paired and reconstructed into dot candidates, with illumination flattening
for unevenly lit pages, dot-geometry skew correction, and spacing-tolerant
row clustering. On the bundled synthetic embossed set (12 samples covering
low contrast, shadows, skew, paper noise, uneven light, numbers,
multi-line, wide/tight spacing, faint dots, and rotation) the engine
decodes all samples at CER 0.000 with confidence honestly capped at 0.82.

This does **not** make the engine production-certified. Output remains
**draft-only**: InsightEd AI holds every draft for mandatory QTVI /
Braille-literate specialist verification before teacher feedback or
export. Real photographs of real embossed pages will be harder than the
simulation — treat flags and confidence as the honest signal they are.

**Image-capture guidance for best results:**

- photograph or scan **Braille-only page sections** — avoid English print
  or handwriting on the same image;
- use good, even lighting; mild directional light is fine (it is what
  makes embossed dots visible) but avoid harsh shadows;
- keep the page **flat** and the camera square to it (small tilt is
  corrected; heavy skew or curvature is not);
- crop to the Braille area with a small margin, at a resolution where a
  dot is at least ~6 pixels across;
- never include pupil names or identifying material — use synthetic or
  anonymised pages for any testing.

## Docker

```powershell
docker build -t insighted-braille-ocr-engine .
docker run --rm -p 8000:8000 insighted-braille-ocr-engine
```

## InsightEd AI integration (later — do not connect yet)

The main InsightEd AI app calls this service through its
`external_braille_ocr` adapter. When the time comes, the app side is
configured with:

```
AI_MODE=real
BRAILLE_OCR_PROVIDER=external_braille_ocr
BRAILLE_OCR_ENDPOINT=http://localhost:8000/ocr
BRAILLE_OCR_API_KEY=local-test-key
BRAILLE_OCR_TIMEOUT_MS=30000
```

On this engine, set `OCR_ENGINE_API_KEY=local-test-key` (or any shared
secret) so the adapter's key is enforced. The adapter sends the key as
`Authorization: Bearer <key>`, which this engine accepts directly (it also
accepts `X-API-Key: <key>` for manual testing). The request/response shapes
in this README are exactly what the adapter sends and expects. Integration
notes:

- This engine remains draft-only: InsightEd AI keeps its mandatory QTVI /
  Braille-literate verification gate regardless of the confidence returned.
- `/ocr` answers a valid request with HTTP 200 controlled JSON even when OCR
  fails, so the adapter can rely on the response shape.
- Keep the standalone service tested on its own before wiring it into the
  main app. Do not modify the main InsightEd AI app to do so.

## Project structure

```
app/
  main.py            FastAPI app + global error handling
  api/               health, version, ocr endpoints
  core/              config, logging (metadata-only), API key security
  models/            Pydantic request/response contract models
  ocr/               decode → preprocess → dots → cells → lines → braille → confidence
  translation/       braille maps, Grade 1 fallback, optional Liblouis adapter
  evaluation/        CER/WER metrics, repeatability, harness, sample generator
  tests/             pytest suite (contract, decode safety, shape, metrics, e2e)
samples/             synthetic images, ground truth, contract examples
Dockerfile, .env.example, limitations.md
```

## Limitations

Read [limitations.md](limitations.md). Headline: output is **draft-only and
requires QTVI or Braille-literate specialist verification**; v1 handles
clear dark-dot images (not embossed-paper photographs), PNG/JPEG only, no
PDF, Grade 1 back-translation fallback, single page.
