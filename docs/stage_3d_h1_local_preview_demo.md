# Stage 3D-H1: Local preview and human demo workflow

Two small, local-only viewers for the existing OCR pipeline, so a human can
see what the engine does with one image without reading JSON in a terminal:

1. a **CLI preview** — `python -m app.demo.local_preview <image>` — that runs
   the pipeline in-process and prints a readable draft report;
2. an optional **browser demo page** — `GET /demo` — that uploads one image
   to the local service's existing `/ocr` endpoint and renders the response.

This stage changes **no OCR logic**, **no `/ocr` contract**, adds **no
dependencies**, and does **no deployment work**. It does not touch the main
InsightEd AI app.

> OCR output remains **draft-only**. Everything either tool displays is an
> unverified draft: QTVI/Braille-literate specialist verification is
> mandatory in InsightEd AI before any teacher feedback or export. Both
> tools show this warning prominently and permanently.

## Safety rules (unchanged, and they apply here too)

- Only preview material allowed by the collection protocols — self-authored
  or public-domain test pages, never real pupil work, school assessment
  material, or anything identifying a pupil or school. See
  [stage_3d_g6_real_capture_collection_protocol.md](stage_3d_g6_real_capture_collection_protocol.md)
  sections 1–2.
- Neither tool stores anything: the CLI writes no files; the demo page sends
  the image only to the local service (same origin), which processes it in
  memory. Service logging policy is unchanged (metadata only).
- A preview is a demonstration, **never evidence of accuracy**. Do not
  screenshot a good result and present it as a capability claim; accuracy
  statements come only from the evaluation harnesses.

## 1. CLI preview (no server needed)

From the repo root, with the virtual environment active:

```powershell
cd D:\insighted-braille-ocr-engine
.venv\Scripts\activate

# Preview a bundled synthetic sample
python -m app.demo.local_preview samples\images\sample_01_hello_world.png

# Preview an embossed-style sample, or any local PNG/JPEG
python -m app.demo.local_preview samples\embossed_images\embossed_01_clean.png
python -m app.demo.local_preview C:\path\to\your\braille_photo.jpg
```

The report shows: the draft-only banner, image name/size, request id,
duration, detected cell/line counts, confidence (an internal heuristic, not
a probability), every uncertainty flag, the detected Braille cells
(`rawBraille`), and the draft back-translation.

To see the exact `/ocr` contract response instead:

```powershell
python -m app.demo.local_preview samples\images\sample_01_hello_world.png --json
```

(`--json` prints pure JSON on stdout; the banner goes to stderr so the JSON
stays pipeable.)

Failure behaviour matches the service: an unreadable or dot-free image
prints an honest empty-draft report with high-severity flags — never a crash
and never fabricated text. A missing file or unsupported extension exits
with a usage error (code 2).

## 2. Browser demo page (for showing a human)

The page is **off by default** — `GET /demo` returns 404 until it is
explicitly enabled. Enable it for a local session and start the service:

```powershell
cd D:\insighted-braille-ocr-engine
.venv\Scripts\activate
$env:DEMO_PAGE_ENABLED = "true"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

(Or put `DEMO_PAGE_ENABLED=true` in your local `.env`. Never enable it in a
deployed integration.)

**Bind to loopback only while the demo is enabled.** Use
`--host 127.0.0.1` (or omit `--host`, which defaults to loopback) — never
`--host 0.0.0.0`. The page's promise that the image and API key stay on
this machine is only true while the service is unreachable from the
network; on `0.0.0.0` anyone on the same Wi‑Fi/LAN could open `/demo` and
send images and the API key across the network in cleartext.

Then open <http://localhost:8000/demo> in a browser:

1. choose a PNG or JPEG Braille page image (allowed material only);
2. if you started the service with `OCR_ENGINE_API_KEY` set, paste that key
   into the API key field (it is sent as `X-API-Key` to `/ocr`); otherwise
   leave it empty;
3. click **Run draft OCR**.

The page shows the permanent draft-only banner, confidence, uncertainty
flags with severities, the detected Braille cells, the draft
back-translation, and (collapsed) the full contract JSON — useful when
demonstrating the integration shape. The page is a single self-contained
HTML file with no external resources: it works offline and talks only to
the local service that served it.

A failed image (too small, unreadable, no dots) renders the same way the
contract behaves: empty draft, confidence 0, and clear flags — safe failure
is designed behaviour and worth showing in a demo.

## 3. Suggested five-minute demo script

1. Start the service with the demo page enabled (section 2).
2. In the browser, run `samples\images\sample_01_hello_world.png` — a clean
   synthetic page that decodes with high (but capped, honest) confidence.
3. Run `samples\embossed_images\embossed_06_uneven_light.png` — show that an
   embossed-style photo decodes with **lower** confidence and more flags,
   and say why (honesty caps).
4. Run a deliberately bad image (a tiny or blurry crop) — show the safe
   empty-draft failure with high-severity flags instead of invented text.
5. End on the banner: every result was a draft; a QTVI or Braille-literate
   specialist verifies everything before any real use in InsightEd AI.

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `GET /demo` returns 404 | Set `DEMO_PAGE_ENABLED=true` (env var or `.env`) and restart the service. |
| Page says HTTP 401 | The service has `OCR_ENGINE_API_KEY` set — paste that key into the page's API key field. |
| "Could not reach the local service" | The service is not running (or a different port): `uvicorn app.main:app --port 8000`. |
| Braille shows as `?` in the terminal | The console cannot render U+2800 glyphs. Use Windows Terminal, or a font with Braille support; the CLI already degrades safely instead of crashing. |
| Port 8000 already in use | Start on another port (`--port 8010`) and open `http://localhost:8010/demo`. |
| Empty draft / very low confidence | Expected for hard images. See the image-capture guidance in the README and the collection protocol; flags explain the failure. |

## What this stage deliberately does not do

- No OCR pipeline or contract changes — both tools are viewers.
- No new Python or JavaScript dependencies (the page is hand-written static
  HTML served by the existing FastAPI app).
- No persistence: no upload folders, no history, no saved reports.
- No public exposure or deployment changes: the demo page is 404 unless
  locally enabled, and `/health`, `/version`, `/ocr` behave exactly as
  before.

## Recommended next stage

Use the demo workflow while collecting the first allowed real-capture set
(Stage 3D-G6): the CLI preview is the quickest way to sanity-check a
candidate photo's readability before it enters the formal intake, audit,
and evaluation pipeline.
