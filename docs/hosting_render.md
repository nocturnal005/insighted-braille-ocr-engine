# Hosting the Braille OCR engine on Render

The deployed InsightEd AI app (Vercel) cannot reach `localhost:8000`. To make
Braille transcription work on the hosted app, the engine must run somewhere
public. This repo ships a `Dockerfile` and `render.yaml` for that.

## 1. Deploy the engine

1. Render Dashboard → **New** → **Blueprint** → connect this repository.
2. Render reads `render.yaml`, builds the Dockerfile, and starts the service.
3. Note the URL: `https://<service-name>.onrender.com`.
4. In the service's **Environment** tab, copy the generated
   `OCR_ENGINE_API_KEY` value (Render created it for you).

Verify it is up:

```
curl https://<service-name>.onrender.com/health          # -> 200
curl https://<service-name>.onrender.com/ocr             # -> 401 (key required)
```

## 2. Point the app at it

In the InsightEd AI project (Vercel → Settings → Environment Variables):

| Variable | Value |
| --- | --- |
| `BRAILLE_OCR_PROVIDER` | `external_braille_ocr` |
| `BRAILLE_OCR_ENDPOINT` | `https://<service-name>.onrender.com/ocr` |
| `BRAILLE_OCR_API_KEY` | the `OCR_ENGINE_API_KEY` value from Render |
| `AI_MODE` | `real` |

Redeploy the app so the new variables take effect.

## 3. Cold starts (free plan)

Free Render instances sleep after ~15 minutes idle and take ~50s to wake. The
app's Braille OCR timeout is 30s (`BRAILLE_OCR_TIMEOUT_MS`), so **the first
request after a sleep will fail**. Options:

- **Before a demo**, wake it: `curl https://<service>.onrender.com/health` and
  wait for a 200. It then stays warm while in use.
- **For reliability**, move the service to the `starter` plan (always-on) by
  changing `plan: free` to `plan: starter` in `render.yaml`.
- Or raise `BRAILLE_OCR_TIMEOUT_MS` in the app above the cold-start time.

## Data protection

The engine stores nothing: images are decoded in memory, processed, and
discarded. Logs are metadata only (counts, durations, confidence) — never
image data, transcription text, file names, or raw task ids.

Even so, hosting moves pupil photographs off the school's machines to a
third-party host. **Keep `ALLOW_REAL_PUPIL_DATA=false` in the app** (its
default) until that transfer is covered by the school's data-protection
approval. With it false, the app blocks pupil-linked tasks before any image
leaves it, and non-pupil-linked material (demo/sample pages) still transcribes
normally — which is what a demo should use.

## What this does not change

Hosting fixes reachability, not accuracy. Output remains a draft that a QTVI
or Braille-literate specialist must verify; the engine never claims certified
Braille accuracy. See `limitations.md`.
