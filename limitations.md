# Limitations — insighted-braille-ocr-engine v0.3.0

## Draft-only output (the most important limitation)

**Every output of this engine is a draft transcription only.** It must never
be treated as accurate, certified, or final. InsightEd AI requires — and this
engine assumes — that a **QTVI or explicitly Braille-literate staff member
verifies every transcription** before it is used for teacher feedback or
export. This engine does not claim, and must never be described as having,
certified Braille accuracy. Confidence scores are internal heuristics, not
calibrated probabilities, and a high score is not a substitute for specialist
verification.

## Version 1 scope

- **File types:** PNG and JPEG only. **PDF is not supported** — a PDF upload
  returns a controlled response with a clear flag explaining this.
- **Single page:** every image is treated as one page (`pageResults` always
  contains one entry).
- **6-dot Braille only:** 8-dot Braille is not recognised.
- **Grade 1 (uncontracted) UEB back-translation (fallback):** the built-in
  fallback translator handles letters, the capital sign, the number sign
  with digits, and a small punctuation set (comma, full stop, apostrophe,
  hyphen). **Grade 2 contractions are not interpreted by the fallback** and
  will decode incorrectly (each contraction appears as its literal letter
  or `?`). A `possible_contraction_issue` flag records this on every draft.
- **Grade 2 (contracted) UEB via Liblouis (optional, Stage 3D-I1):** when
  Liblouis is installed and configured with `LIBLOUIS_TABLE=en-ueb-g2.ctb`,
  the engine interprets Grade 2 contractions. The result is still an
  unverified draft requiring specialist verification.
- **Liblouis is optional and not bundled.** When the python `louis` bindings
  and tables are installed, the engine uses Liblouis for back-translation of
  the detected Unicode Braille (Liblouis is not image OCR). When absent, the
  fallback translator is used and a clear uncertainty flag is added. The
  build and tests never require Liblouis.
- **English/UEB orientation:** other languages and codes are untested.

## Image assumptions

- The dot detector handles two input styles: **clear, dark, roughly circular
  dots on a light background** (scans of printed Braille diagrams,
  inkprint-style renders, the bundled synthetic samples), and — since
  Stage 3D-D — **embossed-paper-style photographs** where each raised dot
  appears as a highlight/shadow crescent pair under directional light.
- **Embossed-photo support is simulation-validated, not field-validated.**
  The embossed path decodes all 12 bundled synthetic embossed samples (low
  contrast, shadows, mild skew, paper noise, uneven illumination, wide/tight
  spacing, faint dots, rotation) at CER 0.000, but those simulations are
  cleaner than real photographs of real embossed pages. Expect real-world
  accuracy to be lower, confidence to be capped at 0.82, and heavier
  flagging. Real embossed photograph collection and validation (with proper
  permissions, never live pupil work) remains future work.
- Embossed detection needs **mild directional light** — perfectly flat,
  shadowless lighting leaves raised dots invisible to the camera, and very
  harsh light crushes one side of the pair. Either case degrades to empty or
  heavily flagged output rather than a crash.
- **Resolution floor:** dots must be roughly 6 pixels across or larger, and
  dot rows must be separable. Below that (e.g. tightly spaced Braille in a
  low-resolution photo) the engine deliberately returns an **empty draft
  with a high-severity flag** instead of guessing.
- The grid fitter assumes a **reasonably regular Braille layout** (consistent
  dot pitch and cell spacing) with limited skew (small rotations are
  deskewed at image level, and residual tilt up to ~10° is corrected from
  the dot geometry; heavy skew, page curvature, or perspective distortion
  are not).
- Rare layouts can shift the grid anchor: a line whose cells contain **no
  top-row dots at all**, or a line consisting only of right-column patterns,
  may be decoded with a row/column offset. Such lines are usually accompanied
  by low confidence and `unclear_braille_cell` flags, but not always.
- Interpoint Braille (dots embossed on both sides of the page) is not
  handled.

## Quality and evaluation

- The bundled samples are **synthetic images** (ideal renders plus
  embossed-photo simulations); evaluation numbers on them demonstrate that
  the geometric pipeline round-trips correctly under simulated conditions,
  not that the engine performs well on real pupil work.
- Character Error Rate on real-world embossed photographs should still be
  expected to be substantially higher than on the simulations; confidence
  and flags are calibrated to say so (embossed runs cap at 0.82, fallback
  translation at 0.95).
- **Real-photo validation (Stage 3D-E) is a framework, not a result.** The
  `real_anonymised` dataset is local-only and gitignored; no real
  photographs have been evaluated in this repository yet. Until safe,
  anonymised, approved samples are collected and measured, the engine's
  real-world accuracy is **unknown** — do not extrapolate from the
  synthetic CER 0.000 figures. Real-photo validation measures usefulness
  and correction burden; it does not certify Braille accuracy.
- Do not put pupil-identifying information in sample file names or ground
  truth files — file names appear in evaluation output.

## Security posture

- The service never logs image data, base64 payloads, transcription text,
  file names, task titles, raw task ids, pupil data, or API keys. Task ids
  appear in logs only as a short non-reversible hash.
- Uploads are validated (MIME type, size, pixel count, safe base64 decode)
  and failures always return controlled JSON.
- The optional API key check (`X-API-Key` or `Authorization: Bearer`) is a
  simple shared-secret gate suitable for local/dev integration. Production
  deployment would need transport security (TLS), secret rotation, and
  network-level controls, which are out of scope for v1.
- Images are processed in memory and are not persisted by the engine.

## Future work

- ~~Embossed-dot photograph detection (shadow/highlight pair modelling).~~
  Added in v0.2.0 (Stage 3D-D) — validated on synthetic simulations only;
  validation on real embossed photographs (with permissions, anonymised)
  is the next step.
- PDF intake and multi-page handling.
- ~~Grade 2 (contracted) UEB awareness.~~ Added in v0.5.0 (Stage 3D-I1)
  via optional Liblouis integration. The built-in fallback translator
  remains Grade 1 only.
- Confidence calibration against ground-truth error rates.
- Word-level flags anchored to text spans (`flags[].text` is currently empty
  for whole-image flags and holds a single cell/letter for cell-level flags).
- Interpoint (double-sided) embossed pages.
