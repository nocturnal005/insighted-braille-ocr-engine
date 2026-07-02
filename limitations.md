# Limitations — insighted-braille-ocr-engine v0.1.0

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
- **Grade 1 (uncontracted) UEB back-translation:** the built-in fallback
  translator handles letters, the capital sign, the number sign with digits,
  and a small punctuation set (comma, full stop, apostrophe, hyphen).
  **Grade 2 contractions are not interpreted** and will decode incorrectly
  (each contraction appears as its literal letter or `?`). A
  `possible_contraction_issue` flag records this on every draft.
- **Liblouis is optional and not bundled.** When the python `louis` bindings
  and tables are installed, the engine uses Liblouis for back-translation of
  the detected Unicode Braille (Liblouis is not image OCR). When absent, the
  fallback translator is used and a clear uncertainty flag is added. The
  build and tests never require Liblouis.
- **English/UEB orientation:** other languages and codes are untested.

## Image assumptions

- The dot detector targets **clear, dark, roughly circular dots on a light
  background** — scans of printed Braille diagrams, inkprint-style renders,
  or the bundled synthetic samples.
- **Photographs of embossed Braille paper are not yet reliable.** Embossed
  dots appear as low-contrast shadow/highlight pairs, not dark dots. Such
  images will usually produce empty or heavily flagged output. This is the
  main planned area of future work.
- The grid fitter assumes a **reasonably regular Braille layout** (consistent
  dot pitch and cell spacing) with limited skew (small rotations are
  deskewed; heavy skew, curvature, or perspective distortion are not).
- Rare layouts can shift the grid anchor: a line whose cells contain **no
  top-row dots at all**, or a line consisting only of right-column patterns,
  may be decoded with a row/column offset. Such lines are usually accompanied
  by low confidence and `unclear_braille_cell` flags, but not always.
- Interpoint Braille (dots embossed on both sides of the page) is not
  handled.

## Quality and evaluation

- The bundled samples are **synthetic best-case images**; evaluation numbers
  on them demonstrate that the geometric pipeline round-trips correctly, not
  that the engine performs well on real pupil work.
- Character Error Rate on real-world images should be expected to be high in
  v1 until dedicated embossed-dot detection is added.
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

## Not in version 1 (future work)

- Embossed-dot photograph detection (shadow/highlight pair modelling).
- PDF intake and multi-page handling.
- Grade 2 (contracted) UEB awareness in the fallback translator.
- Confidence calibration against ground-truth error rates.
- Word-level flags anchored to text spans (`flags[].text` is currently empty
  for whole-image flags and holds a single cell/letter for cell-level flags).
