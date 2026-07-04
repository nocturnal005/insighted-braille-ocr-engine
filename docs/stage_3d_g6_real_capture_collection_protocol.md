# Stage 3D-G6: Real-capture rawBraille collection protocol (commit-safe)

A practical, safety-first protocol for collecting real photographed or scanned
Braille samples for **cell-level (rawBraille) validation**, and for proving the
intake workflow works before any real evaluation claims are made. This is a
protocol and dry-run stage: no OCR logic changes, no new accuracy claims.

> OCR output remains **draft-only**. QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. Grade 2 English transcription remains **out of scope**: real-capture
> datasets are evaluated at the cell level only, never with English CER/WER.

## 1. What samples are allowed

Only material that is safe by construction:

- **Non-pupil physical Braille samples created specifically for testing** —
  e.g. pages embossed or produced by the project owner for OCR validation.
- **Public-domain or self-authored Braille practice material.**
- **Test sheets produced by the project owner** specifically for this project.
- **Anonymised Braille-only crops with explicit permission** and no school,
  pupil, or assessment identifiers of any kind.

## 2. What samples are forbidden

Never collect, photograph, store, or evaluate:

- real pupil work of any kind;
- school assessment material;
- live homework or exam content;
- anything showing identifiable pupil names or school names;
- confidential SEN records or support-plan material;
- copyrighted UKAAF source PDFs/BRFs/DOC/DOCX files (local-only reference
  material must never become "real capture" input or be committed);
- copyrighted book or worksheet pages;
- downloaded external resources committed into the repo;
- any image containing faces, handwriting, names, labels, class names, school
  logos, or camera/file metadata that could identify a pupil or school.

If in doubt, the sample is forbidden. The readiness audit screens file names
and metadata, but pattern checks are a backstop — the human collecting the
sample is responsible for these rules.

## 3. How to photograph or scan safely

- Work only with allowed material (section 1) on a clean, plain background.
- Fill the frame with the Braille area; hold the camera flat (parallel to the
  page) to minimise skew and perspective.
- Prefer even, diffuse light. For embossed (unprinted) Braille, gentle side
  light makes dots readable; avoid harsh single-point shadows.
- Avoid anything else in frame: no hands, faces, desks with papers, labels,
  or logos.
- Use PNG or JPEG. Aim for dots at least ~8 px across in the final image
  (the engine's readable floor is ~6 px; below it pages fail safely).
- Strip or avoid embedded metadata that could identify a person, device
  location, or school (e.g. GPS EXIF). Scanners are preferable to phones for
  this reason.

## 4. Cropping

Crop to the **Braille-only region** before intake: no page headers, margins
with print text, stickers, or identifying marks. `crop_quality` equivalents
from the real-photo protocol apply: a crop that includes non-Braille content
is not accepted.

## 5. Anonymous naming

Sample IDs must be anonymous and mechanical, e.g.:

```
rb_capture_001
rb_capture_002_low_light
```

Never use names, initials, school references, class names, dates of birth, or
words like "pupil", "homework", "exam", "assessment". The audit withholds and
blocks IDs that match unsafe patterns, and evaluation reports print safe
labels only.

## 6. Intake layout and required files

Real-capture samples live in the local-only, gitignored intake folders
(`.gitkeep` placeholders are the only committed content):

```
samples/real_rawbraille_images/    rb_capture_001.png
samples/real_rawbraille_expected/  rb_capture_001.braille
samples/real_rawbraille_metadata/  rb_capture_001.json
```

Each sample needs all three files. Templates with safe placeholder values:

- metadata: `docs/templates/real_capture_rawbraille_metadata.template.json`
- manifest entry (optional dataset manifest):
  `docs/templates/real_capture_rawbraille_manifest.template.json`

Required metadata includes `permission_status: "approved_for_testing"`
(anything else is skipped), `contains_real_pupil_data: false`,
`contains_live_assessment_material: false`, and
`requires_english_transcript: false`. A consent/safety note stating who made
or approved the sample is required in the metadata or manifest.

## 7. Expected `.braille` ground truth (cell-level)

The expected file contains the page's Braille **cells** as Unicode Braille
(U+2800 block), one line per embossed line, spaces for blank cells — exactly
the engine's `rawBraille` conventions:

- Produce it from the *source* of the page, never by trusting OCR output:
  - if the page was embossed from a BRF you authored, decode the BRF bytes
    with the project's Braille-ASCII codec (`app/evaluation/braille_ascii.py`);
  - if the page was written manually, have a Braille-literate person record
    the cells and a second check verify them.
- Strip leading indentation, drop blank separator lines, and cap runs of more
  than 5 blank cells (see `braille_ascii.expected_rawbraille` — the same
  normalisation used for the controlled dataset).
- The file must contain **cells only**. English text is not accepted.

## 8. Why English Grade 2 transcripts are not accepted

The engine back-translates Grade 1 only. An English transcript of a Grade 2
page cannot be scored fairly (it would measure a deliberately unimplemented
capability), and storing transcripts invites exactly the kind of content the
safety rules exclude. The evaluation schema is structurally locked to
`rawbraille_cell_level`; the audit **blocks** `.txt` files in the expected
folder and any `requires_english_transcript` marker.

## 9. Running the readiness audit

Before any evaluation, audit the intake (reports only, never modifies files;
exit code 2 on blocking issues):

```bash
python -m app.evaluation.audit_rawbraille_dataset --dataset real_capture_grade2_raw
# with an optional dataset manifest:
python -m app.evaluation.audit_rawbraille_dataset --dataset real_capture_grade2_raw \
    --manifest samples/real_rawbraille_metadata/manifest.json
```

An empty intake reports `verdict: EMPTY (expected until samples are added)`.
Fix every BLOCK line before evaluating; review every WARN line.

## 10. Running the real-capture evaluation

Only after the audit says READY:

```bash
python -m app.evaluation.run_rawbraille_evaluation --dataset real_capture_grade2_raw
# optional local-only sanitized report:
python -m app.evaluation.run_rawbraille_evaluation --dataset real_capture_grade2_raw \
    --write-report reports/real_capture_rawbraille/baseline.json
```

Reports are local-only (`reports/` is gitignored) and contain metrics and safe
labels only — never Braille content, images, or draft text.

## 11. Reporting results honestly

- Results describe **the evaluated sample set only**. Never generalise to
  "real-world accuracy", certified accuracy, or school-deployment readiness.
- Real-capture results are reported **separately** from controlled-render
  baselines — the report's `capture_type` labelling enforces this; keep the
  separation in any human-written summary too.
- Failed or low-confidence samples are findings, not embarrassments: safe
  failure with honest flags is designed behaviour.
- Every report and summary keeps the draft-only wording: OCR output requires
  QTVI/Braille-literate specialist verification before any use in teacher
  feedback or export.

## Recommended next stage

Collect a first small allowed set (e.g. 5–10 self-authored test sheets across
lighting/skew conditions), run the audit to READY, then perform the first
real-capture rawBraille baseline and report it strictly against this protocol.
