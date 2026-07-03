# Stage 3D-F Sample Collection Pack

A practical, step-by-step guide for whoever collects the first real
(anonymised) Braille samples for the validation framework. It complements
the [Real Anonymised Embossed Photograph Validation Protocol](real_photo_validation_protocol.md)
(Stage 3D-E), which remains the authoritative source for anonymisation,
naming, and metadata rules. Read both before collecting anything.

> OCR output remains **draft-only**. QTVI/Braille-literate verification is
> mandatory in InsightEd AI before any teacher feedback or export.
> Real-photo validation measures performance on the approved dataset -
> nothing more.

## 1. Purpose of Stage 3D-F

The engine currently scores CER 0.000 on both bundled synthetic datasets.
That proves the pipeline works on generated images; it does **not** prove
anything about real photographs. Stage 3D-F exists to collect a small,
safe, fully-approved set of real Braille photographs so the existing
audit and evaluation tooling (Stage 3D-E) can produce an honest
real-photo baseline. Current results apply only to the evaluated sample
set - no wider claim is made until real samples are measured.

## 2. What counts as an acceptable sample

- A synthetic Braille printout (embossed or printed for testing)
  photographed by phone.
- A safe demo Braille page photographed under varied conditions
  (lighting, angle, contrast).
- An anonymised Braille-only sample that has been explicitly approved
  for testing.
- A Braille-only crop where all identifiers have been removed (from the
  image *and* from the Braille text itself).
- A non-assessment sample created solely for evaluation purposes.

## 3. What must never be included

- Real pupil homework containing any identifying data.
- Live, current, or future exam/assessment material.
- Any school-branded document (logos, letterheads, stamps).
- Any image where an English print transcription is visible.
- Any image containing a pupil name, school name, date of birth, email
  address, barcode/QR code, or assessment title - in print or in the
  Braille itself.
- Any sample without a recorded permission/approval status.
- Any sample whose ground truth has not been verified by a human reader.

If in doubt, leave it out. A smaller clean dataset beats a larger risky
one.

## 4. How to photograph or scan Braille safely

- **Light**: mild directional light helps embossed dots cast a
  highlight/shadow pair, which is what the detector looks for. Avoid
  harsh glare and very flat, shadowless light.
- **Page**: keep the page flat - no curling corners or folds.
- **Angle**: hold the camera square to the page (perpendicular), not at
  an angle.
- **Resolution**: each Braille dot should span at least ~6 pixels in the
  final image; below that floor the engine cannot reliably separate
  dots. Aim for 8-12 px per dot.
- **Framing**: fill the frame with the Braille region only. Never include
  faces, hands, desks with other documents, screens, or room
  backgrounds.

## 5. How to crop to Braille-only content

Crop the image so that nothing but Braille cells and blank paper
remains. Remove headers, footers, printed margins, hole-punch edges,
stickers, and stamps. If any non-Braille content cannot be cropped out,
do not use the image. The metadata field `crop_quality` must honestly
describe the result; `includes_non_braille` causes the sample to be
skipped by the evaluation.

## 6. How to remove English print transcription from the image

Teachers often write the print translation between Braille lines
(interlining). **Crop or exclude** those lines entirely - do not blur
them. Blurred text can sometimes be recovered, and a blurred region also
confuses dot detection. If interlined print runs through every Braille
line and cannot be excluded, the image cannot be used.

## 7. How to strip EXIF/location metadata

Phone photos embed EXIF data: GPS location, device model, timestamp.

1. Re-export the cropped image (for example, open the crop in an editor
   and save it as a fresh PNG). Re-saving discards the original EXIF.
2. Verify with an EXIF viewer that no GPS, device, or timestamp data
   survives.
3. Record the result in the metadata field `exif_removed: true`.

## 8. How to create the matching ground-truth .txt file

- One file per image: `samples/real_anonymised_ground_truth/<sample_id>.txt`.
- Plain UTF-8 text containing exactly what a Braille-literate person
  reads from the image - verified by a person, never by the engine.
- Line breaks must match the Braille lines in the image.
- The text must be anonymised by the same rules as the image (section 3).
- A starter file is provided at
  `templates/real_sample_ground_truth_template.txt`.

## 9. How to create the metadata JSON

Copy `templates/real_sample_metadata_template.json` to
`samples/real_anonymised_metadata/<sample_id>.json` and fill in every
field:

- `sample_id` - the file stem, e.g. `real_001` (never content-based).
- `source_type` - `synthetic_printout`, `demo_sample`,
  `anonymised_school_sample`, or `other`.
- `braille_type` - `ueb_grade_1`, `ueb_grade_2`, or `unknown`. Grade 2
  is reported separately; the engine does not claim Grade 2 support.
- `capture_method` - `phone_photo`, `scanner`, `screenshot`, or `other`.
- `lighting` / `contrast` / `skew` - honest description of capture
  conditions; these drive the grouped metrics.
- `crop_quality` - `braille_only`, `extra_margin`,
  `includes_non_braille` (skipped), or `unknown`.
- `dot_size_px_estimate` - approximate pixels per dot (floor is ~6).
- `permission_status` - `synthetic`, `anonymised_only`,
  `approved_for_testing`, or `not_approved` (always skipped). Missing or
  invalid metadata is also skipped.
- `ground_truth_verified_by_role` - see section 14.
- `contains_real_pupil_data` - must be `false`; `true` means the sample
  must not exist in this dataset at all.
- `contains_live_assessment_material` - must be `false`, same rule.
- `exif_removed` - `true` once section 7 is done.
- `notes` - free text; must not contain names, schools, or dates.

## 10. How to run the audit

```powershell
python -m app.evaluation.audit_dataset --dataset real_anonymised
```

This is report-only: it checks safety (file naming, metadata presence
and validity, permission status) and completeness (image / ground truth
/ metadata triples) without evaluating anything. Fix every warning it
raises before evaluating.

## 11. How to run the evaluation

```powershell
python -m app.evaluation.run_evaluation --dataset real_anonymised
```

This runs the metadata-gated evaluation: skipped samples are reported
with reasons, approved samples get per-sample and grouped metrics, error
buckets, a confidence-calibration check, and recommendations. It exits
cleanly with a "no samples" message when the folders are empty. It never
prints draft text, ground truth, or image data.

## 12. How to interpret CER/WER/confidence/failure flags

- **CER (character error rate)** and **WER (word error rate)** measure
  the correction burden a specialist would face. CER 0.15 means roughly
  one character in seven needs fixing. Lower is better; 0.000 means the
  draft matched the verified ground truth exactly.
- **Confidence** is a heuristic in [0, 1], not a probability. It is
  capped at 0.82 for embossed-photo runs and 0.95 for fallback
  (non-Liblouis) translation, so it can never signal certainty.
- **Calibration warning** means high-error images are receiving high
  confidence - the confidence score cannot be trusted as a quality
  signal for those capture conditions.
- **Failure flags** bucket what went wrong (for example dot detection,
  cell segmentation, translation), which tells you whether to fix the
  capture (light, crop, resolution) or expect an engine limitation.

## 13. How to decide whether a sample should be excluded

Exclude a sample when any of the following hold:

- It fails any rule in section 3, even after cropping.
- The audit flags its file name, metadata, or permission status and the
  problem cannot be fixed honestly.
- The ground truth cannot be verified by a Braille-literate person.
- Dots are below the ~6 px floor and a better capture is possible -
  recapture rather than keep a known-degraded image, unless you are
  deliberately measuring the degraded condition (record that in
  `notes`).
- Its `braille_type` is uncertain in a way that would contaminate the
  Grade 1 results - mark it `unknown` rather than guessing.

Exclusion is recorded by removing the files or setting
`permission_status: "not_approved"` (which the tooling always skips).
Never "fix" a problem by editing the ground truth to match the OCR
output.

## 14. How to record verification source without identifying a person

Use the metadata field `ground_truth_verified_by_role` with one of:

- `"qtvi"` - verified by a Qualified Teacher of Vision Impairment.
- `"braille_literate_staff"` - verified by Braille-literate staff.
- `"generated_synthetic"` - ground truth is the generation input of a
  synthetic printout (no human reading needed).
- `"unknown"` - verification source not recorded (weakest; avoid).

Record a **role, never a name**. Who specifically verified a sample, and
when, belongs in your own off-repository records - never in this
repository, its metadata, or its file names.

## 15. Draft-only reminder

OCR output remains **draft-only**. Nothing in this pack, and no
evaluation result however good, certifies Braille accuracy or removes
the need for QTVI/Braille-literate specialist verification in InsightEd
AI before teacher feedback or export. Current results apply only to the
evaluated sample set.

## Recommended sample targets (Part F)

Use `templates/real_sample_checklist.md` per sample as you collect.

### Minimum baseline

- 10 approved Braille-only images.
- Each with human-verified ground truth and complete metadata.
- Enough to produce a first honest real-photo baseline - not enough for
  any generalised claim.

### Better baseline

- 30-50 samples.
- Mix of phone photos and scans.
- Varied lighting and contrast; both mild skew and clean flat captures.
- Content including numbers and punctuation, plus multi-line pages.

### Strong future validation

- 100+ samples collected over time.
- Separate development and held-out validation sets.
- Grade 1 and Grade 2 results reported separately (Grade 2 remains
  unsupported and is measured only for transparency).
- QTVI/Braille-literate review of the actual correction burden, not just
  the metric numbers.
