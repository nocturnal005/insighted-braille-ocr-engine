# Real Anonymised Embossed Photograph Validation Protocol (Stage 3D-E)

This protocol governs how real-world Braille photographs/scans may be used
to validate the engine. It exists to measure real performance **safely** —
without ever exposing pupil data — and to keep claims honest.

> The engine remains a **draft OCR tool**. Real-photo validation measures
> usefulness and correction burden; it does **not** certify Braille
> accuracy. QTVI/Braille-literate specialist verification in InsightEd AI
> remains mandatory before teacher feedback or export.

## 1. What images are allowed

- **Synthetic printouts**: Braille rendered, embossed or printed for testing,
  then photographed/scanned — no school origin at all. Safest option.
- **Demo samples**: pages produced specifically for demonstrations.
- **Anonymised school samples**: only when *all* of the following hold:
  1. explicit approval has been obtained from the school's data-protection
     lead **before** the image is used (record who approved and when — in
     your own records, not in this repository);
  2. the page contains **no pupil name, school name, date, class, or
     assessment title** — in Braille or print;
  3. the image is cropped to the **Braille-only region**;
  4. it is not live or future assessment material.

## 2. What images are NOT allowed

- Anything containing a pupil's name or handwriting-style signature —
  including *inside the Braille itself*. If the Braille text names a pupil,
  the page cannot be used, cropped or not.
- School names, logos, letterheads, stamps, or barcodes/QR codes.
- Live, current, or future exam/assessment material.
- Photographs that include faces, hands, desks with other documents, screens,
  or room backgrounds.
- Anything you do not have recorded permission to use.

## 3. How to anonymise an image

1. **Crop to the Braille-only region.** Remove page headers, footers,
   margins with printed text, and any interlined English transcription
   (teachers often write the print translation between Braille lines — crop
   or exclude those lines entirely; do not blur them, exclusion is safer).
2. **Check the Braille content itself.** Have a Braille reader confirm the
   text contains no names or identifying details before use.
3. **Remove barcodes, stamps, and stickers** by cropping (not by blurring —
   blurred regions can sometimes be recovered).
4. **Strip file metadata.** Phone photos embed EXIF data (GPS location,
   device, timestamp). Re-export the image (e.g. re-save the cropped PNG) so
   no EXIF survives. Verify with an EXIF viewer.
5. **Re-name the file safely** (section 5) — the original camera or school
   file name must not survive.

## 4. Ground truth files

- One `samples/real_anonymised_ground_truth/<sample_id>.txt` per image,
  containing exactly the text a Braille-literate person reads from the
  image — verified by a person, not by the engine.
- Plain UTF-8 text, line breaks matching the Braille lines.
- The ground truth must be anonymised by the same rules as the image.

## 5. File naming

Use `real_NNN_<condition tags>.png` — describe the *capture conditions*,
never the content or origin.

**Safe examples:**

```
real_001_clean_flat_good_light.png
real_002_low_contrast_angle_light.png
real_003_mild_skew_shadow.png
```

**Unsafe examples (the audit flags these):**

```
pupil_name_homework.png
schoolname_year11_exam.png
real_student_assessment.png
```

Avoid: pupil/student/school/exam/test/assessment/homework/name, personal
names, dates of birth, email-like strings, and spaces in file names.

## 6. Metadata files

One `samples/real_anonymised_metadata/<sample_id>.json` per image:

```json
{
  "sample_id": "real_001",
  "source_type": "anonymised_school_sample | synthetic_printout | demo_sample | other",
  "braille_type": "ueb_grade_1 | ueb_grade_2 | unknown",
  "capture_method": "phone_photo | scanner | screenshot | other",
  "lighting": "good_even | directional | shadowed | low_light | unknown",
  "contrast": "high | medium | low | unknown",
  "skew": "none | mild | moderate | severe | unknown",
  "crop_quality": "braille_only | extra_margin | includes_non_braille | unknown",
  "dot_size_px_estimate": 9,
  "permission_status": "synthetic | anonymised_only | approved_for_testing | not_approved",
  "notes": "free text - MUST NOT contain names, schools, or dates"
}
```

Rules enforced by the tooling:

- `permission_status: "not_approved"` → the sample is **always skipped**.
- Missing/invalid metadata → skipped (permission unknown).
- `crop_quality: "includes_non_braille"` → skipped (crop it properly).
- `braille_type: "ueb_grade_2"` → evaluated but **reported separately**;
  the engine does not claim Grade 2 support.

## 7. Local-only storage

The three `samples/real_anonymised_*` folders are **gitignored**. Real
photographs, ground truth, and metadata stay on the local machine by
default. Do not commit them unless a file is unambiguously safe (e.g. a
synthetic printout you created) *and* its addition is deliberately reviewed.

## 8. Workflow

```powershell
# 1. add image + ground truth + metadata, then audit (report-only):
python -m app.evaluation.audit_dataset --dataset real_anonymised

# 2. fix any warnings (rename, crop, complete metadata), re-audit

# 3. evaluate:
python -m app.evaluation.run_evaluation --dataset real_anonymised
```

The evaluation prints per-sample metrics (by sample id, never file paths or
text), grouped results by capture conditions, error buckets, a confidence
calibration check, failure-mode flags, and recommendations. It never prints
draft text, ground truth, or image data.

## 9. Interpreting results

- **CER/WER** measure the correction burden a specialist would face — a
  CER of 0.15 means roughly one character in seven needs fixing.
- **Confidence** is a heuristic in [0, 1], capped at 0.82 for embossed-mode
  runs and 0.95 for fallback translation. The calibration section warns if
  high-error images are receiving high confidence.
- **Synthetic success does not equal real-world accuracy.** The bundled
  synthetic datasets evaluate at CER 0.000; real photographs are expected
  to score worse. That gap is exactly what this protocol measures.
