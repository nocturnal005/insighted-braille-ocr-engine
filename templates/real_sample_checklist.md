# Real Sample Pre-Submission Checklist

Complete this checklist BEFORE adding a real anonymised sample to the
local-only sample folders. If any box cannot be ticked, do not add the sample.

Detailed rules live in:

- docs/real_photo_validation_protocol.md
- docs/stage_3d_f_sample_collection_pack.md

## Where files go (local-only, gitignored)

- Image: samples/real_anonymised_images/<stem>.png
- Ground truth: samples/real_anonymised_ground_truth/<stem>.txt
- Metadata: samples/real_anonymised_metadata/<stem>.json

All three files MUST share the same <stem> (e.g.
real_001_clean_flat_good_light). These folders are gitignored and must never
be committed or synced anywhere.

Safe file-name style: real_NNN_<condition tags>.png, e.g.
real_001_clean_flat_good_light.png. Unsafe: anything containing
pupil/student/school/exam/test/assessment/homework/name, personal names,
dates, emails, or spaces.

## Allowed metadata values

The JSON template (templates/real_sample_metadata_template.json) cannot hold
comments, so the allowed values for each field are documented here.

| Field | Allowed values |
|---|---|
| sample_id | Placeholder id matching the file stem prefix, e.g. "real_001" - never a real name or title |
| source_type | synthetic_printout \| anonymised_school_sample \| demo_sample \| other |
| braille_type | ueb_grade_1 \| ueb_grade_2 \| unknown |
| capture_method | phone_photo \| scanner \| screenshot \| other |
| lighting | good_even \| directional \| shadowed \| low_light \| unknown |
| contrast | high \| medium \| low \| unknown |
| skew | none \| mild \| moderate \| severe \| unknown |
| crop_quality | braille_only \| extra_margin \| includes_non_braille \| unknown |
| dot_size_px_estimate | a number (approximate dot diameter in pixels) or null |
| permission_status | synthetic \| anonymised_only \| approved_for_testing \| not_approved |
| ground_truth_verified_by_role | qtvi \| braille_literate_staff \| generated_synthetic \| unknown |
| contains_real_pupil_data | must be false to use the sample |
| contains_live_assessment_material | must be false to use the sample |
| exif_removed | should be true |
| notes | free text - must contain no names, schools, dates, emails, or other identifiers |

Evaluation gating reminders:

- permission_status "not_approved", or missing/invalid metadata, means the
  sample is SKIPPED by evaluation.
- crop_quality "includes_non_braille" means the sample is SKIPPED.
- braille_type "ueb_grade_2" is reported separately - Grade 2 is not
  supported by the engine.

## Checklist

Tick every box before adding the sample:

- [ ] Image is cropped to the Braille region only (no surrounding page
      content, hands, desks, or labels).
- [ ] Any English print transcription visible on the page has been excluded
      from the crop.
- [ ] No pupil, school, teacher, or assessment identifiers appear anywhere:
      not in the image, not in the ground truth, not in the file name, and
      not in the metadata notes.
- [ ] EXIF and other embedded metadata have been stripped from the image,
      and exif_removed is set to true.
- [ ] File name follows the safe style real_NNN_<condition tags>.png (no
      pupil/student/school/exam/test/assessment/homework/name words, no
      personal names, no dates, no emails, no spaces).
- [ ] Matching <stem>.txt ground truth and <stem>.json metadata files exist,
      with the same <stem> as the image.
- [ ] Ground truth was transcribed and verified by a human Braille-literate
      reader (or is generated_synthetic for synthetic samples), and
      ground_truth_verified_by_role is set accordingly.
- [ ] permission_status is set correctly for how the sample was obtained.
      Remember: not_approved samples are skipped by evaluation.
- [ ] If the sample is UEB Grade 2 (contracted Braille), braille_type is set
      to ueb_grade_2 so it is reported separately (Grade 2 not supported).
- [ ] contains_real_pupil_data is false and
      contains_live_assessment_material is false.
- [ ] Metadata file is valid JSON (parseable, no trailing commas, no
      comments).

## Draft-only reminder

OCR output from this engine is DRAFT-ONLY. Verification by a QTVI or
Braille-literate specialist is mandatory before any output is used
downstream. Adding a sample here is for engine evaluation only and never
replaces human verification.
