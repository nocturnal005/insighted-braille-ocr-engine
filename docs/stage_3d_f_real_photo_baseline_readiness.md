# Stage 3D-F Real-Photo Baseline Readiness Report

Engine version 0.3.0. This report contains no real pupil data, no school
data, and no sample content - it is safe to commit.

> OCR output remains **draft-only**. QTVI/Braille-literate verification
> is mandatory in InsightEd AI before any teacher feedback or export.

## Status: framework READY, dataset EMPTY

The Stage 3D-E validation framework is **ready**:

- `python -m app.evaluation.audit_dataset --dataset real_anonymised`
  runs and exits cleanly.
- `python -m app.evaluation.run_evaluation --dataset real_anonymised`
  runs and exits cleanly, reporting the "no samples" message on an empty
  dataset.

**No approved real samples are present.** The three local-only,
gitignored folders (`samples/real_anonymised_images/`,
`samples/real_anonymised_ground_truth/`,
`samples/real_anonymised_metadata/`) contain only `.gitkeep` files.

## No real-world accuracy claim

**No real-world OCR accuracy claim can be made yet.** The engine scores
CER 0.000 on both bundled synthetic datasets, but synthetic-dataset
success does **not** prove real-world accuracy. Real photographs are
expected to score worse; measuring that gap is the whole point of the
real-photo baseline. When results exist, they will apply only to the
evaluated sample set.

## Baseline evidence

Verified on 2026-07-03 against engine version 0.3.0.

| Check | Result |
| --- | --- |
| `python -m pytest` | 100/100 passed |
| Original synthetic dataset | mean CER 0.000, WER 0.000, confidence 0.950 |
| Embossed synthetic dataset | mean CER 0.000, WER 0.000, confidence 0.820 |
| `real_anonymised` dataset | empty (folders hold only `.gitkeep`); audit and evaluation exit cleanly with the "no samples" message |
| Real-photo baseline | **not yet established** - no approved samples |

Note: confidence is a heuristic, deliberately capped at 0.82 for
embossed-photo runs and 0.95 for fallback (non-Liblouis) translation.

## Next operational step: safe sample collection

Collect the first approved real samples following:

- [Stage 3D-F Sample Collection Pack](stage_3d_f_sample_collection_pack.md) -
  practical collection, capture, anonymisation, and evaluation steps.
- [Real Anonymised Embossed Photograph Validation Protocol](real_photo_validation_protocol.md) -
  authoritative anonymisation, naming, and metadata rules.

## Minimum recommended dataset and checklist

- **Minimum baseline**: 10 approved Braille-only images, each with
  human-verified ground truth and complete metadata.
- **Better baseline**: 30-50 samples across varied capture conditions
  (see the collection pack's "Recommended sample targets").
- Work through `templates/real_sample_checklist.md` for every sample;
  starter files are at `templates/real_sample_metadata_template.json`
  and `templates/real_sample_ground_truth_template.txt`.

## Running the baseline once samples exist

```powershell
# 1. Report-only safety and completeness audit - fix all warnings first:
python -m app.evaluation.audit_dataset --dataset real_anonymised

# 2. Metadata-gated evaluation - grouped metrics, error buckets,
#    confidence-calibration check, recommendations:
python -m app.evaluation.run_evaluation --dataset real_anonymised
```

Samples with `permission_status: "not_approved"`, missing/invalid
metadata, or `crop_quality: "includes_non_braille"` are skipped;
`ueb_grade_2` samples are reported separately (Grade 2 is not
supported).

## Closing reminder

The engine is and remains a **draft-only** OCR tool. Real-photo
validation measures performance on the approved dataset; it does not
certify Braille accuracy. QTVI/Braille-literate specialist verification
is mandatory in InsightEd AI before any teacher feedback or export.
