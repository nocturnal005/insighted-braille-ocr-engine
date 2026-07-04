# Stage 3D-G0: UKAAF UEB Resource Intake (commit-safe overview)

This document describes how UKAAF UEB resources are acquired and organised
locally to prepare controlled OCR validation. It intentionally contains no
UKAAF file contents, no sample text, no thumbnails, and no local report
contents - those all stay local-only.

> OCR output remains **draft-only**. QTVI/Braille-literate specialist
> verification is mandatory in InsightEd AI before any teacher feedback or
> export. Nothing in this stage changes that.

## Why UKAAF

UKAAF (UK Association for Accessible Formats) publishes the authoritative
UK guidance and demonstration materials for Unified English Braille (UEB):
paired sample documents (print text + braille-view PDF + embossable BRF),
maths/STEM guidance, the UEB Quick Reference Guide, and braille production
standards. For a UK/UEB-focused engine this is the most appropriate source
of controlled, professionally produced braille material - far better
provenance than ad-hoc web images, and with exact print equivalents to use
as ground truth.

## The acquisition pack

A locally supplied ZIP (`UKAAF_UEB_Acquisition_Pack_for_InsightEd.zip`)
contains a verified download manifest (CSV/JSON), PowerShell and Python
downloader scripts, and a resource index. It is extracted to the
local-only workspace below and the downloader fetches each resource
directly from ukaaf.org over HTTPS.

## Local-only storage layout

All external material lives under `_external_sources/`, which is
**gitignored** (as are `resources/`, `external_sample_reports/`, and
`reports/`):

```
_external_sources/ukaaf/acquisition_pack/      extracted pack (manifest, scripts)
_external_sources/ukaaf/downloaded_resources/  downloaded UKAAF files, by category
_external_sources/ukaaf/preview/               preview index HTML + low-res thumbnails
_external_sources/ukaaf/reports/               download audit, candidate catalogue, sample plan
```

## Why these files are not committed

UKAAF materials are copyrighted publications downloaded for local
validation preparation only. Redistribution rights have not been
confirmed, so downloaded documents, generated previews/thumbnails, derived
images, and reports quoting their content must never be committed or
published. Only this overview document is commit-safe.

## Previewing candidates

Open `_external_sources/ukaaf/preview/UKAAF_UEB_Preview_Index.html`
locally. It lists every resource with its category, paired
text/braille/BRF availability, a suitability score (1-5) for OCR sample
generation, intended test use, cautions, and a low-resolution first-page
thumbnail where available. The accompanying local-only reports are the
download audit, the candidate sample catalogue (md/csv/json), and the
controlled OCR sample plan.

## How the resources will support controlled OCR validation

The strongest candidates are the nine UEB sample triplets: a braille-view
PDF (renderable to scanner-equivalent images), a print-text PDF (ground
truth), and an embossable BRF (for real embossed photographs later). The
plan is: clean renders first, then printed/embossed captures under varied
lighting, skew, contrast, and crop conditions, mapped into the existing
`real_anonymised` dataset with full metadata.

One capability caveat is documented up front: the UKAAF UEB samples are
contracted (Grade 2) braille, while the engine currently back-translates
Grade 1 only. Grade 2 captures therefore validate dot detection and cell
geometry (reported separately by the harness), and the pack's uncontracted
Quick Reference BRF provides the Grade 1 material for fair end-to-end CER
measurement.

## What this stage does not prove

Resource intake and clean renders do not prove real-world school accuracy.
That still requires photographed/scanned samples captured under realistic
conditions, evaluated through the `real_anonymised` harness, with results
that apply only to the evaluated sample set. Until then, no real-world OCR
accuracy claim is made.
