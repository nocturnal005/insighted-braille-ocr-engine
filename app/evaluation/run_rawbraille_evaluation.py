"""Cell-level rawBraille evaluation harness (Stage 3D-G3, hardened in 3D-G5).

Usage:
    python -m app.evaluation.run_rawbraille_evaluation --dataset ukaaf_grade2_raw
    python -m app.evaluation.run_rawbraille_evaluation --dataset real_capture_grade2_raw
    python -m app.evaluation.run_rawbraille_evaluation --dataset ukaaf_grade2_raw \\
        --write-report reports/ukaaf_g3_rawbraille/ukaaf-grade2-rawbraille-report.json

Runs the OCR pipeline on a rawBraille dataset and compares the returned
``rawBraille`` to expected Braille cells. Scores the VISUAL pipeline only:

    rawBraille CER, cell error rate (space-agnostic), line-count mismatch,
    cell-count mismatch, exact-sample-match rate, line-reconstruction accuracy,
    confidence, flags, processing time, repeatability of rawBraille.

Every dataset is self-describing (see ``rawbraille_dataset.DATASETS``): reports
and console output state the dataset's capture type (controlled render,
synthetic, or real capture) so controlled-render results can never be mistaken
for real-capture results, and vice versa.

When Liblouis is available with a Grade 2 (contracted) table, supplementary
English CER/WER is computed by back-translating the expected rawBraille
through Liblouis to derive a reference English text. These metrics are
supplementary: cell-level rawBraille remains the primary evaluation, and
English scores are reported separately with a caveat that they depend on
both the visual pipeline and Liblouis translation quality. When Liblouis
Grade 2 is not available, English scoring is skipped and reports carry
``english_cer_wer_computed: false``.

Prints and writes metrics only. Never prints or stores expected or predicted
rawBraille, draft text, image data, or unsafe file names. An empty/missing
dataset exits cleanly (sample folders are local-only).
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

from app.core.config import get_settings
from app.evaluation.metrics import character_error_rate, normalise_text, word_error_rate
from app.evaluation.rawbraille_dataset import (
    DATASET_NAME,
    DATASETS,
    RawBrailleDatasetSpec,
    discover_dataset,
    get_spec,
)
from app.evaluation.rawbraille_metrics import sample_metrics
from app.models.requests import OcrRequest
from app.ocr.pipeline import run_ocr
from app.translation.liblouis_adapter import (
    is_grade2_table,
    liblouis_available,
    liblouis_back_translate,
)

MIME_BY_EXTENSION = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

REPORT_SCHEMA_VERSION = "1.1"  # 1.1: Stage 3D-G5 dataset descriptor + scope fields
REPORT_NOTE = (
    "Cell-level (rawBraille) validation only. This is NOT English Grade 2 "
    "transcription accuracy - the engine does not interpret contractions. "
    "OCR output is draft-only; QTVI/Braille-literate specialist verification "
    "remains mandatory."
)

_CAPTURE_BANNERS = {
    "controlled_render": (
        "CONTROLLED RENDER dataset: locally rendered images, not photographs. "
        "Results do NOT demonstrate real-world capture accuracy."
    ),
    "synthetic": (
        "SYNTHETIC dataset: generated test material. Results do NOT "
        "demonstrate real-world capture accuracy."
    ),
    "real_capture": (
        "REAL CAPTURE dataset: photographed/scanned physical Braille. Report "
        "results separately from controlled-render baselines."
    ),
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _run_id() -> str:
    return "rawbraille-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dataset_descriptor(spec: RawBrailleDatasetSpec) -> dict:
    """Safe, self-describing dataset block for reports and console output."""
    return {
        "name": spec.name,
        "dataset_category": "rawbraille_validation",
        "capture_type": spec.capture_type,
        "source_type": spec.source_type,
        "grade_mode": spec.grade_mode,
        "evaluation_mode": spec.evaluation_mode,
        "description": spec.description,
    }


def _english_scoring_available() -> bool:
    """True when Liblouis is available and configured for Grade 2."""
    settings = get_settings()
    return (
        settings.liblouis_enabled
        and is_grade2_table(settings.liblouis_table)
        and liblouis_available()
    )


def _derive_english_reference(expected_rawbraille: str) -> str | None:
    """Back-translate expected rawBraille through Liblouis Grade 2 to derive
    an English reference text. Returns None if unavailable."""
    settings = get_settings()
    result = liblouis_back_translate(expected_rawbraille, settings.liblouis_table)
    if result is None:
        return None
    return normalise_text(result)


def _evaluate(sample, runs: int, english_scoring: bool = False) -> dict:
    """Run the pipeline `runs` times on one sample; cell-level metrics plus
    optional English CER/WER when Liblouis Grade 2 is available."""
    expected = sample.expected_rawbraille()
    mime = MIME_BY_EXTENSION[sample.image_path.suffix.lower()]
    data_url = (
        f"data:{mime};base64,"
        + base64.b64encode(sample.image_path.read_bytes()).decode("ascii")
    )
    request = OcrRequest(
        taskId=f"rawbraille-{sample.sample_id}",
        title="rawbraille-cell-level-validation",
        fileName=f"{sample.sample_id}{sample.image_path.suffix.lower()}",
        mimeType=mime,
        dataUrl=data_url,
    )

    raw_outputs: list[str] = []
    draft_texts: list[str] = []
    durations_ms: list[float] = []
    confidence = 0.0
    flag_categories: set[str] = set()
    errored = False
    for run_index in range(max(1, runs)):
        started = perf_counter()
        try:
            response = run_ocr(request)
        except Exception:
            errored = True
            break
        durations_ms.append((perf_counter() - started) * 1000)
        raw_outputs.append(response.rawBraille or "")
        draft_texts.append(normalise_text(response.draftText))
        confidence = response.confidence
        if run_index == 0:
            flag_categories = {f.category for f in response.flags}

    predicted = raw_outputs[0] if raw_outputs else ""
    metrics = sample_metrics(expected, predicted)
    repeatable = len(set(raw_outputs)) <= 1 if raw_outputs else False
    metrics.update(
        {
            "confidence": confidence,
            "flag_categories": flag_categories,
            "ms": _mean(durations_ms),
            "repeatable": repeatable,
            "failed": errored or predicted == "",
            "category": sample.category,
            "variant": sample.variant,
            "capture_type": sample.capture_type,
            "label": sample.safe_label,
        }
    )

    if english_scoring and draft_texts:
        english_ref = _derive_english_reference(expected)
        if english_ref is not None:
            hypothesis = draft_texts[0]
            metrics["english_cer"] = character_error_rate(english_ref, hypothesis)
            metrics["english_wer"] = word_error_rate(english_ref, hypothesis)

    return metrics


def _hardest_categories(rows: list[dict]) -> list[tuple[str, float, int]]:
    """Mean cell error rate per category, worst first."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row["cell_error_rate"])
    ranked = sorted(
        ((cat, _mean(v), len(v)) for cat, v in by_cat.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked


def _confidence_summary(rows: list[dict]) -> dict:
    scores = [r["confidence"] for r in rows]
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": round(_mean(scores), 4),
        "median": round(median(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }


def _print_row(row: dict, english: bool = False) -> None:
    line = (
        f"{row['label']:<40} {row['cell_error_rate']:>7.3f} "
        f"{row['rawbraille_cer']:>7.3f} {row['line_count_mismatch']:>5} "
        f"{row['cell_count_mismatch']:>5} {str(row['exact_sample_match']):>6} "
        f"{row['confidence']:>6.3f} {row['ms']:>7.1f}"
    )
    if english and "english_cer" in row:
        line += f" {row['english_cer']:>7.3f} {row['english_wer']:>7.3f}"
    print(line)


def _build_report(
    spec: RawBrailleDatasetSpec,
    rows: list[dict],
    samples: int,
    skipped: int,
    run_id: str,
    english_scoring: bool = False,
) -> dict:
    failed = [r for r in rows if r["failed"]]
    exact = [r for r in rows if r["exact_sample_match"]]
    flag_counts: Counter[str] = Counter()
    for row in rows:
        flag_counts.update(row["flag_categories"])
    english_rows = [r for r in rows if "english_cer" in r]
    report: dict = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "engine_version": get_settings().service_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset": _dataset_descriptor(spec),
        "english_cer_wer_computed": english_scoring and bool(english_rows),
        "grade2_english_transcription": (
            "supplementary_via_liblouis" if english_scoring and english_rows
            else "out_of_scope"
        ),
        "note": REPORT_NOTE,
        "capture_type_note": _CAPTURE_BANNERS[spec.capture_type],
        "counts": {
            "samples": samples,
            "evaluated": len(rows),
            "skipped": skipped,
            "failed": len(failed),
        },
        "summary": {
            "mean_cell_error_rate": round(_mean([r["cell_error_rate"] for r in rows]), 4),
            "median_cell_error_rate": round(
                median(r["cell_error_rate"] for r in rows), 4
            ),
            "mean_rawbraille_cer": round(_mean([r["rawbraille_cer"] for r in rows]), 4),
            "exact_sample_match_rate": round(len(exact) / max(len(rows), 1), 4),
            "line_count_mismatch_rate": round(
                sum(1 for r in rows if r["line_count_mismatch"]) / max(len(rows), 1), 4
            ),
            "cell_count_mismatch_rate": round(
                sum(1 for r in rows if r["cell_count_mismatch"]) / max(len(rows), 1), 4
            ),
            "mean_line_reconstruction_accuracy": round(
                _mean([r["line_reconstruction_accuracy"] for r in rows]), 4
            ),
            "mean_ms": round(_mean([r["ms"] for r in rows]), 1),
            "repeatable_all": all(r["repeatable"] for r in rows),
        },
        "confidence_summary": _confidence_summary(rows),
        "hardest_categories": [
            {"category": cat, "mean_cell_error_rate": round(cer, 4), "n": n}
            for cat, cer, n in _hardest_categories(rows)
        ],
        "flag_categories": [
            {"category": category, "samples": count}
            for category, count in flag_counts.most_common()
        ],
    }
    if english_rows:
        report["english_summary"] = {
            "n": len(english_rows),
            "mean_english_cer": round(
                _mean([r["english_cer"] for r in english_rows]), 4
            ),
            "median_english_cer": round(
                median(r["english_cer"] for r in english_rows), 4
            ),
            "mean_english_wer": round(
                _mean([r["english_wer"] for r in english_rows]), 4
            ),
            "median_english_wer": round(
                median(r["english_wer"] for r in english_rows), 4
            ),
            "note": (
                "Supplementary English CER/WER via Liblouis Grade 2 "
                "back-translation. Reference text is derived from the "
                "expected rawBraille cells, not from a separately authored "
                "English transcript. Scores reflect both visual pipeline "
                "accuracy and Liblouis translation quality."
            ),
        }
    sample_entries = []
    for row in rows:
        entry = {
            "label": row["label"],
            "category": row["category"],
            "variant": row["variant"],
            "capture_type": row["capture_type"],
            "cell_error_rate": round(row["cell_error_rate"], 4),
            "rawbraille_cer": round(row["rawbraille_cer"], 4),
            "expected_lines": row["expected_lines"],
            "predicted_lines": row["predicted_lines"],
            "line_count_mismatch": row["line_count_mismatch"],
            "cell_count_mismatch": row["cell_count_mismatch"],
            "exact_sample_match": row["exact_sample_match"],
            "line_reconstruction_accuracy": round(
                row["line_reconstruction_accuracy"], 4
            ),
            "confidence": round(row["confidence"], 4),
            "ms": round(row["ms"], 1),
            "flag_categories": sorted(row["flag_categories"]),
            "failed": row["failed"],
        }
        if "english_cer" in row:
            entry["english_cer"] = round(row["english_cer"], 4)
            entry["english_wer"] = round(row["english_wer"], 4)
        sample_entries.append(entry)
    report["samples"] = sample_entries
    return report


def run(dataset: str, runs: int, write_report: Path | None) -> int:
    spec = get_spec(dataset)
    run_id = _run_id()
    english_scoring = _english_scoring_available()
    settings = get_settings()
    print(f"dataset={spec.name} capture_type={spec.capture_type} "
          f"grade_mode={spec.grade_mode} evaluation_mode={spec.evaluation_mode} "
          f"run_id={run_id}")
    print(_CAPTURE_BANNERS[spec.capture_type])
    if english_scoring:
        print(
            f"english_scoring=True (Liblouis Grade 2 via {settings.liblouis_table} "
            "- supplementary English CER/WER will be computed)"
        )

    samples = discover_dataset(spec)
    if not samples:
        print(
            f"\nNo '{spec.name}' samples found - this is expected on a fresh "
            "checkout (sample folders are local-only and gitignored)."
        )
        if spec.capture_type == "real_capture":
            print(
                "Real-capture intake is empty until safe, anonymised, approved "
                "physical samples are added. Audit readiness first with:\n"
                f"  python -m app.evaluation.audit_rawbraille_dataset --dataset {spec.name}"
            )
        print(f"\nnote: {REPORT_NOTE}")
        return 0

    skipped = [s for s in samples if not s.evaluable]
    evaluable = [s for s in samples if s.evaluable]
    if skipped:
        print("skipped samples:")
        for sample in skipped:
            print(f"  {sample.safe_label}: {'; '.join(sample.skip_reasons)}")
    if not evaluable:
        print(f"\n{len(samples)} sample(s) present but none evaluable - fix the above.")
        return 0

    header = (
        f"{'sample':<40} {'cellER':>7} {'rawCER':>7} {'dLine':>5} "
        f"{'dCell':>5} {'exact':>6} {'conf':>6} {'ms':>7}"
    )
    if english_scoring:
        header += f" {'engCER':>7} {'engWER':>7}"
    print()
    print(header)
    rows = [_evaluate(sample, runs, english_scoring=english_scoring) for sample in evaluable]
    for row in rows:
        _print_row(row, english=english_scoring)

    failed = [r for r in rows if r["failed"]]
    exact = [r for r in rows if r["exact_sample_match"]]
    print()
    print(
        f"=== cell-level summary ({spec.capture_type}; rawBraille vs expected "
        "cells) ==="
    )
    print(
        f"samples={len(samples)} evaluated={len(rows)} skipped={len(skipped)} "
        f"failed={len(failed)}"
    )
    print(
        f"mean_cell_error_rate={_mean([r['cell_error_rate'] for r in rows]):.3f} "
        f"median={median(r['cell_error_rate'] for r in rows):.3f} "
        f"mean_rawbraille_cer={_mean([r['rawbraille_cer'] for r in rows]):.3f}"
    )
    print(
        f"exact_sample_match_rate={len(exact) / max(len(rows), 1):.3f} "
        f"line_count_mismatch_rate="
        f"{sum(1 for r in rows if r['line_count_mismatch']) / max(len(rows), 1):.3f} "
        f"cell_count_mismatch_rate="
        f"{sum(1 for r in rows if r['cell_count_mismatch']) / max(len(rows), 1):.3f}"
    )
    conf = _confidence_summary(rows)
    print(
        f"mean_line_reconstruction_accuracy="
        f"{_mean([r['line_reconstruction_accuracy'] for r in rows]):.3f} "
        f"confidence(mean/median/min/max)="
        f"{conf['mean']:.3f}/{conf['median']:.3f}/{conf['min']:.3f}/{conf['max']:.3f} "
        f"mean_ms={_mean([r['ms'] for r in rows]):.1f} "
        f"repeatable_all={all(r['repeatable'] for r in rows)}"
    )

    english_rows = [r for r in rows if "english_cer" in r]
    if english_rows:
        print()
        print(
            f"=== supplementary English CER/WER (via Liblouis "
            f"{settings.liblouis_table}) ==="
        )
        print(
            f"n={len(english_rows)} "
            f"mean_english_CER={_mean([r['english_cer'] for r in english_rows]):.3f} "
            f"median={median(r['english_cer'] for r in english_rows):.3f} "
            f"mean_english_WER={_mean([r['english_wer'] for r in english_rows]):.3f} "
            f"median={median(r['english_wer'] for r in english_rows):.3f}"
        )
        print(
            "caveat: English reference text is derived from expected rawBraille "
            "via Liblouis, not from a separately authored transcript. Scores "
            "reflect both visual pipeline accuracy and Liblouis translation "
            "quality."
        )

    print()
    print("hardest categories (mean cell error rate, worst first):")
    for cat, cer, n in _hardest_categories(rows):
        print(f"  {cat:<28} cellER={cer:.3f}  n={n}")

    flag_counts: Counter[str] = Counter()
    for row in rows:
        flag_counts.update(row["flag_categories"])
    if flag_counts:
        print()
        print("uncertainty flag categories (samples raising each):")
        for category, count in flag_counts.most_common():
            print(f"  {category}={count}")

    print()
    if english_scoring:
        print(
            f"english_cer_wer_computed=True "
            f"(supplementary, via Liblouis {settings.liblouis_table})"
        )
    else:
        print(
            "english_cer_wer_computed=False (Liblouis Grade 2 not available - "
            "configure LIBLOUIS_TABLE=en-ueb-g2.ctb to enable)"
        )
    print(f"note: {REPORT_NOTE}")

    if write_report is not None:
        report = _build_report(
            spec, rows, len(samples), len(skipped), run_id,
            english_scoring=english_scoring,
        )
        write_report.parent.mkdir(parents=True, exist_ok=True)
        write_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"\nreport written: {write_report} (metrics + safe labels only - "
            "no expected or predicted rawBraille, no draft text)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default=DATASET_NAME,
        help="rawBraille dataset to evaluate (all are cell-level only).",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Runs per image for repeatability (default 3)"
    )
    parser.add_argument(
        "--write-report",
        metavar="PATH",
        help=(
            "Write a sanitized JSON report to PATH (metrics + safe labels only "
            "- never expected/predicted rawBraille, draft text, or image data)."
        ),
    )
    args = parser.parse_args(argv)
    return run(args.dataset, args.runs, Path(args.write_report) if args.write_report else None)


if __name__ == "__main__":
    raise SystemExit(main())
