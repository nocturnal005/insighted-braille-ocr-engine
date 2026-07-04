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

English draft-text CER/WER is deliberately NOT computed - the engine does not
interpret Grade 2 contractions, so English scoring would be meaningless here.
Reports carry explicit ``english_cer_wer_computed: false`` and scope wording.

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


def _evaluate(sample, runs: int) -> dict:
    """Run the pipeline `runs` times on one sample; cell-level metrics only."""
    expected = sample.expected_rawbraille()
    mime = MIME_BY_EXTENSION[sample.image_path.suffix.lower()]
    data_url = (
        f"data:{mime};base64,"
        + base64.b64encode(sample.image_path.read_bytes()).decode("ascii")
    )
    # Safe request context only - no titles/file names that could carry
    # identifying text (sample material stays local).
    request = OcrRequest(
        taskId=f"rawbraille-{sample.sample_id}",
        title="rawbraille-cell-level-validation",
        fileName=f"{sample.sample_id}{sample.image_path.suffix.lower()}",
        mimeType=mime,
        dataUrl=data_url,
    )

    raw_outputs: list[str] = []
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


def _print_row(row: dict) -> None:
    print(
        f"{row['label']:<40} {row['cell_error_rate']:>7.3f} "
        f"{row['rawbraille_cer']:>7.3f} {row['line_count_mismatch']:>5} "
        f"{row['cell_count_mismatch']:>5} {str(row['exact_sample_match']):>6} "
        f"{row['confidence']:>6.3f} {row['ms']:>7.1f}"
    )


def _build_report(
    spec: RawBrailleDatasetSpec,
    rows: list[dict],
    samples: int,
    skipped: int,
    run_id: str,
) -> dict:
    failed = [r for r in rows if r["failed"]]
    exact = [r for r in rows if r["exact_sample_match"]]
    flag_counts: Counter[str] = Counter()
    for row in rows:
        flag_counts.update(row["flag_categories"])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "engine_version": get_settings().service_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset": _dataset_descriptor(spec),
        # Explicit scope statement: this report can never carry English scores.
        "english_cer_wer_computed": False,
        "grade2_english_transcription": "out_of_scope",
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
        # Per-sample: metrics + safe label only. No Braille text ever.
        "samples": [
            {
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
            for row in rows
        ],
    }


def run(dataset: str, runs: int, write_report: Path | None) -> int:
    spec = get_spec(dataset)
    run_id = _run_id()
    print(f"dataset={spec.name} capture_type={spec.capture_type} "
          f"grade_mode={spec.grade_mode} evaluation_mode={spec.evaluation_mode} "
          f"run_id={run_id}")
    print(_CAPTURE_BANNERS[spec.capture_type])

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

    print()
    print(
        f"{'sample':<40} {'cellER':>7} {'rawCER':>7} {'dLine':>5} "
        f"{'dCell':>5} {'exact':>6} {'conf':>6} {'ms':>7}"
    )
    rows = [_evaluate(sample, runs) for sample in evaluable]
    for row in rows:
        _print_row(row)

    failed = [r for r in rows if r["failed"]]
    exact = [r for r in rows if r["exact_sample_match"]]
    print()
    print(
        f"=== cell-level summary ({spec.capture_type}; rawBraille vs expected "
        "cells; NOT English) ==="
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
    print("english_cer_wer_computed=False (Grade 2 English transcription is out of scope)")
    print(f"note: {REPORT_NOTE}")

    if write_report is not None:
        report = _build_report(spec, rows, len(samples), len(skipped), run_id)
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
