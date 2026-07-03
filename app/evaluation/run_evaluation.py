"""Evaluation harness: OCR output vs ground truth.

Usage:
    python -m app.evaluation.run_evaluation --images ./samples/images --truth ./samples/ground_truth
    python -m app.evaluation.run_evaluation --dataset embossed
    python -m app.evaluation.run_evaluation --dataset original
    python -m app.evaluation.run_evaluation --dataset real_anonymised

For each image with a matching <stem>.txt ground-truth file, runs the OCR
pipeline and reports CER, WER, repeatability across runs, processing time,
failure count, a flag-category summary, and a confidence-vs-error summary.
`--dataset` is shorthand for the bundled sample directories; explicit
`--images`/`--truth` paths keep working unchanged.

The `real_anonymised` dataset (Stage 3D-E) is local-only and gitignored:
its samples are gated by per-sample metadata (permission, crop quality,
Braille type - see app/evaluation/real_dataset.py), results are grouped by
capture conditions, and a diagnostic summary reports error buckets,
confidence calibration, failure modes, and recommendations. An empty real
dataset exits cleanly - that is the expected state until safe, anonymised,
approved samples are added.

`--write-report PATH` (real_anonymised only) additionally writes a
sanitized JSON baseline report - metrics, buckets, groups, and
recommendations only; never draft text, ground truth, image data, or
unsafe file names.

Prints metrics only - never transcription text, ground truth, image data,
or unsafe file names.
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

from app.evaluation.real_dataset import (
    DATASET_NAME as REAL_DATASET,
    GROUPING_FIELDS,
    IMAGES_DIR as REAL_IMAGES_DIR,
    METADATA_DIR as REAL_METADATA_DIR,
    TRUTH_DIR as REAL_TRUTH_DIR,
    discover_samples,
    dot_size_bucket,
)

DATASETS = {
    "original": ("./samples/images", "./samples/ground_truth"),
    "embossed": ("./samples/embossed_images", "./samples/embossed_ground_truth"),
}

from app.core.config import get_settings
from app.evaluation.metrics import (
    character_error_rate,
    normalise_text,
    word_error_rate,
)
from app.evaluation.repeatability import repeatability_score
from app.models.requests import OcrRequest
from app.ocr.pipeline import run_ocr

MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

REPORT_SCHEMA_VERSION = "1.0"
REPORT_NOTE = (
    "Draft-only OCR. Real-photo validation measures usefulness and "
    "correction burden; it does not certify Braille accuracy. "
    "QTVI/Braille-literate specialist verification remains mandatory."
)


def _build_request(path: Path, mime: str) -> OcrRequest:
    data_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return OcrRequest(
        taskId=f"eval-{path.stem}",
        title=path.stem,
        fileName=path.name,
        mimeType=mime,
        dataUrl=data_url,
    )


def _evaluate_request(request: OcrRequest, reference: str, runs: int) -> dict:
    """Run the pipeline `runs` times for one image; metrics only, no text."""
    outputs: list[str] = []
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
        outputs.append(normalise_text(response.draftText))
        confidence = response.confidence
        if run_index == 0:
            flag_categories = {f.category for f in response.flags}
    hypothesis = outputs[0] if outputs else ""
    return {
        "cer": character_error_rate(reference, hypothesis),
        "wer": word_error_rate(reference, hypothesis),
        "repeatability": repeatability_score(outputs),
        "confidence": confidence,
        "ms": sum(durations_ms) / len(durations_ms) if durations_ms else 0.0,
        "flag_categories": flag_categories,
        "failed": errored or (not hypothesis and bool(reference)),
        "empty_draft": not hypothesis,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bucket(row: dict) -> str:
    if row["failed"] and row["empty_draft"]:
        return "failed"
    if row["cer"] <= 0.10:
        return "low"
    if row["cer"] <= 0.30:
        return "medium"
    return "high"


def _calibration_note(bucket_confidence: dict[str, list[float]]) -> str:
    """One-line honesty check: does confidence fall as error rises?"""
    problem_scores = bucket_confidence["high"] + [
        c for c in bucket_confidence["failed"] if c > 0
    ]
    if not problem_scores:
        return "no high-error samples yet - calibration cannot be judged."
    problem_avg = _mean(problem_scores)
    if problem_avg > 0.55:
        return (
            f"WARNING: high-error/failed samples average confidence "
            f"{problem_avg:.2f} - confidence appears over-optimistic on hard "
            "images; treat scores with caution."
        )
    low_avg = _mean(bucket_confidence["low"])
    if bucket_confidence["low"] and problem_avg >= low_avg:
        return (
            "WARNING: high-error samples score confidence >= low-error "
            "samples; confidence is not separating good from bad drafts."
        )
    return "confidence decreases as error increases - ordering looks honest so far."


def _recommendations(
    rows: list[dict],
    group_stats: dict[str, dict[str, list[dict]]],
    grade2_count: int,
) -> list[str]:
    recs: list[str] = []
    hard = [r for r in rows if _bucket(r) in ("high", "failed")]

    # Capture-condition advice: metadata groups whose error is clearly worse.
    for fieldname, by_value in group_stats.items():
        scored = {
            value: _mean([r["cer"] for r in members])
            for value, members in by_value.items()
            if value != "unknown" and len(members) >= 2
        }
        if len(scored) >= 2:
            worst = max(scored, key=scored.get)
            best = min(scored, key=scored.get)
            if scored[worst] > 0.15 and scored[worst] >= 2 * max(scored[best], 0.02):
                recs.append(
                    f"capture: '{fieldname}={worst}' images error far above "
                    f"'{fieldname}={best}' (CER {scored[worst]:.2f} vs "
                    f"{scored[best]:.2f}) - prefer {best.replace('_', ' ')} capture."
                )

    if hard:
        flag_counts = Counter()
        for row in hard:
            flag_counts.update(row["flag_categories"])
        dominant = ", ".join(c for c, _ in flag_counts.most_common(3))
        recs.append(
            f"pipeline: {len(hard)} sample(s) in high-error/failed buckets; "
            f"dominant flags there: {dominant or 'none raised - flags may be missing a failure mode'}."
        )

    if grade2_count:
        recs.append(
            f"scope: {grade2_count} Grade 2 sample(s) present - keep Grade 2 "
            "excluded from accuracy claims (contractions are not interpreted)."
        )
    if len(rows) < 10:
        recs.append(
            f"data: only {len(rows)} evaluable sample(s) - collect more "
            "(target 10+ across varied lighting/skew/contrast) before drawing "
            "conclusions."
        )
    if not recs:
        recs.append("no obvious issues detected at this sample size.")
    return recs


def _report_skeleton(samples: int, evaluated: int, skipped: int, failed: int) -> dict:
    """Top-level fields shared by minimal and full baseline reports."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "engine_version": get_settings().service_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": REAL_DATASET,
        "counts": {
            "samples": samples,
            "evaluated": evaluated,
            "skipped": skipped,
            "failed": failed,
        },
        "note": REPORT_NOTE,
    }


def _summary_block(rows: list[dict]) -> dict:
    """Aggregate metrics for one group of rows (metrics only, no text)."""
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_cer": round(_mean([r["cer"] for r in rows]), 4),
        "median_cer": round(median(r["cer"] for r in rows), 4),
        "mean_wer": round(_mean([r["wer"] for r in rows]), 4),
        "median_wer": round(median(r["wer"] for r in rows), 4),
        "mean_confidence": round(_mean([r["confidence"] for r in rows]), 4),
        "mean_repeatability": round(_mean([r["repeatability"] for r in rows]), 4),
        "mean_ms": round(_mean([r["ms"] for r in rows]), 1),
    }


def _write_report_file(path: Path, report: dict, evaluated: int) -> None:
    """Write the sanitized JSON report and print one confirmation line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"baseline report written: {path} (evaluated={evaluated}, "
        "metrics only - no draft or ground-truth text)"
    )


def run_real_evaluation(
    images_dir: Path,
    truth_dir: Path,
    metadata_dir: Path,
    runs: int,
    write_report: Path | None = None,
) -> int:
    """Metadata-gated evaluation + diagnostics for the real-photo dataset."""
    samples = discover_samples(images_dir, truth_dir, metadata_dir)

    if not samples:
        print(
            "No real anonymised samples found in "
            f"{images_dir} - this is expected until safe, anonymised, approved "
            "samples are added.\nSee docs/real_photo_validation_protocol.md, "
            "then check the dataset with:\n"
            "  python -m app.evaluation.audit_dataset --dataset real_anonymised"
        )
        print(
            "\nnote: all OCR output is draft-only and requires QTVI or "
            "Braille-literate specialist verification."
        )
        if write_report is not None:
            print("\nno evaluable samples - writing minimal baseline report.")
            _write_report_file(write_report, _report_skeleton(0, 0, 0, 0), 0)
        return 0

    skipped = [s for s in samples if not s.evaluable]
    evaluable = [s for s in samples if s.evaluable]
    if skipped:
        print("skipped samples:")
        for sample in skipped:
            print(f"  {sample.safe_label}: {'; '.join(sample.skip_reasons)}")

    if not evaluable:
        print(
            f"\n{len(samples)} sample(s) present but none are evaluable - fix "
            "the reasons above (run the dataset audit for details)."
        )
        if write_report is not None:
            print("no evaluable samples - writing minimal baseline report.")
            _write_report_file(
                write_report, _report_skeleton(len(samples), 0, len(skipped), 0), 0
            )
        return 0

    rows: list[dict] = []
    grade2_rows: list[dict] = []
    flag_images = Counter()
    group_stats: dict[str, dict[str, list[dict]]] = {
        fieldname: defaultdict(list) for fieldname in GROUPING_FIELDS
    }
    group_stats["dot_size"] = defaultdict(list)

    print()
    print(f"{'sample':<36} {'CER':>6} {'WER':>6} {'conf':>6} {'repeat':>7} {'ms':>7} {'flags':>6}")
    for sample in evaluable:
        reference = normalise_text(sample.truth_path.read_text(encoding="utf-8"))
        mime = MIME_BY_EXTENSION[sample.image_path.suffix.lower()]
        # Safe request context only: the harness never sends titles/file names
        # that could carry identifying text (metadata stays local).
        data_url = (
            f"data:{mime};base64,"
            + base64.b64encode(sample.image_path.read_bytes()).decode("ascii")
        )
        request = OcrRequest(
            taskId=f"eval-{sample.sample_id}",
            title="real-anonymised-evaluation",
            fileName=f"{sample.sample_id}{sample.image_path.suffix.lower()}",
            mimeType=mime,
            dataUrl=data_url,
        )
        row = _evaluate_request(request, reference, runs)
        row["sample"] = sample
        flag_images.update(row["flag_categories"])

        is_grade2 = (sample.metadata or {}).get("braille_type") == "ueb_grade_2"
        (grade2_rows if is_grade2 else rows).append(row)

        for fieldname in GROUPING_FIELDS:
            value = (sample.metadata or {}).get(fieldname) or "unknown"
            group_stats[fieldname][value].append(row)
        group_stats["dot_size"][dot_size_bucket(sample.metadata)].append(row)

        label = sample.safe_label + (" [G2]" if is_grade2 else "")
        print(
            f"{label:<36} {row['cer']:>6.3f} {row['wer']:>6.3f} "
            f"{row['confidence']:>6.3f} {row['repeatability']:>7.3f} "
            f"{row['ms']:>7.1f} {len(row['flag_categories']):>6}"
        )

    all_rows = rows + grade2_rows
    failed = [r for r in all_rows if r["failed"]]

    # --- 1. overall ---------------------------------------------------------
    print()
    print("=== diagnostic summary ===")
    print(
        f"samples={len(samples)} evaluated={len(all_rows)} "
        f"skipped={len(skipped)} failed={len(failed)}"
    )
    if rows:
        print(
            f"grade1/unknown (n={len(rows)}): "
            f"mean_CER={_mean([r['cer'] for r in rows]):.3f} "
            f"median_CER={median(r['cer'] for r in rows):.3f} "
            f"mean_WER={_mean([r['wer'] for r in rows]):.3f} "
            f"median_WER={median(r['wer'] for r in rows):.3f}"
        )
        print(
            f"  mean_confidence={_mean([r['confidence'] for r in rows]):.3f} "
            f"mean_repeatability={_mean([r['repeatability'] for r in rows]):.3f} "
            f"mean_ms={_mean([r['ms'] for r in rows]):.1f}"
        )
    if grade2_rows:
        print(
            f"grade2 - reported separately, NOT a support claim (n={len(grade2_rows)}): "
            f"mean_CER={_mean([r['cer'] for r in grade2_rows]):.3f} "
            f"mean_confidence={_mean([r['confidence'] for r in grade2_rows]):.3f}"
        )

    # --- 2 + 3. error buckets and confidence calibration ---------------------
    bucket_confidence: dict[str, list[float]] = {
        "low": [], "medium": [], "high": [], "failed": []
    }
    for row in all_rows:
        bucket_confidence[_bucket(row)].append(row["confidence"])
    print()
    print("error buckets (all evaluated):")
    for name, label in (
        ("low", "low error (CER<=0.10)"),
        ("medium", "medium error (0.10<CER<=0.30)"),
        ("high", "high error (CER>0.30)"),
        ("failed", "failed / no draft"),
    ):
        scores = bucket_confidence[name]
        avg = f"{_mean(scores):.3f}" if scores else "n/a"
        print(f"  {label:<32} n={len(scores):<3} avg_conf={avg}")
    print(f"calibration: {_calibration_note(bucket_confidence)}")

    # --- 4. failure modes ------------------------------------------------------
    if flag_images:
        print()
        print("flag categories (images raising each):")
        for category, count in flag_images.most_common():
            print(f"  {category}={count}")

    # --- grouped metrics --------------------------------------------------------
    print()
    print("grouped results (mean CER / mean conf / n):")
    for fieldname, by_value in group_stats.items():
        cells = [
            f"{value}: {_mean([r['cer'] for r in members]):.2f}/"
            f"{_mean([r['confidence'] for r in members]):.2f}/{len(members)}"
            for value, members in sorted(by_value.items())
            if members
        ]
        if cells:
            print(f"  {fieldname}: " + "  ".join(cells))

    # --- 5. recommendations ------------------------------------------------------
    recommendations = _recommendations(all_rows, group_stats, len(grade2_rows))
    print()
    print("recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")

    print()
    print(
        "note: all OCR output is draft-only and requires QTVI or "
        "Braille-literate specialist verification. Real-photo validation "
        "measures usefulness and correction burden; it does not certify "
        "Braille accuracy."
    )

    # --- optional sanitized baseline report (metrics only, no text) -----------
    if write_report is not None:
        report = _report_skeleton(
            len(samples), len(all_rows), len(skipped), len(failed)
        )
        report["summary"] = _summary_block(rows)
        if grade2_rows:
            report["grade2_summary"] = _summary_block(grade2_rows)
        report["error_buckets"] = {
            name: {
                "n": len(scores),
                "avg_confidence": round(_mean(scores), 4) if scores else None,
            }
            for name, scores in bucket_confidence.items()
        }
        report["calibration"] = _calibration_note(bucket_confidence)
        report["groups"] = {
            fieldname: {
                value: {
                    "mean_cer": round(_mean([r["cer"] for r in members]), 4),
                    "mean_confidence": round(
                        _mean([r["confidence"] for r in members]), 4
                    ),
                    "n": len(members),
                }
                for value, members in sorted(by_value.items())
                if members
            }
            for fieldname, by_value in group_stats.items()
        }
        report["flag_categories"] = [
            {"category": category, "images": count}
            for category, count in flag_images.most_common()
        ]
        report["recommendations"] = recommendations
        report["samples"] = [
            {
                "label": row["sample"].safe_label,
                "cer": round(row["cer"], 4),
                "wer": round(row["wer"], 4),
                "confidence": round(row["confidence"], 4),
                "repeatability": round(row["repeatability"], 4),
                "ms": round(row["ms"], 1),
                "bucket": _bucket(row),
                "flag_categories": sorted(row["flag_categories"]),
            }
            for row in all_rows
        ]
        print()
        _write_report_file(write_report, report, len(all_rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", help="Directory of sample images")
    parser.add_argument("--truth", help="Directory of .txt ground-truth files")
    parser.add_argument("--metadata", help="Metadata directory (real_anonymised only)")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS) + [REAL_DATASET],
        help="Bundled dataset shorthand (alternative to --images/--truth)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Runs per image for repeatability (default 3)"
    )
    parser.add_argument(
        "--write-report",
        metavar="PATH",
        help=(
            "Write a sanitized JSON baseline report to PATH (metrics only - "
            "never draft text, ground truth, or image data). Applies to the "
            "real_anonymised dataset; ignored for other datasets."
        ),
    )
    args = parser.parse_args(argv)

    if args.dataset == REAL_DATASET:
        return run_real_evaluation(
            Path(args.images) if args.images else REAL_IMAGES_DIR,
            Path(args.truth) if args.truth else REAL_TRUTH_DIR,
            Path(args.metadata) if args.metadata else REAL_METADATA_DIR,
            args.runs,
            write_report=Path(args.write_report) if args.write_report else None,
        )

    if args.dataset:
        default_images, default_truth = DATASETS[args.dataset]
        images = args.images or default_images
        truth = args.truth or default_truth
    elif args.images and args.truth:
        images, truth = args.images, args.truth
    else:
        parser.error("provide --dataset, or both --images and --truth")

    images_dir = Path(images)
    truth_dir = Path(truth)
    if not images_dir.is_dir():
        print(f"images directory not found: {images_dir}")
        return 1
    if not truth_dir.is_dir():
        print(f"ground-truth directory not found: {truth_dir}")
        return 1

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in MIME_BY_EXTENSION
    )
    if not image_paths:
        print(f"no supported images (png/jpg/jpeg) found in {images_dir}")
        return 1

    results: list[dict] = []
    failed_images = 0
    skipped = 0
    flag_categories: Counter[str] = Counter()

    for path in image_paths:
        truth_path = truth_dir / f"{path.stem}.txt"
        if not truth_path.exists():
            print(f"skip (no ground truth): {path.name}")
            skipped += 1
            continue

        reference = normalise_text(truth_path.read_text(encoding="utf-8"))
        request = _build_request(path, MIME_BY_EXTENSION[path.suffix.lower()])

        outputs: list[str] = []
        durations_ms: list[float] = []
        confidence = 0.0
        flag_count = 0
        errored = False
        for run_index in range(max(1, args.runs)):
            started = perf_counter()
            try:
                response = run_ocr(request)
            except Exception:
                errored = True
                break
            durations_ms.append((perf_counter() - started) * 1000)
            outputs.append(normalise_text(response.draftText))
            confidence = response.confidence
            flag_count = len(response.flags)
            if run_index == 0:
                flag_categories.update({f.category for f in response.flags})

        if errored or (not outputs[0] and reference):
            failed_images += 1

        hypothesis = outputs[0] if outputs else ""
        results.append(
            {
                "name": path.name,
                "cer": character_error_rate(reference, hypothesis),
                "wer": word_error_rate(reference, hypothesis),
                "repeatability": repeatability_score(outputs),
                "confidence": confidence,
                "ms": sum(durations_ms) / len(durations_ms) if durations_ms else 0.0,
                "flags": flag_count,
            }
        )

    if not results:
        print("no images had matching ground truth; nothing to evaluate")
        return 1

    print()
    print(f"{'file':<36} {'CER':>6} {'WER':>6} {'conf':>6} {'repeat':>7} {'ms':>7} {'flags':>6}")
    for row in results:
        print(
            f"{row['name']:<36} {row['cer']:>6.3f} {row['wer']:>6.3f} "
            f"{row['confidence']:>6.3f} {row['repeatability']:>7.3f} "
            f"{row['ms']:>7.1f} {row['flags']:>6}"
        )

    count = len(results)
    mean = lambda key: sum(r[key] for r in results) / count  # noqa: E731
    print()
    print(
        f"images={count} skipped={skipped} failed={failed_images} "
        f"mean_CER={mean('cer'):.3f} mean_WER={mean('wer'):.3f} "
        f"mean_confidence={mean('confidence'):.3f} "
        f"mean_repeatability={mean('repeatability'):.3f} mean_ms={mean('ms'):.1f}"
    )

    if flag_categories:
        summary = " ".join(
            f"{category}={count}" for category, count in flag_categories.most_common()
        )
        print(f"flag categories (images raising each): {summary}")

    low_error = [r for r in results if r["cer"] <= 0.10]
    high_error = [r for r in results if r["cer"] > 0.10]

    def _avg_confidence(rows: list[dict]) -> str:
        if not rows:
            return "n/a"
        return f"{sum(r['confidence'] for r in rows) / len(rows):.3f}"

    print(
        "confidence vs error: "
        f"low-error (CER<=0.10) n={len(low_error)} avg_conf={_avg_confidence(low_error)} | "
        f"high-error (CER>0.10) n={len(high_error)} avg_conf={_avg_confidence(high_error)}"
    )
    print()
    print(
        "note: all OCR output is draft-only and requires QTVI or "
        "Braille-literate specialist verification."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
