"""Evaluation harness: OCR output vs ground truth.

Usage:
    python -m app.evaluation.run_evaluation --images ./samples/images --truth ./samples/ground_truth

For each image with a matching <stem>.txt ground-truth file, runs the OCR
pipeline and reports CER, WER, repeatability across runs, processing time,
failure count, and a confidence-vs-error summary.

Prints metrics only — never transcription text or image data.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from time import perf_counter

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


def _build_request(path: Path, mime: str) -> OcrRequest:
    data_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return OcrRequest(
        taskId=f"eval-{path.stem}",
        title=path.stem,
        fileName=path.name,
        mimeType=mime,
        dataUrl=data_url,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory of sample images")
    parser.add_argument("--truth", required=True, help="Directory of .txt ground-truth files")
    parser.add_argument(
        "--runs", type=int, default=3, help="Runs per image for repeatability (default 3)"
    )
    args = parser.parse_args(argv)

    images_dir = Path(args.images)
    truth_dir = Path(args.truth)
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
        for _ in range(max(1, args.runs)):
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
