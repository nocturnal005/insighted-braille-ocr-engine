"""Real-capture Braille OCR diagnostic CLI (Stage 3D-J1). LOCAL-ONLY output.

Usage (from the repo root):

    python -m app.evaluation.run_real_capture_diagnostic
    python -m app.evaluation.run_real_capture_diagnostic --input <folder> \\
        --report reports/real_capture_diagnostic/run.json

Scans a folder of PNG/JPEG Braille capture candidates, runs the existing OCR
pipeline on each (no pipeline changes, no /ocr contract changes), and writes
a local diagnostic report answering, per image:

    * how far the pipeline got (stage ladder L0-L6, see diagnostic_probe);
    * where it stopped (failure point);
    * capture-quality triage (readable / borderline / retake / unusable);
    * whether a Grade 2 draft was produced when Liblouis is configured;
    * cell-level scores, ONLY for gated samples with .braille ground truth.

This is a diagnostic tool, not an accuracy claim. Scores exist only for
samples that pass the Stage 3D-G6 gating (explicit approved_for_testing
permission, safe naming, matching .braille ground truth). Everything else
is preview-only. With an empty intake the run reports BLOCKED - that is
the honest, expected state until real samples are collected.

Safety:
  * The report contains counts, scores, stage labels, flag categories, and
    fixed reason strings only - never rawBraille, draft text, ground truth,
    flag reason prose, or raw file names of unsafely named files.
  * Metadata marked contains_real_pupil_data / contains_live_assessment_material
    blocks the sample entirely - it is not probed, not OCR'd (exit code 2).
  * Report paths inside the repository must be gitignored; the CLI refuses
    to write a committable report.
  * OCR output remains draft-only: QTVI/Braille-literate specialist
    verification is mandatory before any real use in InsightEd AI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import DRAFT_ONLY_WARNING, get_settings
from app.evaluation.capture_quality import assess_capture_quality
from app.evaluation.diagnostic_probe import (
    MIME_BY_EXTENSION,
    probe_image_file,
    score_against_expected,
)
from app.evaluation.rawbraille_dataset import (
    EXPECTED_SUFFIX,
    REAL_EXPECTED_DIR,
    REAL_IMAGES_DIR,
    REAL_METADATA_DIR,
)
from app.evaluation.real_dataset import unsafe_name_reasons

REPORT_SCHEMA_VERSION = "1.0"
STAGE_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]

DIAGNOSTIC_NOTE = (
    "Diagnostic report only. Stage labels and quality classifications are "
    "heuristics describing how far the pipeline progressed on each image - "
    "they are NOT accuracy measurements. Cell-level scores appear only for "
    "samples gated by the Stage 3D-G6 real-capture protocol with .braille "
    "ground truth. " + DRAFT_ONLY_WARNING
)

PERMISSION_APPROVED = "approved_for_testing"
BLOCKING_METADATA_FLAGS = (
    "contains_real_pupil_data",
    "contains_live_assessment_material",
)


def _safe_label(stem: str, withheld_index: int) -> str:
    """Safe display label for a sample.

    Returns the stem when it is safely named. For an unsafely named file it
    returns a positional ``withheld_NNN`` label that reveals nothing about
    the original name. A positional index (not a hash of the name) is used
    deliberately: a truncated hash of a pupil/school name is guess-confirmable
    from a candidate list, which would defeat the purpose of withholding it.
    """
    if unsafe_name_reasons(stem):
        return f"withheld_{withheld_index:03d}"
    return stem


# Repo root derived from this module's own location (app/evaluation/<this>.py),
# so it is correct regardless of the process working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Directory names that .gitignore keeps out of the repo (used as a fail-closed
# fallback when git itself cannot be consulted). ``reports/`` is gitignored.
_GITIGNORED_DIR_NAMES = {"reports"}


def _is_gitignored(path: Path) -> bool | None:
    """True/False when git answers, else None. Anchored to the repo root so
    the answer does not depend on the process working directory."""
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            capture_output=True,
            cwd=str(_REPO_ROOT),
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def _report_path_is_safe(report_path: Path) -> tuple[bool, str]:
    """A report inside the repo must be gitignored; outside the repo is fine.

    Repo membership is decided from the module-derived repo root (not the
    working directory), and the gitignore check is anchored there too, so a
    report path cannot slip past by running from another directory. When git
    cannot answer, fail closed: allow only paths under a known-gitignored
    directory, otherwise refuse.
    """
    resolved = report_path.resolve()
    if _REPO_ROOT not in resolved.parents:
        return True, "outside repository"
    ignored = _is_gitignored(resolved)
    if ignored is True:
        return True, "gitignored"
    if ignored is False:
        return False, (
            "report path is inside the repository but NOT gitignored - "
            "refusing to write a committable diagnostic report"
        )
    # git unavailable: only trust a path under a known-gitignored directory.
    rel_parts = set(resolved.relative_to(_REPO_ROOT).parts)
    if rel_parts & _GITIGNORED_DIR_NAMES:
        return True, "under a known-gitignored directory (git unavailable)"
    return False, (
        "report path is inside the repository and git is unavailable to "
        "confirm it is gitignored - refusing to write a possibly committable "
        "report (put it under reports/ or outside the repo)"
    )


def _load_metadata(metadata_dir: Path, stem: str) -> tuple[dict | None, str | None]:
    """Load per-sample metadata.

    Returns ``(metadata, error)``:
      * ``(None, None)``  - no metadata file present;
      * ``(dict, None)``  - parsed successfully;
      * ``(None, reason)``- a file exists but could not be read or parsed.

    The error case must FAIL CLOSED: a metadata file we cannot read might be
    the one marking forbidden material, so the caller blocks the sample
    rather than assume it is safe.
    """
    candidate = metadata_dir / f"{stem}.json"
    if not candidate.is_file():
        return None, None
    try:
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "metadata file present but could not be read/parsed"
    if not isinstance(loaded, dict):
        return None, "metadata file is not a JSON object"
    return loaded, None


def _evaluate_gating(
    metadata: dict | None,
    metadata_error: str | None,
    expected_path: Path,
    stem: str,
) -> tuple[bool, list[str], list[str]]:
    """(scorable, block_reasons, skip_reasons) for one candidate.

    Fails closed: unparseable metadata blocks the sample (we cannot verify
    it is not forbidden material), and non-True-but-truthy safety flags are
    still treated as blocking.
    """
    blocks: list[str] = []
    skips: list[str] = []

    if metadata_error is not None:
        blocks.append(
            f"{metadata_error} - cannot verify the sample is safe; refusing "
            "to process"
        )
        return False, blocks, skips

    if metadata is not None:
        for flag_name in BLOCKING_METADATA_FLAGS:
            # Any truthy value blocks (not just the literal True) so a
            # "true"/1/"yes" typo cannot silently fail open.
            if metadata.get(flag_name):
                blocks.append(f"metadata marks {flag_name} - forbidden material")
        if metadata.get("requires_english_transcript"):
            skips.append("requires_english_transcript is out of scope")

    if blocks:
        return False, blocks, skips

    if metadata is None:
        skips.append("no metadata - permission must be explicit for scoring")
    elif metadata.get("permission_status") != PERMISSION_APPROVED:
        skips.append(
            "permission_status is not 'approved_for_testing' - not scorable"
        )
    if not expected_path.is_file():
        skips.append("no .braille ground truth - preview diagnostics only")
    if unsafe_name_reasons(stem):
        skips.append("unsafe file name - id withheld; rename per G6 protocol")

    return not skips, blocks, skips


def _markdown_report(report: dict) -> str:
    lines = [
        "# Real-capture Braille OCR diagnostic report",
        "",
        f"generated_at: {report['generated_at']}",
        f"engine_version: {report['engine_version']}",
        f"verdict: **{report['verdict']}**",
        "",
        f"> {report['note']}",
        "",
        "## Counts",
        "",
    ]
    for key, value in report["counts"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Candidates", ""]
    header = (
        "| label | stage | failure point | quality | conf | dots | cells "
        "| lines | grade2 draft | scored |"
    )
    lines.append(header)
    lines.append("|" + " --- |" * 10)
    for entry in report["candidates"]:
        probe = entry["probe"]
        quality = entry["capture_quality"]
        scored = "yes" if entry.get("scores") else "no"
        lines.append(
            f"| {entry['label']} | {probe['stage']} | {probe['failure_point']} "
            f"| {quality['classification']} | {probe['confidence']:.3f} "
            f"| {probe['accepted_dots']} | {probe['total_cells']} "
            f"| {probe['lines_detected']} "
            f"| {'yes' if probe['grade2_draft_produced'] else 'no'} | {scored} |"
        )
    if report.get("blocked"):
        lines += ["", "## Blocked", ""]
        for entry in report["blocked"]:
            lines.append(f"- {entry['label']}: {'; '.join(entry['reasons'])}")
    if report.get("scored_summary"):
        lines += ["", "## Scored summary (gated samples only)", ""]
        for key, value in report["scored_summary"].items():
            lines.append(f"- {key}: {value}")
    lines += ["", f"note: {report['note']}", ""]
    return "\n".join(lines)


def run(
    input_dir: Path,
    expected_dir: Path,
    metadata_dir: Path,
    report_path: Path | None,
) -> int:
    settings = get_settings()
    generated_at = datetime.now(timezone.utc).isoformat()
    is_default_intake = input_dir.resolve() == (_REPO_ROOT / REAL_IMAGES_DIR).resolve()

    print("=== Real-capture Braille OCR diagnostic (Stage 3D-J1) ===")
    print(DIAGNOSTIC_NOTE)
    print()
    print(
        f"input={input_dir} intake_kind="
        f"{'real_capture_intake' if is_default_intake else 'custom_folder'} "
        f"liblouis_enabled={settings.liblouis_enabled} "
        f"liblouis_table={settings.liblouis_table}"
    )

    if report_path is not None:
        safe, why = _report_path_is_safe(report_path)
        if not safe:
            print(f"BLOCK: {why}")
            return 2

    if not input_dir.is_dir():
        print(f"input directory not found: {input_dir}")
        print("verdict: BLOCKED (input directory missing)")
        return 2

    candidates = sorted(
        p
        for p in input_dir.iterdir()
        if p.suffix.lower() in MIME_BY_EXTENSION and p.is_file()
    )

    report: dict = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "engine_version": settings.service_version,
        "intake_kind": (
            "real_capture_intake" if is_default_intake else "custom_folder"
        ),
        "capture_type": "real_capture" if is_default_intake else "unspecified",
        "liblouis_enabled": settings.liblouis_enabled,
        "liblouis_table": settings.liblouis_table,
        "note": DIAGNOSTIC_NOTE,
        "candidates": [],
        "blocked": [],
    }

    if not candidates:
        report["verdict"] = (
            "BLOCKED (no real-capture candidates present - expected until "
            "samples are collected per the Stage 3D-G6 protocol)"
        )
        report["counts"] = {
            "candidates": 0,
            "probed": 0,
            "blocked": 0,
            "scored": 0,
        }
        print()
        print("no PNG/JPEG candidates found.")
        print(f"verdict: {report['verdict']}")
        _write_report(report, report_path)
        return 0

    probed = 0
    blocked_entries: list[dict] = []
    stage_histogram: Counter[str] = Counter()
    quality_histogram: Counter[str] = Counter()
    scored_rows: list[dict] = []

    print()
    print(
        f"{'label':<32} {'stage':>5} {'quality':<22} {'conf':>6} "
        f"{'dots':>5} {'cells':>6} {'g2':>3}"
    )
    withheld_count = 0
    for path in candidates:
        stem = path.stem
        if unsafe_name_reasons(stem):
            withheld_count += 1
        label = _safe_label(stem, withheld_count)
        metadata, metadata_error = _load_metadata(metadata_dir, stem)
        expected_path = expected_dir / f"{stem}{EXPECTED_SUFFIX}"
        scorable, blocks, skips = _evaluate_gating(
            metadata, metadata_error, expected_path, stem
        )

        if blocks:
            blocked_entries.append({"label": label, "reasons": blocks})
            print(f"BLOCK: {label}: {'; '.join(blocks)} - sample not processed")
            continue

        probe = probe_image_file(path)
        quality = assess_capture_quality(path, probe)
        probed += 1

        entry: dict = {
            "label": label,
            "probe": probe.to_safe_dict(),
            "capture_quality": quality.to_safe_dict(),
            "gating_notes": skips,
        }

        # Score only gated samples that actually produced rawBraille: a
        # sample that failed before L4 has nothing to score, and promoting
        # it to L6 would misrepresent how far it got.
        if scorable and not probe.rawbraille_nonempty:
            entry["gating_notes"] = skips + [
                f"gated but produced no rawBraille (stopped at {probe.stage}/"
                f"{probe.failure_point}) - not scored"
            ]
        elif scorable:
            try:
                expected_text = expected_path.read_text(encoding="utf-8")
                entry["scores"] = score_against_expected(probe, expected_text)
                entry["probe"] = probe.to_safe_dict()  # stage promoted to L6
                scored_rows.append(entry["scores"])
            except (OSError, ValueError):
                # OSError: file unreadable. ValueError: bad UTF-8 ground
                # truth, or score_against_expected refusing an empty probe.
                entry["gating_notes"] = skips + [
                    "ground truth unreadable or unscorable - preview only"
                ]

        stage_histogram[probe.stage] += 1
        quality_histogram[quality.classification] += 1
        report["candidates"].append(entry)
        print(
            f"{label:<32} {probe.stage:>5} {quality.classification:<22} "
            f"{probe.confidence:>6.3f} {probe.accepted_dots:>5} "
            f"{probe.total_cells:>6} "
            f"{'yes' if probe.grade2_draft_produced else 'no':>3}"
        )

    report["blocked"] = blocked_entries
    highest_stage = max(
        (STAGE_ORDER.index(s) for s in stage_histogram), default=0
    )
    report["counts"] = {
        "candidates": len(candidates),
        "probed": probed,
        "blocked": len(blocked_entries),
        "scored": len(scored_rows),
    }
    report["stage_histogram"] = dict(sorted(stage_histogram.items()))
    report["quality_histogram"] = dict(sorted(quality_histogram.items()))
    report["highest_stage_reached"] = STAGE_ORDER[highest_stage]

    if scored_rows:
        n = len(scored_rows)
        report["scored_summary"] = {
            "n": n,
            "mean_cell_error_rate": round(
                sum(r["cell_error_rate"] for r in scored_rows) / n, 4
            ),
            "mean_rawbraille_cer": round(
                sum(r["rawbraille_cer"] for r in scored_rows) / n, 4
            ),
            "exact_sample_match_rate": round(
                sum(1 for r in scored_rows if r["exact_sample_match"]) / n, 4
            ),
        }
        english = [r for r in scored_rows if r["english_cer"] is not None]
        if english:
            report["scored_summary"]["english_n"] = len(english)
            report["scored_summary"]["mean_english_cer"] = round(
                sum(r["english_cer"] for r in english) / len(english), 4
            )
            report["scored_summary"]["mean_english_wer"] = round(
                sum(r["english_wer"] for r in english) / len(english), 4
            )

    if blocked_entries:
        verdict = "BLOCKED SAMPLES PRESENT (forbidden material in intake)"
    elif scored_rows:
        verdict = f"diagnosed {probed} candidate(s); {len(scored_rows)} scored"
    else:
        verdict = (
            f"diagnosed {probed} candidate(s); 0 scored - formal real-capture "
            "evaluation remains BLOCKED until gated samples with .braille "
            "ground truth exist"
        )
    report["verdict"] = verdict

    print()
    print(f"stage histogram: {report['stage_histogram']}")
    print(f"quality histogram: {report['quality_histogram']}")
    print(f"highest stage reached: {report['highest_stage_reached']}")
    print(f"verdict: {verdict}")
    print()
    print(f"note: {DIAGNOSTIC_NOTE}")

    _write_report(report, report_path)
    return 2 if blocked_entries else 0


def _write_report(report: dict, report_path: Path | None) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.suffix.lower() == ".md":
        report_path.write_text(_markdown_report(report), encoding="utf-8")
    else:
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"report written: {report_path} (local-only; counts, scores, and "
        "safe labels - no OCR text, no Braille content, no ground truth)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults are anchored to the repo root (not the working directory) so
    # the real-capture intake - and its safety-gating metadata - is found
    # correctly no matter where the CLI is invoked from.
    parser.add_argument(
        "--input",
        default=str(_REPO_ROOT / REAL_IMAGES_DIR),
        help="Folder of PNG/JPEG capture candidates "
        "(default: the Stage 3D-G6 real-capture intake)",
    )
    parser.add_argument(
        "--expected",
        default=str(_REPO_ROOT / REAL_EXPECTED_DIR),
        help="Folder of .braille ground-truth files (default: G6 intake)",
    )
    parser.add_argument(
        "--metadata",
        default=str(_REPO_ROOT / REAL_METADATA_DIR),
        help="Folder of per-sample metadata JSON (default: G6 intake)",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="Write a local JSON (or .md) report. Must be gitignored when "
        "inside the repository.",
    )
    args = parser.parse_args(argv)
    return run(
        Path(args.input),
        Path(args.expected),
        Path(args.metadata),
        Path(args.report) if args.report else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
