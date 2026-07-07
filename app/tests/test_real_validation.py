"""Stage 3D-E tests: real anonymised photo validation framework.

Covers unsafe-name screening, metadata validation, permission gating,
the dataset audit command, the real_anonymised evaluation path (including
empty-dataset behaviour, output safety, grouping, buckets, calibration),
and that the original dataset evaluation still works. All images are
synthetic embossed renders written into tmp directories — no real photos.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.audit_dataset import main as audit_main
from app.evaluation.real_dataset import (
    discover_samples,
    dot_size_bucket,
    unsafe_name_reasons,
    validate_metadata,
)
from app.evaluation.run_evaluation import _bucket, _calibration_note
from app.evaluation.run_evaluation import main as eval_main
from app.evaluation.sample_generator import EmbossedStyle, render_embossed_braille_image

GOOD_METADATA = {
    "sample_id": "real_001",
    "source_type": "synthetic_printout",
    "braille_type": "ueb_grade_1",
    "capture_method": "scanner",
    "lighting": "good_even",
    "contrast": "high",
    "skew": "none",
    "crop_quality": "braille_only",
    "dot_size_px_estimate": 9,
    "permission_status": "synthetic",
    "notes": "generated for framework tests",
}


@pytest.fixture()
def dataset_dirs(tmp_path):
    images = tmp_path / "images"
    truth = tmp_path / "truth"
    metadata = tmp_path / "metadata"
    for directory in (images, truth, metadata):
        directory.mkdir()
    return images, truth, metadata


def add_sample(
    dirs,
    sample_id: str,
    text: str = "the cat sat on the mat",
    style: EmbossedStyle | None = None,
    metadata: dict | None = GOOD_METADATA,
    with_truth: bool = True,
    seed: int = 5,
) -> None:
    images, truth, metadata_dir = dirs
    image = render_embossed_braille_image(text, style or EmbossedStyle(), seed=seed)
    image.save(images / f"{sample_id}.png")
    if with_truth:
        (truth / f"{sample_id}.txt").write_text(text, encoding="utf-8")
    if metadata is not None:
        record = dict(metadata, sample_id=sample_id)
        (metadata_dir / f"{sample_id}.json").write_text(json.dumps(record), encoding="utf-8")


def dir_args(dirs) -> list[str]:
    images, truth, metadata = dirs
    return ["--images", str(images), "--truth", str(truth), "--metadata", str(metadata)]


# --- Unsafe name screening ----------------------------------------------------


@pytest.mark.parametrize(
    "stem",
    [
        "pupil_name_homework",
        "schoolname_year11_exam",
        "real_student_assessment",
        "page for jamie",
        "backup_2024-03-12_scan",
        "contact_someone@example.com",
        "Jane_Doe",
    ],
)
def test_unsafe_names_flagged(stem):
    assert unsafe_name_reasons(stem)


@pytest.mark.parametrize(
    "stem",
    [
        "real_001_clean_flat_good_light",
        "real_002_low_contrast_angle_light",
        "real_003_mild_skew_shadow",
    ],
)
def test_safe_names_pass(stem):
    assert unsafe_name_reasons(stem) == []


# --- Metadata -----------------------------------------------------------------


def test_valid_metadata_has_no_issues():
    assert validate_metadata(GOOD_METADATA) == []


def test_metadata_issues_reported():
    broken = dict(GOOD_METADATA)
    del broken["lighting"]
    broken["permission_status"] = "yes please"
    broken["dot_size_px_estimate"] = "nine"
    issues = validate_metadata(broken)
    assert any("lighting" in issue for issue in issues)
    assert any("permission_status" in issue for issue in issues)
    assert any("dot_size_px_estimate" in issue for issue in issues)


def test_dot_size_buckets():
    assert dot_size_bucket(None) == "unknown"
    assert dot_size_bucket({"dot_size_px_estimate": 4}) == "under_6px"
    assert dot_size_bucket({"dot_size_px_estimate": 9}) == "6_to_10px"
    assert dot_size_bucket({"dot_size_px_estimate": 14}) == "over_10px"


# --- Discovery gating -----------------------------------------------------------


def test_discovery_gates_samples(dataset_dirs):
    add_sample(dataset_dirs, "real_001_ok")
    add_sample(
        dataset_dirs,
        "real_002_blocked",
        metadata=dict(GOOD_METADATA, permission_status="not_approved"),
    )
    add_sample(dataset_dirs, "real_003_no_truth", with_truth=False)
    add_sample(
        dataset_dirs,
        "real_004_bad_crop",
        metadata=dict(GOOD_METADATA, crop_quality="includes_non_braille"),
    )
    add_sample(dataset_dirs, "real_005_no_metadata", metadata=None)

    by_id = {s.sample_id: s for s in discover_samples(*dataset_dirs)}
    assert by_id["real_001_ok"].evaluable
    assert not by_id["real_002_blocked"].evaluable
    assert any("not_approved" in r for r in by_id["real_002_blocked"].skip_reasons)
    assert any("ground-truth" in r for r in by_id["real_003_no_truth"].skip_reasons)
    assert any("non_braille" in r for r in by_id["real_004_bad_crop"].skip_reasons)
    assert any("permission unknown" in r for r in by_id["real_005_no_metadata"].skip_reasons)


# --- Dataset audit ---------------------------------------------------------------


def test_audit_empty_dataset_exits_cleanly(dataset_dirs, capsys):
    assert audit_main(dir_args(dataset_dirs)) == 0
    out = capsys.readouterr().out
    assert "No real validation images present" in out


def test_audit_reports_warnings(dataset_dirs, capsys):
    add_sample(dataset_dirs, "real_001_ok")
    add_sample(dataset_dirs, "pupil_name_homework")  # unsafe name
    add_sample(dataset_dirs, "real_003_no_truth", with_truth=False)
    add_sample(
        dataset_dirs,
        "real_004_blocked",
        metadata=dict(GOOD_METADATA, permission_status="not_approved"),
    )
    (dataset_dirs[0] / "notes.pdf").write_bytes(b"%PDF-fake")  # unsupported type

    assert audit_main(dir_args(dataset_dirs)) == 0
    out = capsys.readouterr().out
    assert "unsafe file name" in out
    assert "missing ground-truth" in out
    assert "not_approved" in out
    assert "unsupported file type" in out
    # 5 discovered (4 png + the pdf); evaluable = ok sample + unsafe-named
    # sample (unsafe names warn and are masked in reports, but do not block)
    assert "summary: samples=5 evaluable=2" in out
    # audit never deletes: everything still on disk
    assert (dataset_dirs[0] / "pupil_name_homework.png").exists()
    assert (dataset_dirs[0] / "notes.pdf").exists()


# --- Real evaluation --------------------------------------------------------------


def test_real_evaluation_empty_dataset_exits_cleanly(dataset_dirs, capsys):
    code = eval_main(["--dataset", "real_anonymised", *dir_args(dataset_dirs)])
    assert code == 0
    out = capsys.readouterr().out
    assert "No real anonymised samples found" in out
    assert "draft-only" in out


def test_real_evaluation_full_run_is_safe_and_grouped(dataset_dirs, capsys):
    secret_text = "the cat sat on the mat"
    add_sample(dataset_dirs, "real_001_good_light", text=secret_text)
    add_sample(
        dataset_dirs,
        "real_002_low_light",
        text="reading by touch",
        style=EmbossedStyle(relief=10.0),
        metadata=dict(GOOD_METADATA, lighting="low_light", contrast="low"),
    )
    # Near the resolution floor. Pre-K2 this failed outright; the K2 lattice
    # fallback now recovers it to a low-confidence, high-severity-flagged
    # draft (honest, not overconfident — see test_embossed), so it no longer
    # lands in the failed/no-draft bucket.
    add_sample(
        dataset_dirs,
        "real_003_tiny_dots",
        text="tight dot spacing",
        style=EmbossedStyle(unit=9, dot_radius=3),
        metadata=dict(GOOD_METADATA, dot_size_px_estimate=5),
        seed=7,
    )
    add_sample(
        dataset_dirs,
        "real_004_blocked",
        metadata=dict(GOOD_METADATA, permission_status="not_approved"),
    )
    add_sample(
        dataset_dirs,
        "real_005_grade2",
        text="shadows on paper",
        metadata=dict(GOOD_METADATA, braille_type="ueb_grade_2"),
    )

    code = eval_main(["--dataset", "real_anonymised", "--runs", "1", *dir_args(dataset_dirs)])
    assert code == 0
    out = capsys.readouterr().out

    # skipping + separate grade-2 reporting
    assert "real_004_blocked: permission_status is not_approved" in out
    assert "real_005_grade2 [G2]" in out
    assert "grade2 - reported separately" in out

    # diagnostic sections
    assert "diagnostic summary" in out
    assert "samples=5 evaluated=4 skipped=1 failed=0" in out
    assert "error buckets" in out
    assert "failed / no draft" in out
    assert "calibration:" in out
    assert "grouped results" in out
    assert "lighting:" in out and "dot_size:" in out
    assert "recommendations:" in out
    assert "draft-only" in out

    # output safety: never prints ground truth / draft text or base64
    assert secret_text not in out
    assert "reading by touch" not in out
    assert "tight dot spacing" not in out
    assert "base64" not in out and "data:image" not in out


def test_real_evaluation_masks_unsafe_sample_names(dataset_dirs, capsys):
    add_sample(dataset_dirs, "pupil_page_photo")
    code = eval_main(["--dataset", "real_anonymised", "--runs", "1", *dir_args(dataset_dirs)])
    assert code == 0
    out = capsys.readouterr().out
    assert "pupil_page_photo" not in out
    assert "(id withheld" in out


# --- Baseline report writing (--write-report) ---------------------------------------


def test_write_report_creates_sanitized_json(dataset_dirs, tmp_path, capsys):
    add_sample(dataset_dirs, "real_001_good_light")
    add_sample(
        dataset_dirs,
        "real_002_low_light",
        text="reading by touch",
        style=EmbossedStyle(relief=10.0),
        metadata=dict(GOOD_METADATA, lighting="low_light", contrast="low"),
    )
    report_path = tmp_path / "reports" / "baseline.json"
    code = eval_main(
        [
            "--dataset", "real_anonymised", "--runs", "1",
            "--write-report", str(report_path), *dir_args(dataset_dirs),
        ]
    )
    assert code == 0
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # top-level summary fields
    assert report["schema_version"] == "1.0"
    assert report["engine_version"]
    assert report["generated_at"]
    assert report["dataset"] == "real_anonymised"
    assert report["counts"] == {
        "samples": 2, "evaluated": 2, "skipped": 0, "failed": 0,
    }
    for key in (
        "mean_cer", "median_cer", "mean_wer", "median_wer",
        "mean_confidence", "mean_repeatability", "mean_ms",
    ):
        assert key in report["summary"]
    assert set(report["error_buckets"]) == {"low", "medium", "high", "failed"}
    assert report["calibration"]
    assert "lighting" in report["groups"] and "dot_size" in report["groups"]
    assert report["groups"]["lighting"]["low_light"]["n"] == 1
    assert isinstance(report["flag_categories"], list)
    assert report["recommendations"]
    assert "mandatory" in report["note"] and "Draft-only" in report["note"]

    # per-sample entries: safe labels + metrics only, nothing else
    labels = [entry["label"] for entry in report["samples"]]
    assert sorted(labels) == ["real_001_good_light", "real_002_low_light"]
    for entry in report["samples"]:
        assert set(entry) == {
            "label", "cer", "wer", "confidence", "repeatability",
            "ms", "bucket", "flag_categories",
        }
        assert entry["bucket"] in {"low", "medium", "high", "failed"}
        assert entry["flag_categories"] == sorted(entry["flag_categories"])

    # single confirmation line with the report path; existing output intact
    out = capsys.readouterr().out
    assert "baseline report written" in out
    assert str(report_path) in out
    assert "diagnostic summary" in out


def test_write_report_contains_no_text_or_image_data(dataset_dirs, tmp_path):
    secret_text = "the cat sat on the mat"
    add_sample(dataset_dirs, "real_001_good_light", text=secret_text)
    add_sample(dataset_dirs, "real_002_touch", text="reading by touch")
    report_path = tmp_path / "report.json"
    code = eval_main(
        [
            "--dataset", "real_anonymised", "--runs", "1",
            "--write-report", str(report_path), *dir_args(dataset_dirs),
        ]
    )
    assert code == 0
    raw = report_path.read_text(encoding="utf-8")
    # never ground truth / draft text, base64, or image data
    assert secret_text not in raw
    assert "reading by touch" not in raw
    assert "base64" not in raw and "data:image" not in raw


def test_write_report_masks_unsafe_sample_names(dataset_dirs, tmp_path):
    add_sample(dataset_dirs, "pupil_page_photo")
    report_path = tmp_path / "report.json"
    code = eval_main(
        [
            "--dataset", "real_anonymised", "--runs", "1",
            "--write-report", str(report_path), *dir_args(dataset_dirs),
        ]
    )
    assert code == 0
    raw = report_path.read_text(encoding="utf-8")
    assert "pupil_page_photo" not in raw
    assert "(id withheld" in raw


def test_write_report_empty_dataset_writes_minimal_report(dataset_dirs, tmp_path, capsys):
    report_path = tmp_path / "report.json"
    code = eval_main(
        [
            "--dataset", "real_anonymised", "--runs", "1",
            "--write-report", str(report_path), *dir_args(dataset_dirs),
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"] == {
        "samples": 0, "evaluated": 0, "skipped": 0, "failed": 0,
    }
    assert "mandatory" in report["note"]
    out = capsys.readouterr().out
    assert "minimal baseline report" in out
    assert "baseline report written" in out


# --- Diagnostics helpers -----------------------------------------------------------


def test_bucket_classification():
    assert _bucket({"cer": 0.05, "failed": False, "empty_draft": False}) == "low"
    assert _bucket({"cer": 0.20, "failed": False, "empty_draft": False}) == "medium"
    assert _bucket({"cer": 0.50, "failed": False, "empty_draft": False}) == "high"
    assert _bucket({"cer": 1.00, "failed": True, "empty_draft": True}) == "failed"


def test_calibration_note_flags_overconfidence():
    over = {"low": [0.9], "medium": [], "high": [0.8, 0.85], "failed": []}
    assert "over-optimistic" in _calibration_note(over)
    honest = {"low": [0.9], "medium": [], "high": [0.3], "failed": [0.0]}
    assert "honest" in _calibration_note(honest)
    no_data = {"low": [0.9], "medium": [], "high": [], "failed": []}
    assert "cannot be judged" in _calibration_note(no_data)


# --- Existing datasets unaffected -----------------------------------------------------


def test_original_dataset_evaluation_still_works(capsys):
    code = eval_main(["--dataset", "original", "--runs", "1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "images=5" in out and "failed=0" in out
