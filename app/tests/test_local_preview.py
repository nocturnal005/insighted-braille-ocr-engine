"""Tests for the Stage 3D-H1 local preview CLI (app.demo.local_preview).

The CLI runs the unchanged pipeline on one image file and prints a
human-readable draft report (or the contract JSON with --json). These tests
check the report carries the draft-only warning, the decoded content, and
that usage errors exit with argparse's code 2 without running OCR.
"""

from __future__ import annotations

import base64
import json

import pytest

from app.demo.local_preview import main
from app.evaluation.sample_generator import image_to_data_url, render_braille_image
from app.tests.helpers import EXPECTED_RESPONSE_KEYS

DATA_URL_PREFIX = "data:image/png;base64,"


@pytest.fixture()
def sample_image(tmp_path):
    data_url = image_to_data_url(render_braille_image("hello world"))
    path = tmp_path / "preview_sample.png"
    path.write_bytes(base64.b64decode(data_url[len(DATA_URL_PREFIX):]))
    return path


def test_report_contains_banner_flags_braille_and_draft(sample_image, capsys):
    assert main([str(sample_image)]) == 0
    out = capsys.readouterr().out
    assert "DRAFT-ONLY BRAILLE OCR PREVIEW" in out
    assert "QTVI or Braille-literate specialist" in out
    assert "Confidence:" in out
    assert "Uncertainty flags" in out
    assert "hello world" in out
    assert "⠓" in out  # ⠓ = 'h': rawBraille cells are shown


def test_json_mode_prints_contract_response_and_banner_on_stderr(
    sample_image, capsys
):
    assert main([str(sample_image), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) == EXPECTED_RESPONSE_KEYS
    assert payload["draftText"] == "hello world"
    assert "DRAFT-ONLY BRAILLE OCR PREVIEW" in captured.err


def test_missing_file_exits_with_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / "does_not_exist.png")])
    assert excinfo.value.code == 2
    assert "not found" in capsys.readouterr().err


def test_unsupported_extension_exits_with_usage_error(tmp_path, capsys):
    path = tmp_path / "not_an_image.gif"
    path.write_bytes(b"GIF89a")
    with pytest.raises(SystemExit) as excinfo:
        main([str(path)])
    assert excinfo.value.code == 2
    assert "unsupported file extension" in capsys.readouterr().err


def test_unreadable_image_still_reports_safely(tmp_path, capsys):
    """A corrupt PNG must produce the controlled empty-draft report, not a
    crash — mirroring the /ocr safe-failure behaviour."""
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot really a png")
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "DRAFT-ONLY BRAILLE OCR PREVIEW" in out
    assert "(empty — no draft was produced; see flags above)" in out
