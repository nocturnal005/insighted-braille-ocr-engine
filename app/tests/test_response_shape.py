"""Response-shape and safety tests: failures return controlled JSON, output
validates against the contract models, and logs never contain image data,
transcription text, or upload metadata."""

from __future__ import annotations

import logging

from app.models.requests import OcrRequest
from app.models.responses import OcrResponse
from app.ocr.pipeline import run_ocr
from app.tests.helpers import make_payload


def test_invalid_data_url_returns_controlled_failure(client):
    payload = make_payload(dataUrl="not-a-data-url")
    response = client.post("/ocr", json=payload)
    assert response.status_code == 200
    body = response.json()
    parsed = OcrResponse.model_validate(body)  # exact contract shape
    assert parsed.draftText == ""
    assert parsed.confidence == 0.0
    assert parsed.rawBraille is None
    assert parsed.rawCells == []
    assert len(parsed.flags) >= 1
    assert len(parsed.pageResults) == 1
    assert parsed.pageResults[0].text == ""
    assert parsed.pageResults[0].flags


def test_successful_response_validates_and_has_sane_values(client):
    response = client.post("/ocr", json=make_payload("hello world"))
    body = response.json()
    parsed = OcrResponse.model_validate(body)
    assert 0.0 <= parsed.confidence <= 1.0
    assert parsed.rawBraille
    assert parsed.rawCells
    for cell in parsed.rawCells:
        assert cell.line >= 1
        assert cell.cellIndex >= 1
        assert all(1 <= d <= 6 for d in cell.dots)
        assert len(cell.bbox) == 4
        assert 0.0 <= cell.confidence <= 1.0
    assert parsed.providerRequestId.startswith("ocr_")


def test_provider_request_id_unique_per_run(client):
    first = client.post("/ocr", json=make_payload()).json()
    second = client.post("/ocr", json=make_payload()).json()
    assert first["providerRequestId"] != second["providerRequestId"]


def test_logs_never_contain_image_data_text_or_identifiers(caplog):
    payload = make_payload("hello world")
    payload["taskId"] = "task-SECRET-RAW-TASK-ID-9876"
    payload["fileName"] = "SECRET-PUPIL-NAME.png"
    payload["title"] = "SECRET TITLE ABOUT A PUPIL"
    base64_fragment = payload["dataUrl"].split(",", 1)[1][:32]

    with caplog.at_level(logging.DEBUG):
        response = run_ocr(OcrRequest(**payload))

    assert response.draftText  # OCR actually ran
    joined_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert base64_fragment not in joined_logs
    assert "hello world" not in joined_logs
    assert "SECRET-PUPIL-NAME" not in joined_logs
    assert "SECRET TITLE" not in joined_logs
    assert "task-SECRET-RAW-TASK-ID-9876" not in joined_logs
    assert "SECRET-RAW-TASK-ID" not in joined_logs


def test_rejected_request_logs_never_contain_raw_task_id(caplog):
    payload = make_payload(
        taskId="task-SECRET-REJECTED-ID-1234",
        mimeType="application/pdf",
        dataUrl="data:application/pdf;base64,QUJDRA==",
    )

    with caplog.at_level(logging.DEBUG):
        response = run_ocr(OcrRequest(**payload))

    assert response.draftText == ""
    joined_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "task-SECRET-REJECTED-ID-1234" not in joined_logs
    assert "SECRET-REJECTED-ID" not in joined_logs
