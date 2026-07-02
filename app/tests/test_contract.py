"""Contract tests: /ocr must accept the InsightEd request shape and always
return the exact expected response shape as controlled JSON.

Authentication behaviour is covered in test_auth.py."""

from __future__ import annotations

from app.tests.helpers import (
    EXPECTED_FLAG_KEYS,
    EXPECTED_PAGE_KEYS,
    EXPECTED_RESPONSE_KEYS,
    make_payload,
)


def test_valid_request_returns_exact_contract_shape(client):
    response = client.post("/ocr", json=make_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == EXPECTED_RESPONSE_KEYS
    assert isinstance(body["draftText"], str)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["providerRequestId"]
    assert isinstance(body["rawCells"], list)
    assert isinstance(body["flags"], list)
    for flag in body["flags"]:
        assert set(flag.keys()) == EXPECTED_FLAG_KEYS
    assert len(body["pageResults"]) == 1
    page = body["pageResults"][0]
    assert set(page.keys()) == EXPECTED_PAGE_KEYS
    assert page["pageNumber"] == 1


def test_optional_subject_and_year_group_accepted(client):
    payload = make_payload(subject="Science", yearGroup="Year 9")
    response = client.post("/ocr", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert any(f["category"] == "subject_specific_term" for f in body["flags"])


def test_pdf_returns_controlled_json_with_clear_flag(client):
    payload = make_payload(
        mimeType="application/pdf",
        dataUrl="data:application/pdf;base64,QUJDRA==",
        fileName="scan.pdf",
    )
    response = client.post("/ocr", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == EXPECTED_RESPONSE_KEYS
    assert body["draftText"] == ""
    assert body["confidence"] == 0.0
    assert body["rawBraille"] is None
    assert body["rawCells"] == []
    assert any("PDF" in flag["reason"] for flag in body["flags"])
    assert all(flag["severity"] in {"low", "medium", "high"} for flag in body["flags"])


def test_unsupported_mime_type_returns_controlled_json(client):
    payload = make_payload(mimeType="image/gif")
    response = client.post("/ocr", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["draftText"] == ""
    assert body["confidence"] == 0.0
    assert any("Unsupported MIME type" in flag["reason"] for flag in body["flags"])


def test_missing_required_field_returns_422(client):
    payload = make_payload()
    del payload["dataUrl"]
    response = client.post("/ocr", json=payload)
    assert response.status_code == 422
    assert response.json()  # still valid JSON
