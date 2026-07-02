"""Authentication tests for POST /ocr.

When OCR_ENGINE_API_KEY is set, the key is accepted via either
X-API-Key: <key> or Authorization: Bearer <key> (the form InsightEd AI's
external_braille_ocr adapter sends). /health and /version stay open.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import app
from app.tests.helpers import make_payload

TEST_KEY = "local-test-key"


@pytest.fixture()
def secured_client(monkeypatch):
    monkeypatch.setenv("OCR_ENGINE_API_KEY", TEST_KEY)
    config.get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        config.get_settings.cache_clear()


def test_x_api_key_header_accepted(secured_client):
    response = secured_client.post(
        "/ocr", json=make_payload(), headers={"X-API-Key": TEST_KEY}
    )
    assert response.status_code == 200


def test_authorization_bearer_accepted(secured_client):
    response = secured_client.post(
        "/ocr", json=make_payload(), headers={"Authorization": f"Bearer {TEST_KEY}"}
    )
    assert response.status_code == 200


def test_missing_key_returns_401(secured_client):
    response = secured_client.post("/ocr", json=make_payload())
    assert response.status_code == 401


def test_wrong_x_api_key_returns_401(secured_client):
    response = secured_client.post(
        "/ocr", json=make_payload(), headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


def test_wrong_bearer_key_returns_401(secured_client):
    response = secured_client.post(
        "/ocr", json=make_payload(), headers={"Authorization": "Bearer wrong-key"}
    )
    assert response.status_code == 401


def test_non_bearer_authorization_scheme_returns_401(secured_client):
    response = secured_client.post(
        "/ocr", json=make_payload(), headers={"Authorization": f"Basic {TEST_KEY}"}
    )
    assert response.status_code == 401


def test_empty_bearer_token_returns_401(secured_client):
    response = secured_client.post(
        "/ocr", json=make_payload(), headers={"Authorization": "Bearer "}
    )
    assert response.status_code == 401


def test_health_and_version_remain_open(secured_client):
    assert secured_client.get("/health").status_code == 200
    assert secured_client.get("/version").status_code == 200


def test_no_key_configured_allows_requests(client):
    response = client.post("/ocr", json=make_payload())
    assert response.status_code == 200
