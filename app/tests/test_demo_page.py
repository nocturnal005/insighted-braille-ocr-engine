"""Tests for the Stage 3D-H1 optional local demo page (GET /demo).

The page is disabled by default (404) and, when enabled, serves a
self-contained HTML viewer that posts to the existing /ocr endpoint. These
tests check the gate, the draft-only wording, and that the page pulls no
external resources (it must work fully offline and same-origin).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import app


@pytest.fixture()
def demo_client(monkeypatch):
    monkeypatch.setenv("DEMO_PAGE_ENABLED", "true")
    config.get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        config.get_settings.cache_clear()


def test_demo_page_disabled_by_default(monkeypatch):
    # Isolate from ambient environment so this asserts the code default.
    monkeypatch.delenv("DEMO_PAGE_ENABLED", raising=False)
    config.get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/demo")
        assert response.status_code == 404
        assert "disabled" in response.json()["detail"]
    finally:
        config.get_settings.cache_clear()


def test_demo_page_serves_html_when_enabled(demo_client):
    response = demo_client.get("/demo")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_demo_page_keeps_draft_only_warning_visible(demo_client):
    body = demo_client.get("/demo").text
    assert "Draft-only OCR" in body
    assert "QTVI" in body
    assert "unverified draft" in body
    # The banner element must stay visible — no hidden attribute or similar.
    assert '<div class="draft-banner" role="alert">' in body


def test_demo_page_is_self_contained_and_same_origin(demo_client):
    """No external scripts, styles, or fonts: the page must work offline and
    send the image only to this local service."""
    body = demo_client.get("/demo").text
    assert "http://" not in body
    assert "https://" not in body
    assert 'fetch("/ocr"' in body


def test_demo_page_stays_open_when_api_key_is_set(monkeypatch):
    """Like /health and /version, GET /demo itself needs no key — the page
    has a field to supply the key to POST /ocr."""
    monkeypatch.setenv("DEMO_PAGE_ENABLED", "true")
    monkeypatch.setenv("OCR_ENGINE_API_KEY", "local-test-key")
    config.get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            assert client.get("/demo").status_code == 200
    finally:
        config.get_settings.cache_clear()
