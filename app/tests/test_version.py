from __future__ import annotations


def test_version_returns_identity_and_draft_warning(client):
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "insighted-braille-ocr-engine"
    assert body["version"]
    assert "draft" in body["warning"].lower()
    assert "specialist" in body["warning"].lower()
    assert set(body["supportedMimeTypes"]) == {"image/png", "image/jpeg", "image/jpg"}
