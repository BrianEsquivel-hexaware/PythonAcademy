# Write tests/test_api.py to disk

import app
from fastapi.testclient import TestClient

client = TestClient(app.app)


def test_normalize_text_happy_path():
    payload = {"text": "Hello Data Engineering world!", "lowercase": False}
    resp = client.post("/normalize-text", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "normalized_text" in data
    assert "char_count" in data
    assert "word_count" in data
    assert isinstance(data["char_count"], int)
    assert isinstance(data["word_count"], int)


def test_normalize_text_invalid_payload_returns_422():
    payload = {"text": "Too short", "lowercase": False}
    resp = client.post("/normalize-text", json=payload)
    assert resp.status_code == 422


def test_normalize_text_lowercase_changes_output():
    payload = {"text": "Hello Data Engineering world!", "lowercase": True}
    resp = client.post("/normalize-text", json=payload)
    assert resp.status_code == 200
    assert resp.json()["normalized_text"].startswith("hello")


def test_keywords_simple_happy_path():
    payload = {
        "text": "Data Engineering is engineering data. Data quality matters a lot.",
        "top_k": 3,
    }
    resp = client.post("/keywords-simple", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "keywords" in data
    assert isinstance(data["keywords"], list)
    assert len(data["keywords"]) <= payload["top_k"]


def test_keywords_simple_top_k_invalid_returns_400():
    for bad in [0, 21]:
        payload = {"text": "some text with enough length to pass validation", "top_k": bad}
        resp = client.post("/keywords-simple", json=payload)
        assert resp.status_code == 400
