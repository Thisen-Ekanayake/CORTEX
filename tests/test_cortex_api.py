"""Tests for the cortex_api FastAPI service.

Heavy model work is monkeypatched at the service boundary, so these tests run
without GPU/models. FastAPI and httpx are optional in the lightweight CI test
environment, so the whole module is skipped when they are unavailable.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from cortex_api.main import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────
def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ── Chat (SSE) ────────────────────────────────────────────────────────────────
def test_chat_streams_sse(client, monkeypatch):
    def fake_stream(query):
        assert query == "hi"
        yield 'data: {"type": "token", "value": "Hi"}\n\n'
        yield 'data: {"type": "done", "route": "chat", "scores": {}, "sources": []}\n\n'

    monkeypatch.setattr("cortex_api.routes.chat.stream_chat", fake_stream)
    res = client.post("/api/chat", json={"query": "hi"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert '"value": "Hi"' in res.text
    assert '"type": "done"' in res.text


# ── Documents ─────────────────────────────────────────────────────────────────
def test_list_documents(client, monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "ignore.bin").write_text("nope")
    monkeypatch.setattr("cortex_api.routes.documents.DATA_DIR", str(tmp_path))

    res = client.get("/api/documents")
    assert res.status_code == 200
    body = res.json()
    names = [d["name"] for d in body["documents"]]
    assert "a.txt" in names
    assert "ignore.bin" not in names
    assert body["indexedCount"] == 1


def test_upload_document_ingests(client, monkeypatch, tmp_path):
    monkeypatch.setattr("cortex_api.routes.documents.DATA_DIR", str(tmp_path))
    monkeypatch.setattr("cortex.ingest.ingest_file", lambda path: 3)

    res = client.post(
        "/api/documents",
        files={"file": ("notes.txt", b"some content", "text/plain")},
    )
    assert res.status_code == 200
    assert res.json()["chunksAdded"] == 3
    assert (tmp_path / "notes.txt").exists()


def test_upload_rejects_unsupported_type(client, monkeypatch, tmp_path):
    monkeypatch.setattr("cortex_api.routes.documents.DATA_DIR", str(tmp_path))
    res = client.post(
        "/api/documents",
        files={"file": ("malware.exe", b"x", "application/octet-stream")},
    )
    assert res.status_code == 400


# ── Search ────────────────────────────────────────────────────────────────────
def test_search_returns_ranked_hits(client, monkeypatch):
    def fake_search(query, k=5):
        return [
            {"source": "data/documents/foo.txt", "page": "N/A", "snippet": "abc", "score": 0.9},
        ]

    monkeypatch.setattr("cortex.query.search_with_scores", fake_search)
    res = client.post("/api/search", json={"query": "foo"})
    assert res.status_code == 200
    hits = res.json()["results"]
    assert hits[0]["source"] == "foo.txt"  # basename, prefix stripped
    assert hits[0]["page"] is None  # "N/A" normalized away
    assert hits[0]["score"] == 0.9


# ── Voice: STT / TTS ──────────────────────────────────────────────────────────
def test_stt_transcribes(client, monkeypatch):
    monkeypatch.setattr(
        "cortex_api.services.stt_service.transcribe_bytes", lambda b: "hello world"
    )
    res = client.post("/api/stt", files={"file": ("a.webm", b"audiobytes", "audio/webm")})
    assert res.status_code == 200
    assert res.json()["text"] == "hello world"


def test_stt_rejects_empty(client):
    res = client.post("/api/stt", files={"file": ("a.webm", b"", "audio/webm")})
    assert res.status_code == 400


def test_tts_returns_wav(client, monkeypatch):
    monkeypatch.setattr(
        "cortex_api.services.tts_service.synthesize", lambda text: b"RIFFfakewav"
    )
    res = client.post("/api/tts", json={"text": "speak this"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content.startswith(b"RIFF")


def test_tts_rejects_empty(client):
    res = client.post("/api/tts", json={"text": "   "})
    assert res.status_code == 400


# ── Vision ────────────────────────────────────────────────────────────────────
def test_vlm_describes_image(client, monkeypatch):
    monkeypatch.setattr(
        "image_understanding.vlm.describe_image", lambda path, prompt: "a cat"
    )
    res = client.post(
        "/api/vlm",
        files={"image": ("cat.png", b"\x89PNGfake", "image/png")},
        data={"prompt": "what is this?"},
    )
    assert res.status_code == 200
    assert res.json()["description"] == "a cat"


def test_vlm_rejects_empty(client):
    res = client.post(
        "/api/vlm",
        files={"image": ("cat.png", b"", "image/png")},
        data={"prompt": "what is this?"},
    )
    assert res.status_code == 400
