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
def test_search_returns_ranked_hits(client, monkeypatch, tmp_path):
    def fake_search(query, k=5):
        return [
            {"source": "data/documents/foo.txt", "page": "N/A", "snippet": "abc", "score": 0.9},
        ]

    monkeypatch.setattr("cortex.query.search_with_scores", fake_search)
    # Keep the search event out of the repo's real query log.
    monkeypatch.setattr("cortex_api.metrics.QUERY_LOG_PATH", str(tmp_path / "events.jsonl"))
    res = client.post("/api/search", json={"query": "foo"})
    assert res.status_code == 200
    hits = res.json()["results"]
    assert hits[0]["source"] == "foo.txt"  # basename, prefix stripped
    assert hits[0]["page"] is None  # "N/A" normalized away
    assert hits[0]["score"] == 0.9


# ── Stats (Overview) ──────────────────────────────────────────────────────────
def test_stats_aggregates_real_data(client, monkeypatch, tmp_path):
    import json
    from datetime import UTC, datetime

    # Document corpus.
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "skip.bin").write_text("z")
    monkeypatch.setattr("cortex_api.metrics.DATA_DIR", str(tmp_path))

    # RL router metrics.
    rl_dir = tmp_path / "rl"
    rl_dir.mkdir()
    (rl_dir / "metrics.json").write_text(
        json.dumps({"overall_accuracy": 0.8, "total_predictions": 10})
    )
    monkeypatch.setattr("cortex_api.metrics.RL_FEEDBACK_DIR", str(rl_dir))

    # Query event log: one chat + one search, both "today".
    log = tmp_path / "events.jsonl"
    now = datetime.now(UTC).isoformat()
    log.write_text(
        json.dumps({"timestamp": now, "kind": "chat", "route": "rag_doc", "query": "q1"})
        + "\n"
        + json.dumps({"timestamp": now, "kind": "search", "route": "search", "query": "q2"})
        + "\n"
    )
    monkeypatch.setattr("cortex_api.metrics.QUERY_LOG_PATH", str(log))

    res = client.get("/api/stats")
    assert res.status_code == 200
    body = res.json()
    assert body["documentsIndexed"] == 2  # .bin ignored
    assert body["queriesToday"] == 2
    assert body["queriesLast7Days"] == 2
    assert body["ragQueries"] == 2  # rag_doc + search both count as retrieval
    assert body["chatQueries"] == 0
    assert body["routerAccuracy"] == 0.8
    assert body["totalPredictions"] == 10
    assert len(body["volumeSeries"]) == 7
    assert body["volumeSeries"][-1]["queries"] == 2  # today is the last point
    assert len(body["activity"]) == 2


def test_stats_empty_falls_back_to_system_facts(client, monkeypatch, tmp_path):
    monkeypatch.setattr("cortex_api.metrics.DATA_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr("cortex_api.metrics.RL_FEEDBACK_DIR", str(tmp_path / "norl"))
    monkeypatch.setattr("cortex_api.metrics.QUERY_LOG_PATH", str(tmp_path / "none.jsonl"))

    res = client.get("/api/stats")
    assert res.status_code == 200
    body = res.json()
    assert body["documentsIndexed"] == 0
    assert body["queriesToday"] == 0
    assert body["routerAccuracy"] is None
    assert len(body["activity"]) == 1  # "No activity yet" fallback


def test_record_query_appends_event(monkeypatch, tmp_path):
    from cortex_api.metrics import record_query

    log = tmp_path / "nested" / "events.jsonl"
    monkeypatch.setattr("cortex_api.metrics.QUERY_LOG_PATH", str(log))
    record_query("rag", kind="chat", query="hello", confidence=0.91)

    import json

    line = log.read_text().strip()
    event = json.loads(line)
    assert event["route"] == "rag"
    assert event["kind"] == "chat"
    assert event["query"] == "hello"
    assert event["confidence"] == 0.91


# ── Voice: STT / TTS ──────────────────────────────────────────────────────────
def test_stt_transcribes(client, monkeypatch):
    monkeypatch.setattr("cortex_api.services.stt_service.transcribe_bytes", lambda b: "hello world")
    res = client.post("/api/stt", files={"file": ("a.webm", b"audiobytes", "audio/webm")})
    assert res.status_code == 200
    assert res.json()["text"] == "hello world"


def test_stt_rejects_empty(client):
    res = client.post("/api/stt", files={"file": ("a.webm", b"", "audio/webm")})
    assert res.status_code == 400


def test_tts_returns_wav(client, monkeypatch):
    monkeypatch.setattr("cortex_api.services.tts_service.synthesize", lambda text: b"RIFFfakewav")
    res = client.post("/api/tts", json={"text": "speak this"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content.startswith(b"RIFF")


def test_tts_rejects_empty(client):
    res = client.post("/api/tts", json={"text": "   "})
    assert res.status_code == 400


# ── Vision ────────────────────────────────────────────────────────────────────
def test_vlm_describes_image(client, monkeypatch):
    monkeypatch.setattr("image_understanding.vlm.describe_image", lambda path, prompt: "a cat")
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
