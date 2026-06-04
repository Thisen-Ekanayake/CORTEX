"""Chat streaming service.

Bridges ``cortex.router.execute`` (which streams LLM tokens through a
``StreamHandler`` callback) onto a Server-Sent Events byte stream that the web
UI can consume incrementally.

The producer (``execute``) runs in a worker thread and pushes tokens onto a
thread-safe queue; the SSE generator drains that queue. Starlette iterates a
sync generator in a threadpool, so the blocking ``queue.get`` is safe.
"""

import json
import logging
import os
import queue
import threading
from collections.abc import Iterator

from cortex_api.metrics import record_query

logger = logging.getLogger(__name__)

# Sentinel marking the end of the token stream.
_DONE = object()


def _prettify_source(src: str) -> str:
    """Turn ``"data/documents/foo.txt, page N/A"`` into ``"foo.txt"``.

    Keeps a meaningful page number when present (e.g. PDFs).
    """
    name, _, page = src.partition(", page ")
    name = os.path.basename(name.strip())
    page = page.strip()
    if page and page.upper() != "N/A":
        return f"{name} (p.{page})"
    return name


def stream_chat(query: str) -> Iterator[str]:
    """Yield SSE-formatted strings for a single chat turn.

    Event payloads (one JSON object per ``data:`` line):
      - ``{"type": "token", "value": "<str>"}`` — one streamed token
      - ``{"type": "done", "route": "<str>", "scores": {...}, "sources": [...]}``
      - ``{"type": "error", "message": "<str>"}``
    """
    # Imported lazily so module import / app startup stays light.
    from cortex.query import get_sources
    from cortex.router import Route, execute
    from cortex.streaming import StreamHandler

    token_q: queue.Queue = queue.Queue()
    result_box: dict = {}

    def on_token(tok: str) -> None:
        token_q.put(tok)

    def worker() -> None:
        try:
            _result, route, scores = execute(query, callbacks=[StreamHandler(on_token)])
            sources: list[str] = []
            if route in {Route.RAG, Route.RAG_DOC}:
                # De-duplicate while preserving order.
                seen: set[str] = set()
                for raw in get_sources(query):
                    src = _prettify_source(raw)
                    if src not in seen:
                        seen.add(src)
                        sources.append(src)
            result_box["route"] = route.value
            result_box["scores"] = scores or {}
            result_box["sources"] = sources
            top_conf = (scores or {}).get(route.value)
            if top_conf is None and scores:
                top_conf = max(scores.values())
            record_query(route.value, kind="chat", query=query, confidence=top_conf)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            logger.exception("chat execution failed for query %r", query)
            result_box["error"] = str(exc)
        finally:
            token_q.put(_DONE)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = token_q.get()
        if item is _DONE:
            break
        yield f"data: {json.dumps({'type': 'token', 'value': item})}\n\n"

    thread.join(timeout=5)

    if "error" in result_box:
        yield f"data: {json.dumps({'type': 'error', 'message': result_box['error']})}\n\n"
        return

    done = {
        "type": "done",
        "route": result_box.get("route", "chat"),
        "scores": result_box.get("scores", {}),
        "sources": result_box.get("sources", []),
    }
    yield f"data: {json.dumps(done)}\n\n"
