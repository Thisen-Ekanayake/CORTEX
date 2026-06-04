"""Query-event logging and Overview statistics.

This module backs the web UI Overview panel with *real* numbers instead of
placeholder data. It has two responsibilities:

1. ``record_query`` — append one JSON line per query the API handles (chat or
   search) to a lightweight event log. This is best-effort: a logging failure
   must never break the request that triggered it.
2. ``build_stats`` — aggregate that event log, the RL router metrics
   (``rl_feedback/metrics.json``) and the document corpus (``DATA_DIR``) into
   the figures rendered on the Overview page.

Everything here is plain stdlib + file I/O, so importing it does not pull in any
ML dependencies and is safe at app-startup time.
"""

import json
import logging
import os
import threading
from collections import Counter
from datetime import UTC, datetime, timedelta

from cortex.config import DATA_DIR, RL_FEEDBACK_DIR

logger = logging.getLogger(__name__)

# Where per-query events are persisted (one JSON object per line). Override with
# CORTEX_QUERY_LOG; defaults to logs/query_events.jsonl under the repo root.
QUERY_LOG_PATH: str = os.getenv("CORTEX_QUERY_LOG", os.path.join("logs", "query_events.jsonl"))

# Routes that involve document/image retrieval (the "RAG" series on the chart).
RAG_ROUTES = {"rag", "rag_doc", "rag_img"}

# Document extensions counted toward "Documents indexed" (mirrors the documents
# route's type labels).
_DOC_EXTS = {".pdf", ".txt", ".docx", ".md"}

# Human-friendly labels for the activity feed.
_ROUTE_LABELS = {
    "chat": "Chat",
    "meta": "Meta",
    "rag": "RAG",
    "rag_doc": "RAG (document)",
    "rag_img": "RAG (image)",
    "search": "Search",
}

_LOCK = threading.Lock()


def record_query(
    route: str,
    *,
    kind: str = "chat",
    query: str = "",
    confidence: float | None = None,
) -> None:
    """Append a single query event to the log. Never raises.

    ``kind`` distinguishes the originating endpoint ("chat" or "search");
    ``query`` is stored truncated so the activity feed can show real queries.
    """
    event: dict = {
        "timestamp": datetime.now(UTC).isoformat(),
        "kind": kind,
        "route": route,
        "query": (query or "")[:160],
    }
    if confidence is not None:
        event["confidence"] = round(float(confidence), 4)

    try:
        with _LOCK:
            parent = os.path.dirname(QUERY_LOG_PATH)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(QUERY_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
    except OSError:  # disk full, permissions, etc. — telemetry must not break chat
        logger.warning("failed to record query event", exc_info=True)


# ── Aggregation ─────────────────────────────────────────────────────────────--


def _parse_ts(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _read_events() -> list[dict]:
    if not os.path.isfile(QUERY_LOG_PATH):
        return []
    events: list[dict] = []
    try:
        with open(QUERY_LOG_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        logger.warning("failed to read query event log", exc_info=True)
    return events


def _count_documents() -> int:
    if not os.path.isdir(DATA_DIR):
        return 0
    return sum(
        1
        for name in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, name))
        and os.path.splitext(name)[1].lower() in _DOC_EXTS
    )


def _router_metrics() -> tuple[float | None, int]:
    """Return (overall_accuracy, total_predictions) from the RL metrics file."""
    path = os.path.join(RL_FEEDBACK_DIR, "metrics.json")
    if not os.path.isfile(path):
        return None, 0
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None, 0
    accuracy = data.get("overall_accuracy")
    accuracy = float(accuracy) if isinstance(accuracy, int | float) else None
    total = data.get("total_predictions", 0)
    total = int(total) if isinstance(total, int | float) else 0
    return accuracy, total


def _relative_time(dt: datetime, now: datetime) -> str:
    delta = max(0, int((now - dt).total_seconds()))
    if delta < 60:
        return "just now"
    for secs, label in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if delta >= secs:
            n = delta // secs
            return f"{n} {label}{'s' if n != 1 else ''} ago"
    return "just now"


def _is_rag(event: dict) -> bool:
    return event.get("route") in RAG_ROUTES or event.get("kind") == "search"


def _build_activity(
    parsed: list[tuple[datetime, dict]],
    now: datetime,
    documents: int,
    accuracy: float | None,
    total_predictions: int,
) -> list[dict]:
    """Most-recent real query events, or system facts when none exist yet."""
    activity: list[dict] = []
    for ts, event in sorted(parsed, key=lambda p: p[0], reverse=True)[:5]:
        route = event.get("route", "")
        label = _ROUTE_LABELS.get(route, route or "Unknown")
        query = (event.get("query") or "").strip()
        message = f'Routed to {label}: "{query}"' if query else f"Routed to {label}"
        activity.append(
            {
                "id": ts.isoformat(),
                "type": "success" if _is_rag(event) else "info",
                "message": message,
                "time": _relative_time(ts, now),
            }
        )

    if activity:
        return activity

    # No queries logged yet — surface real system state instead of fake alerts.
    if documents:
        activity.append(
            {
                "id": "docs",
                "type": "info",
                "message": f"{documents} document{'s' if documents != 1 else ''} indexed in the corpus.",
                "time": "",
            }
        )
    if accuracy is not None and total_predictions:
        activity.append(
            {
                "id": "router",
                "type": "success",
                "message": (
                    f"Router accuracy {accuracy * 100:.0f}% "
                    f"over {total_predictions} prediction"
                    f"{'s' if total_predictions != 1 else ''}."
                ),
                "time": "",
            }
        )
    if not activity:
        activity.append(
            {
                "id": "empty",
                "type": "info",
                "message": "No activity yet — ask a question to get started.",
                "time": "",
            }
        )
    return activity


def build_stats() -> dict:
    """Aggregate real Overview statistics from the event log, RL metrics and corpus."""
    now = datetime.now(UTC)
    today = now.date()
    yesterday = today - timedelta(days=1)
    week_start = today - timedelta(days=6)

    parsed: list[tuple[datetime, dict]] = []
    for event in _read_events():
        ts = _parse_ts(event.get("timestamp"))
        if ts is not None:
            parsed.append((ts, event))

    queries_today = sum(1 for ts, _ in parsed if ts.date() == today)
    queries_yesterday = sum(1 for ts, _ in parsed if ts.date() == yesterday)
    last7 = [(ts, e) for ts, e in parsed if ts.date() >= week_start]

    rag_queries = sum(1 for _, e in last7 if _is_rag(e))
    chat_queries = len(last7) - rag_queries

    by_day_total: Counter = Counter()
    by_day_rag: Counter = Counter()
    for ts, event in last7:
        day = ts.date()
        by_day_total[day] += 1
        if _is_rag(event):
            by_day_rag[day] += 1

    volume_series = []
    for i in range(7):
        day = week_start + timedelta(days=i)
        volume_series.append(
            {
                "name": day.strftime("%a"),
                "date": day.isoformat(),
                "queries": by_day_total.get(day, 0),
                "rag": by_day_rag.get(day, 0),
            }
        )

    documents = _count_documents()
    accuracy, total_predictions = _router_metrics()

    return {
        "documentsIndexed": documents,
        "queriesToday": queries_today,
        "queriesYesterday": queries_yesterday,
        "queriesLast7Days": len(last7),
        "ragQueries": rag_queries,
        "chatQueries": chat_queries,
        "routerAccuracy": accuracy,
        "totalPredictions": total_predictions,
        "volumeSeries": volume_series,
        "activity": _build_activity(parsed, now, documents, accuracy, total_predictions),
    }
