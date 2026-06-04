"""Stats route — real Overview metrics for the web UI dashboard."""

from fastapi import APIRouter

from cortex_api.metrics import build_stats

router = APIRouter(prefix="/api", tags=["Stats"])


@router.get("/stats")
def stats() -> dict:
    """Aggregate document, query-volume and router-accuracy metrics."""
    return build_stats()
