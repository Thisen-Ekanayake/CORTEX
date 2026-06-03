"""Search route — scored semantic retrieval over the vector store."""

import os

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Search"])


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@router.post("/search")
def search(req: SearchRequest) -> dict:
    """Return ranked chunks with relevance scores for ``req.query``."""
    from cortex.query import search_with_scores

    results = []
    for hit in search_with_scores(req.query, k=req.k):
        page = hit.get("page", "")
        results.append(
            {
                "source": os.path.basename(hit["source"]),
                "snippet": hit["snippet"],
                "score": hit["score"],
                "page": page if page and page.upper() != "N/A" else None,
            }
        )
    return {"query": req.query, "results": results}
