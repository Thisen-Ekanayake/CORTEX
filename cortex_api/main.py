"""FastAPI application entry point for the CORTEX AI service.

This service is intentionally separate from ``backend/`` (the user-management
API). It exposes the ``cortex`` AI core — chat/RAG, document ingestion,
semantic search, STT, TTS and VLM — to the web UI.

Run from the repository root::

    uvicorn cortex_api.main:app --port 8001 --reload
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Origins allowed to call this API (the Vite dev server by default).
CORS_ORIGINS = os.getenv(
    "CORTEX_API_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

app = FastAPI(
    title="CORTEX AI API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["System"])
def health_check() -> dict:
    """Liveness probe used by the web UI and ops tooling."""
    return {"status": "ok", "service": "cortex-ai-api"}


# ── Route registration ────────────────────────────────────────────────────────
# Route modules keep heavy ML imports inside their service layer (lazy
# singletons), so importing them here does not load any models at startup.
from cortex_api.routes import chat, documents, search, vision, voice  # noqa: E402

app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(voice.router)
app.include_router(vision.router)
