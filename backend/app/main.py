"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import auth, users, preferences, security, workspaces, notifications, api_keys

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow the Vite dev server and any configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(preferences.router)
app.include_router(security.router)
app.include_router(workspaces.router)
app.include_router(notifications.router)
app.include_router(api_keys.router)


# ── Data export endpoint (doesn't fit neatly into a CRUD router) ──
from fastapi import Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.services.export_service import generate_export


@app.post("/api/export", tags=["Data"])
def export_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export all user data as a ZIP archive."""
    buf = generate_export(db, current_user)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=cortex_data_export.zip"},
    )


@app.get("/api/health", tags=["System"])
def health_check():
    """Simple health check."""
    return {"status": "ok", "service": settings.app_name}
