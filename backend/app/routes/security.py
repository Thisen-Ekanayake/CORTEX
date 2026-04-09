"""Security routes — 2FA toggle and session management."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.session import UserSession
from app.schemas.user import TwoFactorUpdate
from app.schemas.session import SessionOut

router = APIRouter(prefix="/api/users", tags=["Security"])


# ── 2FA ───────────────────────────────────────────────
@router.put("/me/2fa", status_code=status.HTTP_204_NO_CONTENT)
def toggle_two_factor(
    payload: TwoFactorUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Enable or disable two-factor authentication."""
    current_user.two_factor_enabled = payload.enabled
    if not payload.enabled:
        current_user.two_factor_secret = None
    db.commit()


@router.put("/me/session-timeout", status_code=status.HTTP_204_NO_CONTENT)
def toggle_session_timeout(
    enabled: bool,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Enable or disable automatic session timeout."""
    current_user.session_timeout_enabled = enabled
    db.commit()


# ── Sessions ──────────────────────────────────────────
@router.get("/me/sessions", response_model=list[SessionOut])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all active sessions for the current user."""
    return (
        db.query(UserSession)
        .filter(UserSession.user_id == current_user.id)
        .order_by(UserSession.last_active.desc())
        .all()
    )


@router.delete("/me/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Revoke (delete) a specific session."""
    session = (
        db.query(UserSession)
        .filter(UserSession.id == session_id, UserSession.user_id == current_user.id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.is_current:
        raise HTTPException(status_code=400, detail="Cannot revoke the current session.")
    db.delete(session)
    db.commit()
