"""User preferences routes (General tab — Appearance & Workflow)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.models.user import User, UserPreference
from app.schemas.user import PreferencesOut, PreferencesUpdate

router = APIRouter(prefix="/api/users", tags=["Preferences"])


@router.get("/me/preferences", response_model=PreferencesOut)
def get_preferences(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get user appearance and workflow preferences."""
    prefs = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()
    if not prefs:
        # Create defaults if missing
        prefs = UserPreference(user_id=current_user.id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    return prefs


@router.put("/me/preferences", response_model=PreferencesOut)
def update_preferences(
    payload: PreferencesUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update user appearance and workflow preferences."""
    prefs = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()
    if not prefs:
        prefs = UserPreference(user_id=current_user.id)
        db.add(prefs)

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(prefs, key, value)

    db.commit()
    db.refresh(prefs)
    return prefs
