"""Notification preference routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.notification import NotificationPreference
from app.schemas.notification import NotificationPreferencesOut, NotificationPreferencesUpdate

router = APIRouter(prefix="/api/users", tags=["Notifications"])


@router.get("/me/notifications", response_model=NotificationPreferencesOut)
def get_notifications(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get notification preferences."""
    notif = db.query(NotificationPreference).filter(
        NotificationPreference.user_id == current_user.id
    ).first()
    if not notif:
        notif = NotificationPreference(user_id=current_user.id)
        db.add(notif)
        db.commit()
        db.refresh(notif)
    return notif


@router.put("/me/notifications", response_model=NotificationPreferencesOut)
def update_notifications(
    payload: NotificationPreferencesUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update notification preferences."""
    notif = db.query(NotificationPreference).filter(
        NotificationPreference.user_id == current_user.id
    ).first()
    if not notif:
        notif = NotificationPreference(user_id=current_user.id)
        db.add(notif)

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(notif, key, value)

    db.commit()
    db.refresh(notif)
    return notif
