"""Data export service — generates ZIP archive of user data."""

import io
import json
import zipfile
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.user import User


def generate_export(db: Session, user: User) -> io.BytesIO:
    """Generate a ZIP archive containing the user's exportable data."""
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Profile
        profile = {
            "id": str(user.id),
            "email": user.email,
            "display_name": user.display_name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }
        zf.writestr("profile.json", json.dumps(profile, indent=2))

        # Preferences
        if user.preferences:
            prefs = {
                "dark_mode": user.preferences.dark_mode,
                "high_contrast": user.preferences.high_contrast,
                "animations_enabled": user.preferences.animations_enabled,
                "compact_sidebar": user.preferences.compact_sidebar,
                "email_alerts": user.preferences.email_alerts,
            }
            zf.writestr("preferences.json", json.dumps(prefs, indent=2))

        # Notification preferences
        if user.notification_preferences:
            notif = {
                "email_weekly_digest": user.notification_preferences.email_weekly_digest,
                "email_product_updates": user.notification_preferences.email_product_updates,
                "push_mentions": user.notification_preferences.push_mentions,
                "push_system_updates": user.notification_preferences.push_system_updates,
                "slack_enabled": user.notification_preferences.slack_enabled,
                "slack_channel": user.notification_preferences.slack_channel,
            }
            zf.writestr("notification_preferences.json", json.dumps(notif, indent=2))

        # Sessions
        sessions = []
        for s in user.sessions:
            sessions.append({
                "device": s.device,
                "location": s.location,
                "ip_address": s.ip_address,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            })
        zf.writestr("sessions.json", json.dumps(sessions, indent=2))

        # Export metadata
        meta = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        }
        zf.writestr("export_meta.json", json.dumps(meta, indent=2))

    buf.seek(0)
    return buf
