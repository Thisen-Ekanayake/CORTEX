"""NotificationPreference ORM model."""

import uuid

from sqlalchemy import String, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class NotificationPreference(Base):
    """Per-user notification settings (Notifications tab)."""

    __tablename__ = "notification_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True
    )

    # Email
    email_weekly_digest: Mapped[bool] = mapped_column(Boolean, default=True)
    email_product_updates: Mapped[bool] = mapped_column(Boolean, default=True)

    # Push
    push_mentions: Mapped[bool] = mapped_column(Boolean, default=True)
    push_system_updates: Mapped[bool] = mapped_column(Boolean, default=False)

    # Integrations
    slack_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    slack_channel: Mapped[str] = mapped_column(String(100), default="#general")

    # Relationship
    user: Mapped["User"] = relationship(back_populates="notification_preferences")

    def __repr__(self) -> str:
        return f"<NotificationPreference user_id={self.user_id}>"


from app.models.user import User  # noqa: E402
