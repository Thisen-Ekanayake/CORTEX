"""Notification preference Pydantic schemas."""

from pydantic import BaseModel


class NotificationPreferencesOut(BaseModel):
    email_weekly_digest: bool
    email_product_updates: bool
    push_mentions: bool
    push_system_updates: bool
    slack_enabled: bool
    slack_channel: str

    model_config = {"from_attributes": True}


class NotificationPreferencesUpdate(BaseModel):
    email_weekly_digest: bool | None = None
    email_product_updates: bool | None = None
    push_mentions: bool | None = None
    push_system_updates: bool | None = None
    slack_enabled: bool | None = None
    slack_channel: str | None = None
