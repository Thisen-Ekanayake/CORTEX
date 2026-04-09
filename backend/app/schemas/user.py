"""User-related Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


# ── Auth ──────────────────────────────────────────────
class UserRegister(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8)
    display_name: str = Field(..., max_length=100)


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Profile ───────────────────────────────────────────
class UserProfileOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    avatar_initials: str
    avatar_gradient: str
    two_factor_enabled: bool
    session_timeout_enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UserProfileUpdate(BaseModel):
    display_name: str | None = None
    avatar_initials: str | None = None
    avatar_gradient: str | None = None


# ── Password ──────────────────────────────────────────
class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


# ── Preferences ──────────────────────────────────────
class PreferencesOut(BaseModel):
    dark_mode: bool
    high_contrast: bool
    animations_enabled: bool
    compact_sidebar: bool
    email_alerts: bool

    model_config = {"from_attributes": True}


class PreferencesUpdate(BaseModel):
    dark_mode: bool | None = None
    high_contrast: bool | None = None
    animations_enabled: bool | None = None
    compact_sidebar: bool | None = None
    email_alerts: bool | None = None


# ── Security ──────────────────────────────────────────
class TwoFactorUpdate(BaseModel):
    enabled: bool
