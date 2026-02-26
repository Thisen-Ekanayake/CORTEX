"""Workspace-related Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class WorkspaceOut(BaseModel):
    id: uuid.UUID
    name: str
    org_id: str
    default_role: str
    owner_id: uuid.UUID
    created_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceUpdate(BaseModel):
    name: str | None = None


class WorkspacePermissionsUpdate(BaseModel):
    default_role: str = Field(..., pattern="^(member|viewer)$")


# ── Members ───────────────────────────────────────────
class MemberOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    display_name: str
    email: str
    avatar_initials: str
    avatar_gradient: str
    role: str
    joined_at: datetime


class MemberInvite(BaseModel):
    email: str
    role: str = Field(default="member", pattern="^(admin|member|viewer)$")


class MemberRoleUpdate(BaseModel):
    role: str = Field(..., pattern="^(owner|admin|member|viewer)$")
