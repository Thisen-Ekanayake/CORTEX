"""Project, Conversation and Message Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


# ── Projects ──────────────────────────────────────────
class ProjectCreate(BaseModel):
    name: str = Field(..., max_length=200)
    icon: str | None = Field(default=None, max_length=8)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    icon: str | None = Field(default=None, max_length=8)


class ProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    icon: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Messages ──────────────────────────────────────────
class MessageCreate(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    route: str | None = None
    sources: list[str] | None = None
    image_url: str | None = None


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    route: str | None
    sources: list[str] | None
    image_url: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Conversations ─────────────────────────────────────
class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    project_id: uuid.UUID | None = None


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    # Include project_id (even as null) to move the conversation; omit it to
    # leave the current project unchanged. The route inspects model_fields_set.
    project_id: uuid.UUID | None = None


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    project_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]
