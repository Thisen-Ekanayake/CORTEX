"""API key Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel


class ApiKeyOut(BaseModel):
    id: uuid.UUID
    key_prefix: str
    label: str
    created_at: datetime
    revoked_at: datetime | None
    is_active: bool

    model_config = {"from_attributes": True}


class ApiKeyCreate(BaseModel):
    label: str = "Production Key"


class ApiKeyCreated(BaseModel):
    """Returned only once when creating a key — contains the full plaintext key."""
    id: uuid.UUID
    key: str
    key_prefix: str
    label: str
    created_at: datetime
