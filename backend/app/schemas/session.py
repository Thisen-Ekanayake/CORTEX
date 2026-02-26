"""Session-related Pydantic schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel


class SessionOut(BaseModel):
    id: uuid.UUID
    device: str
    location: str
    ip_address: str
    is_current: bool
    created_at: datetime
    last_active: datetime

    model_config = {"from_attributes": True}
