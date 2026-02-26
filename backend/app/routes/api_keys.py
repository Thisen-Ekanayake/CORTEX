"""API key management routes."""

import uuid
import secrets
import hashlib

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.api_key import ApiKey
from app.models.workspace import WorkspaceMember
from app.schemas.api_key import ApiKeyOut, ApiKeyCreate, ApiKeyCreated

router = APIRouter(prefix="/api/api-keys", tags=["API Keys"])


def _generate_key() -> tuple[str, str, str]:
    """Generate a random API key, returning (full_key, prefix, hash)."""
    raw = secrets.token_urlsafe(32)
    full_key = f"sk_live_{raw}"
    prefix = full_key[:15] + "..."
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, prefix, key_hash


@router.get("", response_model=list[ApiKeyOut])
def list_api_keys(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all API keys for the current user."""
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == current_user.id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
    return keys


@router.post("", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
def create_api_key(
    payload: ApiKeyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a new API key. The full key is returned only once."""
    # Find workspace the user belongs to
    membership = db.query(WorkspaceMember).filter(
        WorkspaceMember.user_id == current_user.id
    ).first()
    if not membership:
        raise HTTPException(
            status_code=400,
            detail="You must belong to a workspace to create API keys.",
        )

    full_key, prefix, key_hash = _generate_key()

    api_key = ApiKey(
        user_id=current_user.id,
        workspace_id=membership.workspace_id,
        key_prefix=prefix,
        key_hash=key_hash,
        label=payload.label,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return ApiKeyCreated(
        id=api_key.id,
        key=full_key,
        key_prefix=prefix,
        label=api_key.label,
        created_at=api_key.created_at,
    )


@router.post("/{key_id}/rotate", response_model=ApiKeyCreated)
def rotate_api_key(
    key_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Rotate an API key — revokes the old one and generates a new key."""
    old_key = (
        db.query(ApiKey)
        .filter(ApiKey.id == key_id, ApiKey.user_id == current_user.id)
        .first()
    )
    if not old_key:
        raise HTTPException(status_code=404, detail="API key not found.")

    from datetime import datetime, timezone
    old_key.revoked_at = datetime.now(timezone.utc)

    full_key, prefix, key_hash = _generate_key()

    new_key = ApiKey(
        user_id=current_user.id,
        workspace_id=old_key.workspace_id,
        key_prefix=prefix,
        key_hash=key_hash,
        label=old_key.label,
    )
    db.add(new_key)
    db.commit()
    db.refresh(new_key)

    return ApiKeyCreated(
        id=new_key.id,
        key=full_key,
        key_prefix=prefix,
        label=new_key.label,
        created_at=new_key.created_at,
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(
    key_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Revoke (soft-delete) an API key."""
    api_key = (
        db.query(ApiKey)
        .filter(ApiKey.id == key_id, ApiKey.user_id == current_user.id)
        .first()
    )
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found.")

    from datetime import datetime, timezone
    api_key.revoked_at = datetime.now(timezone.utc)
    db.commit()
