"""Conversation routes — chat threads and their messages, user-scoped."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.conversation import Project, Conversation, Message
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationSummary,
    ConversationDetail,
    MessageCreate,
    MessageOut,
)

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

_DEFAULT_TITLE = "New chat"
_TITLE_MAX = 60


def _get_owned_conversation(
    db: Session, conversation_id: uuid.UUID, user: User
) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


def _validate_owned_project(
    db: Session, project_id: uuid.UUID, user: User
) -> None:
    owned = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == user.id)
        .first()
    )
    if not owned:
        raise HTTPException(status_code=404, detail="Project not found.")


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    project_id: uuid.UUID | None = Query(default=None),
    unfiled: bool = Query(default=False),
    limit: int | None = Query(default=None, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List the user's conversations, most recently updated first.

    - ``project_id`` restricts to one project.
    - ``unfiled`` restricts to conversations not in any project.
    """
    q = db.query(Conversation).filter(Conversation.user_id == current_user.id)
    if project_id is not None:
        q = q.filter(Conversation.project_id == project_id)
    elif unfiled:
        q = q.filter(Conversation.project_id.is_(None))
    q = q.order_by(Conversation.updated_at.desc())
    if limit is not None:
        q = q.limit(limit)
    return q.all()


@router.post("", response_model=ConversationDetail, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new (empty) conversation, optionally filed under a project."""
    if payload.project_id is not None:
        _validate_owned_project(db, payload.project_id, current_user)
    conv = Conversation(
        user_id=current_user.id,
        project_id=payload.project_id,
        title=payload.title or _DEFAULT_TITLE,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch one conversation with its ordered messages."""
    return _get_owned_conversation(db, conversation_id, current_user)


@router.patch("/{conversation_id}", response_model=ConversationSummary)
def update_conversation(
    conversation_id: uuid.UUID,
    payload: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Rename a conversation and/or move it to (or out of) a project."""
    conv = _get_owned_conversation(db, conversation_id, current_user)
    if payload.title is not None:
        conv.title = payload.title
    if "project_id" in payload.model_fields_set:
        if payload.project_id is not None:
            _validate_owned_project(db, payload.project_id, current_user)
        conv.project_id = payload.project_id
    db.commit()
    db.refresh(conv)
    return conv


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a conversation and all of its messages."""
    conv = _get_owned_conversation(db, conversation_id, current_user)
    db.delete(conv)
    db.commit()


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
)
def append_message(
    conversation_id: uuid.UUID,
    payload: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Append a message to a conversation.

    Bumps ``updated_at`` so the thread floats to the top of recents, and
    auto-titles the conversation from the first user message.
    """
    conv = _get_owned_conversation(db, conversation_id, current_user)

    message = Message(
        conversation_id=conv.id,
        role=payload.role,
        content=payload.content,
        route=payload.route,
        sources=payload.sources,
        image_url=payload.image_url,
    )
    db.add(message)

    # Auto-title from the first user message while still on the default title.
    if payload.role == "user" and conv.title in ("", _DEFAULT_TITLE):
        snippet = payload.content.strip().replace("\n", " ")
        if snippet:
            conv.title = snippet[:_TITLE_MAX]

    conv.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(message)
    return message
