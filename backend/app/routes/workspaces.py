"""Workspace routes — CRUD and member management."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember
from app.schemas.workspace import (
    WorkspaceOut,
    WorkspaceUpdate,
    WorkspacePermissionsUpdate,
    MemberOut,
    MemberInvite,
    MemberRoleUpdate,
)

router = APIRouter(prefix="/api/workspaces", tags=["Workspaces"])


# ── Workspace CRUD ────────────────────────────────────
@router.get("", response_model=list[WorkspaceOut])
def list_workspaces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all workspaces the user is a member of."""
    return (
        db.query(Workspace)
        .join(WorkspaceMember)
        .filter(WorkspaceMember.user_id == current_user.id)
        .all()
    )


@router.get("/{workspace_id}", response_model=WorkspaceOut)
def get_workspace(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get workspace details."""
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return ws


@router.put("/{workspace_id}", response_model=WorkspaceOut)
def update_workspace(
    workspace_id: uuid.UUID,
    payload: WorkspaceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update workspace name."""
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the owner can update the workspace.")
    if payload.name is not None:
        ws.name = payload.name
    db.commit()
    db.refresh(ws)
    return ws


@router.put("/{workspace_id}/permissions", response_model=WorkspaceOut)
def update_permissions(
    workspace_id: uuid.UUID,
    payload: WorkspacePermissionsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update the default role for new members."""
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the owner can change permissions.")
    ws.default_role = payload.default_role
    db.commit()
    db.refresh(ws)
    return ws


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a workspace permanently (danger zone)."""
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the owner can delete the workspace.")
    db.delete(ws)
    db.commit()


# ── Members ───────────────────────────────────────────
@router.get("/{workspace_id}/members", response_model=list[MemberOut])
def list_members(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all members of a workspace."""
    members = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id)
        .all()
    )
    result = []
    for m in members:
        user = db.query(User).filter(User.id == m.user_id).first()
        if user:
            result.append(
                MemberOut(
                    id=m.id,
                    user_id=user.id,
                    display_name=user.display_name,
                    email=user.email,
                    avatar_initials=user.avatar_initials,
                    avatar_gradient=user.avatar_gradient,
                    role=m.role,
                    joined_at=m.joined_at,
                )
            )
    return result


@router.post("/{workspace_id}/members", response_model=MemberOut, status_code=status.HTTP_201_CREATED)
def invite_member(
    workspace_id: uuid.UUID,
    payload: MemberInvite,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Invite a user to the workspace by email."""
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    # Check requester is owner or admin
    requester_member = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == current_user.id)
        .first()
    )
    if not requester_member or requester_member.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can invite members.")

    # Find user by email
    invitee = db.query(User).filter(User.email == payload.email.lower()).first()
    if not invitee:
        raise HTTPException(status_code=404, detail="No user found with that email.")

    # Check not already a member
    existing = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == invitee.id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="User is already a member.")

    member = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=invitee.id,
        role=payload.role,
    )
    db.add(member)
    db.commit()
    db.refresh(member)

    return MemberOut(
        id=member.id,
        user_id=invitee.id,
        display_name=invitee.display_name,
        email=invitee.email,
        avatar_initials=invitee.avatar_initials,
        avatar_gradient=invitee.avatar_gradient,
        role=member.role,
        joined_at=member.joined_at,
    )


@router.put("/{workspace_id}/members/{user_id}", response_model=MemberOut)
def update_member_role(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: MemberRoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a member's role in the workspace."""
    member = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id)
        .first()
    )
    if not member:
        raise HTTPException(status_code=404, detail="Member not found.")

    member.role = payload.role
    db.commit()
    db.refresh(member)

    user = db.query(User).filter(User.id == user_id).first()
    return MemberOut(
        id=member.id,
        user_id=user.id,
        display_name=user.display_name,
        email=user.email,
        avatar_initials=user.avatar_initials,
        avatar_gradient=user.avatar_gradient,
        role=member.role,
        joined_at=member.joined_at,
    )


@router.delete("/{workspace_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Remove a member from the workspace."""
    member = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id)
        .first()
    )
    if not member:
        raise HTTPException(status_code=404, detail="Member not found.")
    if member.role == "owner":
        raise HTTPException(status_code=400, detail="Cannot remove the workspace owner.")
    db.delete(member)
    db.commit()
