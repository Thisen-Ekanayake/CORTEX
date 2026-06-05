"""Project routes — CRUD for chat folders, scoped to the current user."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.conversation import Project
from app.schemas.conversation import ProjectCreate, ProjectUpdate, ProjectOut

router = APIRouter(prefix="/api/projects", tags=["Projects"])


def _get_owned_project(db: Session, project_id: uuid.UUID, user: User) -> Project:
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List the current user's projects, most recently updated first."""
    return (
        db.query(Project)
        .filter(Project.user_id == current_user.id)
        .order_by(Project.updated_at.desc())
        .all()
    )


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new project folder."""
    project = Project(user_id=current_user.id, name=payload.name)
    if payload.icon:
        project.icon = payload.icon
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: uuid.UUID,
    payload: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Rename a project or change its icon."""
    project = _get_owned_project(db, project_id, current_user)
    if payload.name is not None:
        project.name = payload.name
    if payload.icon is not None:
        project.icon = payload.icon
    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a project. Its conversations are unfiled (project_id → NULL)."""
    project = _get_owned_project(db, project_id, current_user)
    db.delete(project)
    db.commit()
