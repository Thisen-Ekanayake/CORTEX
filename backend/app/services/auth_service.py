"""Authentication service — password hashing, JWT creation."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.user import User, UserPreference
from app.models.notification import NotificationPreference
from app.models.workspace import Workspace, WorkspaceMember

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token for the given user ID."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def register_user(db: Session, email: str, password: str, display_name: str) -> User:
    """Create a new user with default preferences and a dedicated workspace."""
    # Generate initials from display name
    parts = display_name.strip().split()
    if parts:
        initials = "".join([p[0].upper() for p in parts[:2]])
    else:
        initials = "U"

    user = User(
        email=email.lower(),
        display_name=display_name,
        avatar_initials=initials,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.flush()  # Get user.id

    # Create default preferences
    prefs = UserPreference(user_id=user.id)
    db.add(prefs)

    # Create default notification preferences
    notif = NotificationPreference(user_id=user.id)
    db.add(notif)

    # Create default workspace
    ws_name = f"{display_name}'s Workspace"
    ws = Workspace(
        name=ws_name,
        org_id=f"org_{uuid.uuid4().hex[:8]}",
        owner_id=user.id
    )
    db.add(ws)
    db.flush()

    # Add user as owner of the workspace
    member = WorkspaceMember(
        workspace_id=ws.id,
        user_id=user.id,
        role="owner"
    )
    db.add(member)

    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password. Returns None on failure."""
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
