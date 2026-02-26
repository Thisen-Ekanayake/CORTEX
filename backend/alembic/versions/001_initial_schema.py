"""Initial database schema — all 7 tables.

Revision ID: 001
Revises: None
Create Date: 2026-02-26
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Users ─────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("avatar_initials", sa.String(4), server_default=""),
        sa.Column("avatar_gradient", sa.String(100), server_default="from-accent-primary to-purple-600"),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("two_factor_enabled", sa.Boolean, server_default="false"),
        sa.Column("two_factor_secret", sa.String(255), nullable=True),
        sa.Column("session_timeout_enabled", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── User Preferences ─────────────────────────
    op.create_table(
        "user_preferences",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), unique=True),
        sa.Column("dark_mode", sa.Boolean, server_default="true"),
        sa.Column("high_contrast", sa.Boolean, server_default="false"),
        sa.Column("animations_enabled", sa.Boolean, server_default="true"),
        sa.Column("compact_sidebar", sa.Boolean, server_default="false"),
        sa.Column("email_alerts", sa.Boolean, server_default="true"),
    )

    # ── Workspaces ────────────────────────────────
    op.create_table(
        "workspaces",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("org_id", sa.String(50), unique=True, nullable=False),
        sa.Column("default_role", sa.String(20), server_default="member"),
        sa.Column("owner_id", UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Workspace Members ─────────────────────────
    op.create_table(
        "workspace_members",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="CASCADE")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("role", sa.String(20), server_default="member"),
        sa.Column("joined_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_user"),
    )

    # ── Sessions ──────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("device", sa.String(200), server_default="Unknown Device"),
        sa.Column("location", sa.String(200), server_default="Unknown"),
        sa.Column("ip_address", sa.String(45), server_default="0.0.0.0"),
        sa.Column("is_current", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_active", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Notification Preferences ──────────────────
    op.create_table(
        "notification_preferences",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), unique=True),
        sa.Column("email_weekly_digest", sa.Boolean, server_default="true"),
        sa.Column("email_product_updates", sa.Boolean, server_default="true"),
        sa.Column("push_mentions", sa.Boolean, server_default="true"),
        sa.Column("push_system_updates", sa.Boolean, server_default="false"),
        sa.Column("slack_enabled", sa.Boolean, server_default="false"),
        sa.Column("slack_channel", sa.String(100), server_default="#general"),
    )

    # ── API Keys ──────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="CASCADE")),
        sa.Column("key_prefix", sa.String(20), nullable=False),
        sa.Column("key_hash", sa.Text, nullable=False),
        sa.Column("label", sa.String(100), server_default="Production Key"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("api_keys")
    op.drop_table("notification_preferences")
    op.drop_table("sessions")
    op.drop_table("workspace_members")
    op.drop_table("workspaces")
    op.drop_table("user_preferences")
    op.drop_table("users")
