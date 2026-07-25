"""Add citations and response_metadata columns to chat_messages

Revision ID: 0002_add_message_metadata
Revises: 0001_initial_schema
Create Date: 2026-07-25

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_add_message_metadata"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("citations", sa.JSON(), nullable=True))
    op.add_column("chat_messages", sa.Column("response_metadata", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "response_metadata")
    op.drop_column("chat_messages", "citations")
