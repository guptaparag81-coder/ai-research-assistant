"""Regression tests for native-enum column value mapping.

SQLAlchemy's `Enum(PythonEnum, ...)` sends the enum member's `.name` (not `.value`)
to a native PostgreSQL enum type by default. Our Alembic migration creates the
Postgres enum types using the lowercase `.value` labels, so without
`values_callable` these columns would raise `InvalidTextRepresentationError`
against real PostgreSQL while appearing to work fine against SQLite (which does
not enforce native enum labels). These tests assert the SQLAlchemy column type's
`.enums` match the lowercase values actually created in the migration, regardless
of which database dialect the test suite runs against.
"""

from sqlalchemy import Enum as SqlEnum

from ai_research_assistant.db.models.chat import ChatMessage, MessageRole
from ai_research_assistant.db.models.document import Document, DocumentStatus, DocumentType


def test_document_type_column_uses_lowercase_values() -> None:
    column_type = Document.__table__.c.document_type.type
    assert isinstance(column_type, SqlEnum)
    assert column_type.enums == [member.value for member in DocumentType]


def test_document_status_column_uses_lowercase_values() -> None:
    column_type = Document.__table__.c.status.type
    assert isinstance(column_type, SqlEnum)
    assert column_type.enums == [member.value for member in DocumentStatus]


def test_message_role_column_uses_lowercase_values() -> None:
    column_type = ChatMessage.__table__.c.role.type
    assert isinstance(column_type, SqlEnum)
    assert column_type.enums == [member.value for member in MessageRole]
