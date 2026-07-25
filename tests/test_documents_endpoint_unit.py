"""Direct unit tests for document endpoint functions, bypassing the ASGI transport."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai_research_assistant.api.v1.documents import (
    delete_document,
    search_documents,
    upload_document,
)
from ai_research_assistant.core.exceptions import UnsupportedFileTypeError
from ai_research_assistant.db.models.user import User
from ai_research_assistant.schemas.document import DocumentSearchRequest


def _user() -> User:
    return User(id=uuid4(), email="a@example.com", hashed_password="x")


async def test_upload_document_rejects_missing_content_type() -> None:
    upload_file = MagicMock()
    upload_file.content_type = None
    document_service = MagicMock()

    with pytest.raises(UnsupportedFileTypeError):
        await upload_document(upload_file, _user(), document_service)

    document_service.ingest_upload.assert_not_called()


async def test_upload_document_delegates_to_service() -> None:
    upload_file = MagicMock()
    upload_file.content_type = "text/plain"
    upload_file.filename = "notes.txt"
    upload_file.read = AsyncMock(return_value=b"hello")
    document_service = MagicMock()
    document_service.ingest_upload = AsyncMock(return_value="the-document")
    user = _user()

    result = await upload_document(upload_file, user, document_service)

    assert result == "the-document"  # type: ignore[comparison-overlap]
    document_service.ingest_upload.assert_awaited_once_with(
        owner_id=user.id, filename="notes.txt", content_type="text/plain", raw=b"hello"
    )


async def test_delete_document_delegates_to_service() -> None:
    document_service = MagicMock()
    document_service.delete_document = AsyncMock()
    user = _user()
    document_id = uuid4()

    await delete_document(document_id, user, document_service)

    document_service.delete_document.assert_awaited_once_with(
        document_id=document_id, owner_id=user.id
    )


async def test_search_documents_delegates_to_retriever() -> None:
    retriever = MagicMock()
    retriever.search = AsyncMock(return_value=[])
    user = _user()

    result = await search_documents(
        DocumentSearchRequest(query="find me", top_k=3), user, retriever
    )

    assert result == []
    retriever.search.assert_awaited_once_with(
        query="find me", owner_id=user.id, document_id=None, top_k=3
    )
