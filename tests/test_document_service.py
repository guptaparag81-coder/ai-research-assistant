"""Unit tests for DocumentService, with the repository and pipeline mocked."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import FileTooLargeError, NotFoundError
from ai_research_assistant.db.models.document import Document, DocumentStatus, DocumentType
from ai_research_assistant.services.document_service import DocumentService
from ai_research_assistant.services.ingestion.loaders import ExtractedDocument


@pytest.fixture
def document_repository() -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock(side_effect=lambda document: document)
    repo.add_chunks = AsyncMock()
    repo.save = AsyncMock(side_effect=lambda document: document)
    repo.commit = AsyncMock()
    repo.delete = AsyncMock()
    return repo


@pytest.fixture
def ingestion_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.extract = MagicMock(
        return_value=ExtractedDocument(
            text="hello", title="Extracted Title", page_count=None, document_type=DocumentType.TXT
        )
    )
    pipeline.process = AsyncMock(return_value=[])
    return pipeline


@pytest.fixture
def service(document_repository: MagicMock, ingestion_pipeline: MagicMock) -> DocumentService:
    return DocumentService(
        document_repository=document_repository,
        ingestion_pipeline=ingestion_pipeline,
        settings=Settings(max_upload_size_bytes=1000),
    )


async def test_ingest_upload_rejects_oversized_files(service: DocumentService) -> None:
    with pytest.raises(FileTooLargeError):
        await service.ingest_upload(
            owner_id=uuid4(), filename="big.txt", content_type="text/plain", raw=b"x" * 2000
        )


async def test_ingest_upload_succeeds_and_marks_ready(
    service: DocumentService, document_repository: MagicMock, ingestion_pipeline: MagicMock
) -> None:
    document = await service.ingest_upload(
        owner_id=uuid4(), filename="notes.txt", content_type="text/plain", raw=b"hello"
    )

    assert document.status == DocumentStatus.READY
    assert document.title == "Extracted Title"
    document_repository.create.assert_awaited_once()
    document_repository.add_chunks.assert_awaited_once()
    document_repository.save.assert_awaited_once()
    document_repository.commit.assert_not_awaited()


async def test_ingest_upload_marks_failed_on_processing_error(
    service: DocumentService, document_repository: MagicMock, ingestion_pipeline: MagicMock
) -> None:
    ingestion_pipeline.process = AsyncMock(side_effect=RuntimeError("chunking exploded"))

    with pytest.raises(RuntimeError):
        await service.ingest_upload(
            owner_id=uuid4(), filename="notes.txt", content_type="text/plain", raw=b"hello"
        )

    saved_document = document_repository.save.await_args.args[0]
    assert saved_document.status == DocumentStatus.FAILED
    assert saved_document.error_message == "chunking exploded"
    document_repository.commit.assert_awaited_once()


async def test_get_document_raises_not_found(
    service: DocumentService, document_repository: MagicMock
) -> None:
    document_repository.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(NotFoundError):
        await service.get_document(document_id=uuid4(), owner_id=uuid4())


async def test_get_document_returns_document(
    service: DocumentService, document_repository: MagicMock
) -> None:
    owner_id = uuid4()
    document = Document(
        id=uuid4(),
        owner_id=owner_id,
        filename="a.txt",
        document_type=DocumentType.TXT,
        content_type="text/plain",
        file_size_bytes=1,
        status=DocumentStatus.READY,
    )
    document_repository.get_by_id = AsyncMock(return_value=document)

    result = await service.get_document(document_id=document.id, owner_id=owner_id)

    assert result is document


async def test_list_documents_delegates_to_repository(
    service: DocumentService, document_repository: MagicMock
) -> None:
    document_repository.list_by_owner = AsyncMock(return_value=[])
    owner_id = uuid4()

    result = await service.list_documents(owner_id=owner_id)

    assert result == []
    document_repository.list_by_owner.assert_awaited_once_with(owner_id)


async def test_delete_document_removes_vectors_and_row(
    service: DocumentService, document_repository: MagicMock, ingestion_pipeline: MagicMock
) -> None:
    owner_id = uuid4()
    document = Document(
        id=uuid4(),
        owner_id=owner_id,
        filename="a.txt",
        document_type=DocumentType.TXT,
        content_type="text/plain",
        file_size_bytes=1,
        status=DocumentStatus.READY,
    )
    document_repository.get_by_id = AsyncMock(return_value=document)

    await service.delete_document(document_id=document.id, owner_id=owner_id)

    ingestion_pipeline.delete_document_vectors.assert_called_once_with(document.id)
    document_repository.delete.assert_awaited_once_with(document)
