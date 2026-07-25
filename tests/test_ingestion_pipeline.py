"""Unit tests for the ingestion pipeline, with embeddings/vector store mocked."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai_research_assistant.core.config import Settings
from ai_research_assistant.db.models.document import Document, DocumentStatus, DocumentType
from ai_research_assistant.services.ingestion.loaders import ExtractedDocument
from ai_research_assistant.services.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def settings() -> Settings:
    return Settings(chunk_size=20, chunk_overlap=5)


@pytest.fixture
def embedding_service() -> MagicMock:
    service = MagicMock()
    service.embed_documents = AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts])
    return service


@pytest.fixture
def vector_store() -> MagicMock:
    return MagicMock()


@pytest.fixture
def pipeline(
    embedding_service: MagicMock, vector_store: MagicMock, settings: Settings
) -> IngestionPipeline:
    return IngestionPipeline(
        embedding_service=embedding_service, vector_store=vector_store, settings=settings
    )


def _document() -> Document:
    return Document(
        id=uuid4(),
        owner_id=uuid4(),
        filename="notes.txt",
        title=None,
        document_type=DocumentType.TXT,
        content_type="text/plain",
        file_size_bytes=100,
        status=DocumentStatus.PROCESSING,
    )


def test_extract_delegates_to_loaders(pipeline: IngestionPipeline) -> None:
    extracted = pipeline.extract(raw=b"hello world", content_type="text/plain")
    assert extracted.text == "hello world"


async def test_process_chunks_embeds_and_stores(
    pipeline: IngestionPipeline, embedding_service: MagicMock, vector_store: MagicMock
) -> None:
    document = _document()
    extracted = ExtractedDocument(
        text="This is a somewhat long piece of text used to validate chunk processing.",
        title=None,
        page_count=None,
        document_type=DocumentType.TXT,
    )

    chunks = await pipeline.process(document=document, extracted=extracted)

    assert len(chunks) > 0
    assert all(chunk.document_id == document.id for chunk in chunks)
    embedding_service.embed_documents.assert_awaited_once()
    vector_store.add_chunks.assert_called_once()

    call_kwargs = vector_store.add_chunks.call_args.kwargs
    assert len(call_kwargs["vector_ids"]) == len(chunks)
    assert all(meta["owner_id"] == str(document.owner_id) for meta in call_kwargs["metadatas"])
    assert all(meta["document_id"] == str(document.id) for meta in call_kwargs["metadatas"])


async def test_process_uses_filename_when_title_missing(
    pipeline: IngestionPipeline, vector_store: MagicMock
) -> None:
    document = _document()
    extracted = ExtractedDocument(
        text="Short text that still produces one chunk.",
        title=None,
        page_count=None,
        document_type=DocumentType.TXT,
    )

    await pipeline.process(document=document, extracted=extracted)

    metadatas = vector_store.add_chunks.call_args.kwargs["metadatas"]
    assert metadatas[0]["document_title"] == "notes.txt"


async def test_process_returns_empty_list_for_empty_chunks(
    pipeline: IngestionPipeline, embedding_service: MagicMock, vector_store: MagicMock
) -> None:
    document = _document()
    extracted = ExtractedDocument(
        text="", title=None, page_count=None, document_type=DocumentType.TXT
    )

    chunks = await pipeline.process(document=document, extracted=extracted)

    assert chunks == []
    embedding_service.embed_documents.assert_not_awaited()
    vector_store.add_chunks.assert_not_called()


def test_delete_document_vectors_delegates_to_vector_store(
    pipeline: IngestionPipeline, vector_store: MagicMock
) -> None:
    document_id = uuid4()
    pipeline.delete_document_vectors(document_id)
    vector_store.delete_by_document.assert_called_once_with(document_id)
