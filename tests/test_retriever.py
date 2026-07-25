"""Unit tests for the Retriever class."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from ai_research_assistant.core.config import Settings
from ai_research_assistant.services.retrieval.retriever import Retriever
from ai_research_assistant.services.vectorstore.chroma_store import VectorSearchResult


def _result(*, distance: float, document_id: str, chunk_index: int = 0) -> VectorSearchResult:
    return VectorSearchResult(
        vector_id="v1",
        document="chunk text",
        metadata={
            "document_id": document_id,
            "document_title": "Some Doc",
            "chunk_index": chunk_index,
        },
        distance=distance,
    )


async def test_search_converts_distance_to_similarity_and_uses_settings_top_k() -> None:
    settings = Settings(retrieval_top_k=7, retrieval_score_threshold=0.0)
    embedding_service = MagicMock()
    embedding_service.embed_query = AsyncMock(return_value=[0.1, 0.2])
    vector_store = MagicMock()
    document_id = str(uuid4())
    vector_store.query = MagicMock(return_value=[_result(distance=0.4, document_id=document_id)])

    retriever = Retriever(
        vector_store=vector_store, embedding_service=embedding_service, settings=settings
    )
    results = await retriever.search(query="hello", owner_id=uuid4())

    vector_store.query.assert_called_once()
    assert vector_store.query.call_args.kwargs["top_k"] == 7
    assert len(results) == 1
    assert results[0].score == 0.8
    assert results[0].document_title == "Some Doc"


async def test_search_uses_explicit_top_k_over_settings_default() -> None:
    settings = Settings(retrieval_top_k=7)
    embedding_service = MagicMock()
    embedding_service.embed_query = AsyncMock(return_value=[0.1])
    vector_store = MagicMock()
    vector_store.query = MagicMock(return_value=[])

    retriever = Retriever(
        vector_store=vector_store, embedding_service=embedding_service, settings=settings
    )
    await retriever.search(query="hello", owner_id=uuid4(), top_k=2)

    assert vector_store.query.call_args.kwargs["top_k"] == 2


async def test_search_filters_out_results_below_score_threshold() -> None:
    settings = Settings(retrieval_score_threshold=0.9)
    embedding_service = MagicMock()
    embedding_service.embed_query = AsyncMock(return_value=[0.1])
    vector_store = MagicMock()
    # distance 0.4 -> score 0.8, below the 0.9 threshold -> filtered out
    vector_store.query = MagicMock(return_value=[_result(distance=0.4, document_id=str(uuid4()))])

    retriever = Retriever(
        vector_store=vector_store, embedding_service=embedding_service, settings=settings
    )
    results = await retriever.search(query="hello", owner_id=uuid4())

    assert results == []


async def test_search_passes_document_id_scope_through() -> None:
    settings = Settings()
    embedding_service = MagicMock()
    embedding_service.embed_query = AsyncMock(return_value=[0.1])
    vector_store = MagicMock()
    vector_store.query = MagicMock(return_value=[])
    document_id = uuid4()

    retriever = Retriever(
        vector_store=vector_store, embedding_service=embedding_service, settings=settings
    )
    await retriever.search(query="hello", owner_id=uuid4(), document_id=document_id)

    assert vector_store.query.call_args.kwargs["document_id"] == document_id
