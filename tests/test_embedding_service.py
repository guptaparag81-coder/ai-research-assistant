"""Unit tests for EmbeddingService, mocking the underlying OpenAIEmbeddings client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import LLMServiceError
from ai_research_assistant.services.embeddings.embedding_service import EmbeddingService


@pytest.fixture
def service() -> EmbeddingService:
    # `_embeddings` is a plain (non-pydantic) attribute on our wrapper, so it can be
    # swapped wholesale for a mock; the real OpenAIEmbeddings pydantic model rejects
    # attribute assignment for fields it doesn't declare.
    instance = EmbeddingService(Settings())
    instance._embeddings = MagicMock()
    return instance


async def test_embed_documents_returns_vectors(service: EmbeddingService) -> None:
    service._embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])  # type: ignore[method-assign]

    result = await service.embed_documents(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_documents_wraps_provider_errors(service: EmbeddingService) -> None:
    service._embeddings.aembed_documents = AsyncMock(side_effect=RuntimeError("provider down"))  # type: ignore[method-assign]

    with pytest.raises(LLMServiceError):
        await service.embed_documents(["a"])


async def test_embed_query_returns_vector(service: EmbeddingService) -> None:
    service._embeddings.aembed_query = AsyncMock(return_value=[0.5, 0.6])  # type: ignore[method-assign]

    result = await service.embed_query("hello")

    assert result == [0.5, 0.6]


async def test_embed_query_wraps_provider_errors(service: EmbeddingService) -> None:
    service._embeddings.aembed_query = AsyncMock(side_effect=RuntimeError("provider down"))  # type: ignore[method-assign]

    with pytest.raises(LLMServiceError):
        await service.embed_query("hello")
