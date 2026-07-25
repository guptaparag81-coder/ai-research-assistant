"""Embedding generation via an OpenAI-compatible embeddings endpoint."""

from langchain_openai import OpenAIEmbeddings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import LLMServiceError
from ai_research_assistant.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Generates vector embeddings for document chunks and queries."""

    def __init__(self, settings: Settings) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.llm_api_key.get_secret_value(),
            openai_api_base=settings.llm_base_url,
            dimensions=settings.embedding_dimensions,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._embeddings.aembed_documents(texts)
        except Exception as exc:
            logger.error("embedding_generation_failed", count=len(texts), error=str(exc))
            raise LLMServiceError("Failed to generate embeddings") from exc

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def embed_query(self, text: str) -> list[float]:
        try:
            return await self._embeddings.aembed_query(text)
        except Exception as exc:
            logger.error("query_embedding_failed", error=str(exc))
            raise LLMServiceError("Failed to generate query embedding") from exc
