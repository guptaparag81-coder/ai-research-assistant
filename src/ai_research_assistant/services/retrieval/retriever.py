"""Semantic / similarity search retrieval over the vector store."""

from dataclasses import dataclass
from uuid import UUID

from ai_research_assistant.core.config import Settings
from ai_research_assistant.services.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.services.vectorstore.chroma_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: UUID
    document_title: str | None
    chunk_index: int
    content: str
    score: float


class Retriever:
    """Performs semantic similarity search and returns ranked, scored chunks."""

    def __init__(
        self,
        *,
        vector_store: ChromaVectorStore,
        embedding_service: EmbeddingService,
        settings: Settings,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._settings = settings

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Convert a cosine distance in [0, 2] into a similarity score in [0, 1]."""
        return max(0.0, 1.0 - distance / 2.0)

    async def search(
        self,
        *,
        query: str,
        owner_id: UUID,
        document_id: UUID | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Run semantic similarity search for `query`, scoped to `owner_id`."""
        effective_top_k = top_k or self._settings.retrieval_top_k
        query_embedding = await self._embedding_service.embed_query(query)

        raw_results = self._vector_store.query(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            owner_id=owner_id,
            document_id=document_id,
        )

        chunks: list[RetrievedChunk] = []
        for result in raw_results:
            score = self._distance_to_similarity(result.distance)
            if score < self._settings.retrieval_score_threshold:
                continue
            chunks.append(
                RetrievedChunk(
                    document_id=UUID(str(result.metadata["document_id"])),
                    document_title=result.metadata.get("document_title"),
                    chunk_index=int(result.metadata["chunk_index"]),
                    content=result.document,
                    score=score,
                )
            )
        return chunks
