"""Orchestrates extraction, chunking, embedding, and vector storage for a document."""

from uuid import UUID, uuid4

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.logging import get_logger
from ai_research_assistant.db.models.document import Document, DocumentChunk
from ai_research_assistant.services.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.services.ingestion.chunking import DocumentChunker
from ai_research_assistant.services.ingestion.loaders import ExtractedDocument, extract_document
from ai_research_assistant.services.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """Runs the full ingest flow: parse -> chunk -> embed -> persist."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        settings: Settings,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._chunker = DocumentChunker(settings)

    def extract(self, *, raw: bytes, content_type: str) -> ExtractedDocument:
        return extract_document(raw=raw, content_type=content_type)

    async def process(
        self, *, document: Document, extracted: ExtractedDocument
    ) -> list[DocumentChunk]:
        """Chunk the extracted text, embed each chunk, and store vectors + metadata."""
        text_chunks = self._chunker.split(extracted.text)
        if not text_chunks:
            return []

        embeddings = await self._embedding_service.embed_documents(
            [chunk.content for chunk in text_chunks]
        )

        vector_ids = [str(uuid4()) for _ in text_chunks]
        metadatas = [
            {
                "owner_id": str(document.owner_id),
                "document_id": str(document.id),
                "document_title": document.title or document.filename,
                "chunk_index": chunk.index,
            }
            for chunk in text_chunks
        ]

        self._vector_store.add_chunks(
            vector_ids=vector_ids,
            embeddings=embeddings,
            documents=[chunk.content for chunk in text_chunks],
            metadatas=metadatas,
        )

        return [
            DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.index,
                content=chunk.content,
                token_count=chunk.token_count,
                vector_id=vector_id,
            )
            for chunk, vector_id in zip(text_chunks, vector_ids, strict=True)
        ]

    def delete_document_vectors(self, document_id: UUID) -> None:
        self._vector_store.delete_by_document(document_id)
