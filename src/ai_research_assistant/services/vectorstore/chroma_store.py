"""ChromaDB-backed vector store for document chunk embeddings."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

from ai_research_assistant.core.config import Settings


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    vector_id: str
    document: str
    metadata: dict[str, Any]
    distance: float


def create_chroma_client(settings: Settings) -> ClientAPI:
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        ssl=settings.chroma_ssl,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


class ChromaVectorStore:
    """Encapsulates all ChromaDB collection access for document embeddings."""

    def __init__(self, client: ClientAPI, settings: Settings) -> None:
        self._client = client
        self._collection_name = settings.chroma_collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        *,
        vector_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._collection.add(
            ids=vector_ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        owner_id: UUID,
        document_id: UUID | None = None,
    ) -> list[VectorSearchResult]:
        where: dict[str, Any] = {"owner_id": str(owner_id)}
        if document_id is not None:
            where = {"$and": [where, {"document_id": str(document_id)}]}

        result = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            VectorSearchResult(
                vector_id=vector_id,
                document=document or "",
                metadata=dict(metadata or {}),
                distance=distance,
            )
            for vector_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def delete_by_document(self, document_id: UUID) -> None:
        self._collection.delete(where={"document_id": str(document_id)})
