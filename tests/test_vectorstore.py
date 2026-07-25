"""Unit tests for the ChromaDB vector store wrapper."""

from typing import Any
from uuid import uuid4

import pytest

from ai_research_assistant.core.config import Settings
from ai_research_assistant.services.vectorstore.chroma_store import (
    ChromaVectorStore,
    create_chroma_client,
)


class FakeCollection:
    def __init__(self) -> None:
        self.added: dict[str, Any] = {}
        self.queried_with: dict[str, Any] = {}
        self.deleted_where: dict[str, Any] | None = None
        self._next_query_result: dict[str, Any] = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def add(self, *, ids: list[str], embeddings: Any, documents: list[str], metadatas: Any) -> None:
        self.added = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas,
        }

    def query(
        self, *, query_embeddings: Any, n_results: int, where: dict[str, Any], include: list[str]
    ) -> dict[str, Any]:
        self.queried_with = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "where": where,
            "include": include,
        }
        return self._next_query_result

    def delete(self, *, where: dict[str, Any]) -> None:
        self.deleted_where = where

    def set_query_result(self, **kwargs: Any) -> None:
        self._next_query_result = kwargs


class FakeChromaClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.requested_name: str | None = None

    def get_or_create_collection(self, *, name: str, metadata: dict[str, Any]) -> FakeCollection:
        self.requested_name = name
        return self.collection


@pytest.fixture
def settings() -> Settings:
    return Settings(chroma_collection="test-collection")


def test_init_creates_collection_with_cosine_space(settings: Settings) -> None:
    client = FakeChromaClient()
    ChromaVectorStore(client, settings)  # type: ignore[arg-type]
    assert client.requested_name == "test-collection"


def test_add_chunks_forwards_all_fields(settings: Settings) -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(client, settings)  # type: ignore[arg-type]

    store.add_chunks(
        vector_ids=["v1"],
        embeddings=[[0.1, 0.2]],
        documents=["hello"],
        metadatas=[{"owner_id": "abc"}],
    )

    assert client.collection.added["ids"] == ["v1"]
    assert client.collection.added["embeddings"] == [[0.1, 0.2]]
    assert client.collection.added["documents"] == ["hello"]
    assert client.collection.added["metadatas"] == [{"owner_id": "abc"}]


def test_query_scopes_by_owner_id(settings: Settings) -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(client, settings)  # type: ignore[arg-type]
    owner_id = uuid4()
    document_id = uuid4()
    client.collection.set_query_result(
        ids=[["v1"]],
        documents=[["chunk text"]],
        metadatas=[[{"document_id": str(document_id), "chunk_index": 0}]],
        distances=[[0.4]],
    )

    results = store.query(query_embedding=[0.1, 0.2], top_k=5, owner_id=owner_id)

    assert client.collection.queried_with["where"] == {"owner_id": str(owner_id)}
    assert client.collection.queried_with["n_results"] == 5
    assert len(results) == 1
    assert results[0].vector_id == "v1"
    assert results[0].document == "chunk text"
    assert results[0].distance == 0.4


def test_query_scopes_by_owner_and_document_id(settings: Settings) -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(client, settings)  # type: ignore[arg-type]
    owner_id = uuid4()
    document_id = uuid4()

    store.query(query_embedding=[0.1], top_k=3, owner_id=owner_id, document_id=document_id)

    assert client.collection.queried_with["where"] == {
        "$and": [{"owner_id": str(owner_id)}, {"document_id": str(document_id)}]
    }


def test_query_handles_empty_results(settings: Settings) -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(client, settings)  # type: ignore[arg-type]

    results = store.query(query_embedding=[0.1], top_k=5, owner_id=uuid4())

    assert results == []


def test_query_handles_missing_metadata_and_document(settings: Settings) -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(client, settings)  # type: ignore[arg-type]
    client.collection.set_query_result(
        ids=[["v1"]],
        documents=[[None]],
        metadatas=[[None]],
        distances=[[0.9]],
    )

    results = store.query(query_embedding=[0.1], top_k=1, owner_id=uuid4())

    assert results[0].document == ""
    assert results[0].metadata == {}


def test_delete_by_document(settings: Settings) -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(client, settings)  # type: ignore[arg-type]
    document_id = uuid4()

    store.delete_by_document(document_id)

    assert client.collection.deleted_where == {"document_id": str(document_id)}


def test_create_chroma_client_constructs_http_client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_http_client(*, host: str, port: int, ssl: bool, settings: Any) -> str:
        captured.update(host=host, port=port, ssl=ssl)
        return "fake-client"

    monkeypatch.setattr(
        "ai_research_assistant.services.vectorstore.chroma_store.chromadb.HttpClient",
        fake_http_client,
    )

    client = create_chroma_client(settings)

    assert client == "fake-client"  # type: ignore[comparison-overlap]
    assert captured == {"host": settings.chroma_host, "port": settings.chroma_port, "ssl": False}
