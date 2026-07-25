"""Shared pytest fixtures: an isolated app, in-memory database, and fake AI backends."""

from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from ai_research_assistant.api import deps
from ai_research_assistant.api.router import api_router
from ai_research_assistant.core.exceptions import register_exception_handlers
from ai_research_assistant.db.base import Base
from ai_research_assistant.services.vectorstore.chroma_store import VectorSearchResult


class FakeEmbeddingService:
    """Deterministic embedding stand-in: encodes text length so search is testable."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [float(len(text) % 97), float(sum(map(ord, text)) % 97)]


class FakeVectorStore:
    """In-memory stand-in for ChromaVectorStore."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def add_chunks(
        self,
        *,
        vector_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for vector_id, embedding, document, metadata in zip(
            vector_ids, embeddings, documents, metadatas, strict=True
        ):
            self._items.append(
                {
                    "vector_id": vector_id,
                    "embedding": embedding,
                    "document": document,
                    "metadata": metadata,
                }
            )

    def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        owner_id: UUID,
        document_id: UUID | None = None,
    ) -> list[VectorSearchResult]:
        candidates = [item for item in self._items if item["metadata"]["owner_id"] == str(owner_id)]
        if document_id is not None:
            candidates = [
                item for item in candidates if item["metadata"]["document_id"] == str(document_id)
            ]
        candidates.sort(key=lambda item: sum(item["embedding"]))
        return [
            VectorSearchResult(
                vector_id=item["vector_id"],
                document=item["document"],
                metadata=item["metadata"],
                distance=0.1,
            )
            for item in candidates[:top_k]
        ]

    def delete_by_document(self, document_id: UUID) -> None:
        self._items = [
            item for item in self._items if item["metadata"]["document_id"] != str(document_id)
        ]


class FakeLLMClient:
    """Canned chat completion stand-in that never calls a real provider."""

    model_name = "fake-model"

    async def complete(self, *, system_prompt: str, messages: list[tuple[str, str]]) -> str:
        question = messages[-1][1] if messages else ""
        return f"Fake answer to: {question}"

    async def stream(
        self, *, system_prompt: str, messages: list[tuple[str, str]]
    ) -> AsyncGenerator[str]:
        answer = await self.complete(system_prompt=system_prompt, messages=messages)
        for word in answer.split(" "):
            yield word + " "


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def app(db_session: AsyncSession) -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(api_router, prefix="/api/v1")

    async def override_get_db_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    llm_client = FakeLLMClient()

    application.dependency_overrides[deps.get_db_session] = override_get_db_session
    application.dependency_overrides[deps.get_embedding_service] = lambda: embedding_service
    application.dependency_overrides[deps.get_vector_store] = lambda: vector_store
    application.dependency_overrides[deps.get_llm_client] = lambda: llm_client

    return application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "researcher@example.com", "password": "supersecret123"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "researcher@example.com", "password": "supersecret123"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
