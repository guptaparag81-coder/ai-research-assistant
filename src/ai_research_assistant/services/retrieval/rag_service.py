"""RAG orchestration: retrieval + context + prompt + memory + generation."""

import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from ai_research_assistant.core.exceptions import NotFoundError
from ai_research_assistant.db.models.chat import ChatMessage, ChatSession, MessageRole
from ai_research_assistant.repositories.chat_repository import ChatRepository
from ai_research_assistant.schemas.chat import ResponseMetadata, SourceCitation
from ai_research_assistant.services.llm.llm_client import LLMClient
from ai_research_assistant.services.memory.conversation_memory import ConversationMemoryService
from ai_research_assistant.services.retrieval.context_builder import ContextBuilder
from ai_research_assistant.services.retrieval.prompt_builder import PromptBuilder
from ai_research_assistant.services.retrieval.retriever import RetrievedChunk, Retriever


def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
    return [
        SourceCitation(
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
            excerpt=chunk.content[:280],
        )
        for chunk in chunks
    ]


class RAGChatService:
    """Coordinates the end-to-end retrieval-augmented generation query flow."""

    def __init__(
        self,
        *,
        chat_repository: ChatRepository,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        conversation_memory: ConversationMemoryService,
        llm_client: LLMClient,
    ) -> None:
        self._chat_repository = chat_repository
        self._retriever = retriever
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._conversation_memory = conversation_memory
        self._llm_client = llm_client

    async def create_session(
        self, *, owner_id: UUID, title: str = "New conversation"
    ) -> ChatSession:
        return await self._chat_repository.create_session(
            ChatSession(owner_id=owner_id, title=title)
        )

    async def list_sessions(self, *, owner_id: UUID) -> list[ChatSession]:
        return await self._chat_repository.list_sessions(owner_id)

    async def ensure_session(self, *, session_id: UUID, owner_id: UUID) -> ChatSession:
        session = await self._chat_repository.get_session(session_id, owner_id)
        if session is None:
            raise NotFoundError(f"Chat session {session_id} not found")
        return session

    async def list_messages(self, *, session_id: UUID, owner_id: UUID) -> list[ChatMessage]:
        await self.ensure_session(session_id=session_id, owner_id=owner_id)
        return await self._chat_repository.get_all_messages(session_id)

    async def delete_session(self, *, session_id: UUID, owner_id: UUID) -> None:
        session = await self.ensure_session(session_id=session_id, owner_id=owner_id)
        await self._chat_repository.delete_session(session)

    async def _prepare_turn(
        self, *, session_id: UUID, owner_id: UUID, question: str, top_k: int | None
    ) -> tuple[str, list[tuple[str, str]], list[SourceCitation]]:
        """Validate the session, retrieve context, and persist the user turn."""
        await self.ensure_session(session_id=session_id, owner_id=owner_id)

        retrieved_chunks = await self._retriever.search(
            query=question, owner_id=owner_id, top_k=top_k
        )
        context = self._context_builder.build(retrieved_chunks)
        system_prompt = self._prompt_builder.build_system_prompt(context)
        history = await self._conversation_memory.load_history(session_id)

        await self._conversation_memory.append_turn(
            session_id=session_id, role=MessageRole.USER, content=question
        )

        sources = _build_sources(retrieved_chunks)
        return system_prompt, [*history, ("user", question)], sources

    async def query(
        self, *, session_id: UUID, owner_id: UUID, question: str, top_k: int | None
    ) -> tuple[str, list[SourceCitation], ResponseMetadata]:
        system_prompt, messages, sources = await self._prepare_turn(
            session_id=session_id, owner_id=owner_id, question=question, top_k=top_k
        )

        started_at = time.monotonic()
        answer = await self._llm_client.complete(system_prompt=system_prompt, messages=messages)
        metadata = ResponseMetadata(
            model=self._llm_client.model_name,
            latency_ms=(time.monotonic() - started_at) * 1000,
            retrieved_chunk_count=len(sources),
            generated_at=datetime.now(UTC),
        )

        await self._conversation_memory.append_turn(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            citations=[source.model_dump(mode="json") for source in sources],
            response_metadata=metadata.model_dump(mode="json"),
        )

        return answer, sources, metadata

    async def stream_query(
        self, *, session_id: UUID, owner_id: UUID, question: str, top_k: int | None
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the answer token-by-token, yielding a final `done` event with sources/metadata."""
        system_prompt, messages, sources = await self._prepare_turn(
            session_id=session_id, owner_id=owner_id, question=question, top_k=top_k
        )

        started_at = time.monotonic()
        answer_parts: list[str] = []
        async for token in self._llm_client.stream(system_prompt=system_prompt, messages=messages):
            answer_parts.append(token)
            yield {"type": "token", "text": token}

        answer = "".join(answer_parts)
        metadata = ResponseMetadata(
            model=self._llm_client.model_name,
            latency_ms=(time.monotonic() - started_at) * 1000,
            retrieved_chunk_count=len(sources),
            generated_at=datetime.now(UTC),
        )

        await self._conversation_memory.append_turn(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            citations=[source.model_dump(mode="json") for source in sources],
            response_metadata=metadata.model_dump(mode="json"),
        )

        yield {
            "type": "done",
            "sources": [source.model_dump(mode="json") for source in sources],
            "metadata": metadata.model_dump(mode="json"),
        }
