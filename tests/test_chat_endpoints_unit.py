"""Direct unit tests for chat endpoint functions, bypassing the ASGI transport."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from ai_research_assistant.api.v1.chat import (
    delete_chat_session,
    query_chat_session,
    stream_chat_session_query,
)
from ai_research_assistant.db.models.user import User
from ai_research_assistant.schemas.chat import ChatQueryRequest, ResponseMetadata, SourceCitation


def _user() -> User:
    return User(id=uuid4(), email="a@example.com", hashed_password="x")


async def test_query_chat_session_builds_response() -> None:
    session_id = uuid4()
    metadata = ResponseMetadata(
        model="test-model",
        latency_ms=12.5,
        retrieved_chunk_count=1,
        generated_at=datetime.now(UTC),
    )
    sources = [
        SourceCitation(
            document_id=uuid4(), document_title="Doc", chunk_index=0, score=0.9, excerpt="text"
        )
    ]
    rag_chat_service = MagicMock()
    rag_chat_service.query = AsyncMock(return_value=("The answer", sources, metadata))

    response = await query_chat_session(
        session_id, ChatQueryRequest(question="What?"), _user(), rag_chat_service
    )

    assert response.answer == "The answer"
    assert response.session_id == session_id
    assert response.sources == sources
    assert response.metadata == metadata


async def test_delete_chat_session_delegates() -> None:
    rag_chat_service = MagicMock()
    rag_chat_service.delete_session = AsyncMock()
    session_id = uuid4()
    user = _user()

    await delete_chat_session(session_id, user, rag_chat_service)

    rag_chat_service.delete_session.assert_awaited_once_with(
        session_id=session_id, owner_id=user.id
    )


async def test_stream_chat_session_query_emits_sse_events() -> None:
    session_id = uuid4()
    user = _user()

    async def fake_stream_query(**_kwargs: object) -> object:
        yield {"type": "token", "text": "Hello "}
        yield {"type": "done", "sources": [], "metadata": {"model": "test-model"}}

    rag_chat_service = MagicMock()
    rag_chat_service.ensure_session = AsyncMock()
    rag_chat_service.stream_query = MagicMock(side_effect=fake_stream_query)

    response = await stream_chat_session_query(
        session_id, ChatQueryRequest(question="Stream?"), user, rag_chat_service
    )

    chunks = [chunk async for chunk in response.body_iterator]
    rag_chat_service.ensure_session.assert_awaited_once_with(
        session_id=session_id, owner_id=user.id
    )

    events = []
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else bytes(chunk).decode()
        for block in text.split("\n\n"):
            if not block.strip():
                continue
            data_line = next(line for line in block.splitlines() if line.startswith("data: "))
            events.append(json.loads(data_line[len("data: ") :]))

    assert events[0] == {"type": "token", "text": "Hello "}
    assert events[1]["type"] == "done"
