"""Unit tests for RAGChatService orchestration, with all collaborators mocked."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai_research_assistant.core.exceptions import NotFoundError
from ai_research_assistant.db.models.chat import ChatSession, MessageRole
from ai_research_assistant.services.retrieval.rag_service import RAGChatService
from ai_research_assistant.services.retrieval.retriever import RetrievedChunk


def _make_chunk(**overrides: object) -> RetrievedChunk:
    defaults: dict[str, object] = {
        "document_id": uuid4(),
        "document_title": "Doc Title",
        "chunk_index": 0,
        "content": "Some retrieved content.",
        "score": 0.9,
    }
    defaults.update(overrides)
    return RetrievedChunk(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def collaborators() -> dict[str, MagicMock]:
    chat_repository = MagicMock()
    chat_repository.get_session = AsyncMock()
    chat_repository.create_session = AsyncMock(side_effect=lambda session: session)
    chat_repository.list_sessions = AsyncMock(return_value=[])
    chat_repository.get_all_messages = AsyncMock(return_value=[])
    chat_repository.delete_session = AsyncMock()

    retriever = MagicMock()
    retriever.search = AsyncMock(return_value=[_make_chunk()])

    context_builder = MagicMock()
    context_builder.build = MagicMock(return_value="built context")

    prompt_builder = MagicMock()
    prompt_builder.build_system_prompt = MagicMock(return_value="system prompt")

    conversation_memory = MagicMock()
    conversation_memory.load_history = AsyncMock(return_value=[("user", "previous question")])
    conversation_memory.append_turn = AsyncMock()

    llm_client = MagicMock()
    llm_client.model_name = "mock-model"
    llm_client.complete = AsyncMock(return_value="Mocked LLM answer.")

    async def fake_stream(*, system_prompt: str, messages: list[tuple[str, str]]) -> object:
        for token in ["Mocked ", "streamed ", "answer."]:
            yield token

    llm_client.stream = MagicMock(side_effect=fake_stream)

    return {
        "chat_repository": chat_repository,
        "retriever": retriever,
        "context_builder": context_builder,
        "prompt_builder": prompt_builder,
        "conversation_memory": conversation_memory,
        "llm_client": llm_client,
    }


@pytest.fixture
def service(collaborators: dict[str, MagicMock]) -> RAGChatService:
    return RAGChatService(**collaborators)


async def test_create_session_delegates_to_repository(
    service: RAGChatService, collaborators: dict[str, MagicMock]
) -> None:
    owner_id = uuid4()
    session = await service.create_session(owner_id=owner_id, title="Custom title")

    assert session.owner_id == owner_id
    assert session.title == "Custom title"
    collaborators["chat_repository"].create_session.assert_awaited_once()


async def test_ensure_session_raises_not_found_when_missing(
    service: RAGChatService, collaborators: dict[str, MagicMock]
) -> None:
    collaborators["chat_repository"].get_session.return_value = None

    with pytest.raises(NotFoundError):
        await service.ensure_session(session_id=uuid4(), owner_id=uuid4())


async def test_ensure_session_returns_session_when_found(
    service: RAGChatService, collaborators: dict[str, MagicMock]
) -> None:
    owner_id = uuid4()
    session_id = uuid4()
    expected = ChatSession(id=session_id, owner_id=owner_id, title="t")
    collaborators["chat_repository"].get_session.return_value = expected

    result = await service.ensure_session(session_id=session_id, owner_id=owner_id)

    assert result is expected


async def test_list_messages_requires_existing_session(
    service: RAGChatService, collaborators: dict[str, MagicMock]
) -> None:
    collaborators["chat_repository"].get_session.return_value = None

    with pytest.raises(NotFoundError):
        await service.list_messages(session_id=uuid4(), owner_id=uuid4())

    collaborators["chat_repository"].get_all_messages.assert_not_awaited()


async def test_delete_session_requires_existing_session(
    service: RAGChatService, collaborators: dict[str, MagicMock]
) -> None:
    owner_id = uuid4()
    session_id = uuid4()
    session = ChatSession(id=session_id, owner_id=owner_id, title="t")
    collaborators["chat_repository"].get_session.return_value = session

    await service.delete_session(session_id=session_id, owner_id=owner_id)

    collaborators["chat_repository"].delete_session.assert_awaited_once_with(session)


async def test_query_raises_not_found_for_unknown_session(
    service: RAGChatService, collaborators: dict[str, MagicMock]
) -> None:
    collaborators["chat_repository"].get_session.return_value = None

    with pytest.raises(NotFoundError):
        await service.query(session_id=uuid4(), owner_id=uuid4(), question="Q?", top_k=None)

    collaborators["llm_client"].complete.assert_not_awaited()


async def test_query_orchestrates_full_rag_flow(
    service: RAGChatService, collaborators: dict[str, MagicMock]
) -> None:
    owner_id = uuid4()
    session_id = uuid4()
    collaborators["chat_repository"].get_session.return_value = ChatSession(
        id=session_id, owner_id=owner_id, title="t"
    )

    answer, sources, metadata = await service.query(
        session_id=session_id, owner_id=owner_id, question="What is X?", top_k=3
    )

    assert answer == "Mocked LLM answer."
    assert len(sources) == 1
    assert metadata.model == "mock-model"
    assert metadata.retrieved_chunk_count == 1

    collaborators["retriever"].search.assert_awaited_once_with(
        query="What is X?", owner_id=owner_id, top_k=3
    )
    collaborators["context_builder"].build.assert_called_once()
    collaborators["prompt_builder"].build_system_prompt.assert_called_once_with("built context")
    collaborators["llm_client"].complete.assert_awaited_once()

    append_calls = collaborators["conversation_memory"].append_turn.await_args_list
    assert len(append_calls) == 2
    assert append_calls[0].kwargs["role"] == MessageRole.USER
    assert append_calls[0].kwargs["content"] == "What is X?"
    assert append_calls[1].kwargs["role"] == MessageRole.ASSISTANT
    assert append_calls[1].kwargs["content"] == "Mocked LLM answer."
    assert append_calls[1].kwargs["citations"][0]["document_title"] == "Doc Title"
    assert append_calls[1].kwargs["response_metadata"]["model"] == "mock-model"


async def test_stream_query_yields_tokens_then_done_event(
    service: RAGChatService, collaborators: dict[str, MagicMock]
) -> None:
    owner_id = uuid4()
    session_id = uuid4()
    collaborators["chat_repository"].get_session.return_value = ChatSession(
        id=session_id, owner_id=owner_id, title="t"
    )

    events = [
        event
        async for event in service.stream_query(
            session_id=session_id, owner_id=owner_id, question="Stream this?", top_k=None
        )
    ]

    token_events = [event for event in events if event["type"] == "token"]
    done_events = [event for event in events if event["type"] == "done"]

    assert [event["text"] for event in token_events] == ["Mocked ", "streamed ", "answer."]
    assert len(done_events) == 1
    assert done_events[0]["metadata"]["model"] == "mock-model"
    assert done_events[0]["sources"][0]["document_title"] == "Doc Title"

    append_calls = collaborators["conversation_memory"].append_turn.await_args_list
    assert append_calls[1].kwargs["content"] == "Mocked streamed answer."
