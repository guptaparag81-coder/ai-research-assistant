"""Unit tests for LLMClient, mocking the underlying ChatOpenAI client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import LLMServiceError
from ai_research_assistant.services.llm.llm_client import LLMClient, _to_langchain_message


@pytest.fixture
def client() -> LLMClient:
    # `_chat` is a plain (non-pydantic) attribute on our wrapper, so it can be swapped
    # wholesale for a mock; the real ChatOpenAI pydantic model rejects attribute
    # assignment for fields it doesn't declare.
    instance = LLMClient(Settings(llm_model="test-model"))
    instance._chat = MagicMock()
    return instance


def test_model_name_reflects_settings(client: LLMClient) -> None:
    assert client.model_name == "test-model"


def test_to_langchain_message_maps_system_role() -> None:
    message = _to_langchain_message("system", "be helpful")
    assert message.content == "be helpful"
    assert message.type == "system"


async def test_complete_returns_plain_text_content(client: LLMClient) -> None:
    response = MagicMock()
    response.content = "Plain answer"
    client._chat.ainvoke = AsyncMock(return_value=response)  # type: ignore[method-assign]

    result = await client.complete(system_prompt="sys", messages=[("user", "hi")])

    assert result == "Plain answer"


async def test_complete_flattens_list_content(client: LLMClient) -> None:
    response = MagicMock()
    response.content = ["part-a", "part-b"]
    client._chat.ainvoke = AsyncMock(return_value=response)  # type: ignore[method-assign]

    result = await client.complete(system_prompt="sys", messages=[])

    assert result == "part-apart-b"


async def test_complete_wraps_provider_errors(client: LLMClient) -> None:
    client._chat.ainvoke = AsyncMock(side_effect=RuntimeError("provider down"))  # type: ignore[method-assign]

    with pytest.raises(LLMServiceError):
        await client.complete(system_prompt="sys", messages=[("user", "hi")])


async def test_stream_yields_nonempty_chunks_and_skips_empty(client: LLMClient) -> None:
    chunk_a = MagicMock(content="hello ")
    chunk_b = MagicMock(content="")
    chunk_c = MagicMock(content="world")

    async def fake_astream(_messages: object) -> object:
        for chunk in (chunk_a, chunk_b, chunk_c):
            yield chunk

    client._chat.astream = fake_astream  # type: ignore[method-assign, assignment]

    tokens = [
        token
        async for token in client.stream(system_prompt="sys", messages=[("assistant", "prior")])
    ]

    assert tokens == ["hello ", "world"]


async def test_stream_wraps_provider_errors(client: LLMClient) -> None:
    async def fake_astream(_messages: object) -> object:
        raise RuntimeError("stream broke")
        yield  # pragma: no cover - unreachable, satisfies generator syntax

    client._chat.astream = fake_astream  # type: ignore[method-assign, assignment]

    with pytest.raises(LLMServiceError):
        async for _token in client.stream(system_prompt="sys", messages=[]):
            pass
