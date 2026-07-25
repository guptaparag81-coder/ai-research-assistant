"""API tests for chat sessions, conversation history, and streaming queries."""

import json
from typing import Any

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_and_get_chat_session(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    create_response = await client.post("/api/v1/chat/sessions", headers=auth_headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    get_response = await client.get(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers)

    assert get_response.status_code == 200
    body = get_response.json()
    assert body["id"] == session_id
    assert body["messages"] == []


async def test_get_chat_session_not_found(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    missing_id = "00000000-0000-0000-0000-000000000000"
    response = await client.get(f"/api/v1/chat/sessions/{missing_id}", headers=auth_headers)
    assert response.status_code == 404


async def test_list_chat_sessions(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    await client.post("/api/v1/chat/sessions", headers=auth_headers)
    await client.post("/api/v1/chat/sessions", headers=auth_headers)

    response = await client.get("/api/v1/chat/sessions", headers=auth_headers)

    assert response.status_code == 200
    assert len(response.json()) == 2


async def test_conversation_history_reflects_persisted_messages(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    files = {"file": ("history.txt", b"Some knowledge base content.", "text/plain")}
    await client.post("/api/v1/documents", headers=auth_headers, files=files)

    session_response = await client.post("/api/v1/chat/sessions", headers=auth_headers)
    session_id = session_response.json()["id"]

    await client.post(
        f"/api/v1/chat/sessions/{session_id}/query",
        headers=auth_headers,
        json={"question": "What is in the knowledge base?"},
    )

    messages_response = await client.get(
        f"/api/v1/chat/sessions/{session_id}/messages", headers=auth_headers
    )

    assert messages_response.status_code == 200
    messages = messages_response.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["citations"] is not None
    assert messages[1]["response_metadata"]["model"] == "fake-model"


async def test_list_messages_for_unknown_session_returns_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    missing_id = "00000000-0000-0000-0000-000000000000"
    response = await client.get(
        f"/api/v1/chat/sessions/{missing_id}/messages", headers=auth_headers
    )
    assert response.status_code == 404


async def test_delete_chat_session(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    session_response = await client.post("/api/v1/chat/sessions", headers=auth_headers)
    session_id = session_response.json()["id"]

    delete_response = await client.delete(
        f"/api/v1/chat/sessions/{session_id}", headers=auth_headers
    )
    get_response = await client.get(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers)

    assert delete_response.status_code == 204
    assert get_response.status_code == 404


async def test_delete_chat_session_not_found(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    missing_id = "00000000-0000-0000-0000-000000000000"
    response = await client.delete(f"/api/v1/chat/sessions/{missing_id}", headers=auth_headers)
    assert response.status_code == 404


async def test_streaming_query_yields_token_and_done_events(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    session_response = await client.post("/api/v1/chat/sessions", headers=auth_headers)
    session_id = session_response.json()["id"]

    events: list[dict[str, Any]] = []
    async with client.stream(
        "POST",
        f"/api/v1/chat/sessions/{session_id}/query/stream",
        headers=auth_headers,
        json={"question": "Stream me an answer"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
        for block in buffer.split("\n\n"):
            if not block.strip():
                continue
            data_line = next(line for line in block.splitlines() if line.startswith("data: "))
            events.append(json.loads(data_line[len("data: ") :]))

    token_events = [event for event in events if event["type"] == "token"]
    done_events = [event for event in events if event["type"] == "done"]

    assert len(token_events) > 0
    assert len(done_events) == 1
    assert done_events[0]["metadata"]["model"] == "fake-model"

    history_response = await client.get(
        f"/api/v1/chat/sessions/{session_id}/messages", headers=auth_headers
    )
    messages = history_response.json()
    assert len(messages) == 2
    assert messages[1]["role"] == "assistant"


async def test_streaming_query_unknown_session_returns_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    missing_id = "00000000-0000-0000-0000-000000000000"
    response = await client.post(
        f"/api/v1/chat/sessions/{missing_id}/query/stream",
        headers=auth_headers,
        json={"question": "Anything?"},
    )
    assert response.status_code == 404
