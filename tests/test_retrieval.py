"""Tests for the retrieval pipeline: context building, prompts, and end-to-end RAG chat."""

import pytest
from httpx import AsyncClient

from ai_research_assistant.services.retrieval.context_builder import ContextBuilder
from ai_research_assistant.services.retrieval.prompt_builder import PromptBuilder
from ai_research_assistant.services.retrieval.retriever import RetrievedChunk


def test_context_builder_formats_chunks_with_citations() -> None:
    chunks = [
        RetrievedChunk(
            document_id=__import__("uuid").uuid4(),
            document_title="Photosynthesis 101",
            chunk_index=0,
            content="Plants convert light into chemical energy.",
            score=0.87,
        )
    ]

    context = ContextBuilder().build(chunks)

    assert "[Source 1] Photosynthesis 101" in context
    assert "Plants convert light into chemical energy." in context


def test_context_builder_handles_no_results() -> None:
    context = ContextBuilder().build([])
    assert "No relevant context" in context


def test_prompt_builder_embeds_context() -> None:
    prompt = PromptBuilder().build_system_prompt("some retrieved context")
    assert "some retrieved context" in prompt
    assert "AI research assistant" in prompt


@pytest.mark.asyncio
async def test_rag_chat_query_returns_grounded_answer_with_sources(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    files = {
        "file": (
            "photosynthesis.txt",
            b"Photosynthesis converts light energy into chemical energy stored in glucose.",
            "text/plain",
        )
    }
    await client.post("/api/v1/documents", headers=auth_headers, files=files)

    session_response = await client.post("/api/v1/chat/sessions", headers=auth_headers)
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]

    query_response = await client.post(
        f"/api/v1/chat/sessions/{session_id}/query",
        headers=auth_headers,
        json={"question": "What does photosynthesis convert?"},
    )

    assert query_response.status_code == 200
    body = query_response.json()
    assert body["session_id"] == session_id
    assert "Fake answer to" in body["answer"]
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["document_title"] == "photosynthesis.txt"


@pytest.mark.asyncio
async def test_query_unknown_session_returns_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    fake_session_id = "00000000-0000-0000-0000-000000000000"

    response = await client.post(
        f"/api/v1/chat/sessions/{fake_session_id}/query",
        headers=auth_headers,
        json={"question": "Anything?"},
    )

    assert response.status_code == 404
