"""Chat session, conversation history, and RAG query endpoints (including streaming)."""

import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from ai_research_assistant.api.deps import CurrentUserDep, RAGChatServiceDep
from ai_research_assistant.db.models.chat import ChatMessage, ChatSession
from ai_research_assistant.schemas.chat import (
    ChatMessageRead,
    ChatQueryRequest,
    ChatQueryResponse,
    ChatSessionDetailRead,
    ChatSessionRead,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/sessions",
    response_model=ChatSessionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new chat session",
)
async def create_chat_session(
    current_user: CurrentUserDep, rag_chat_service: RAGChatServiceDep
) -> ChatSession:
    return await rag_chat_service.create_session(owner_id=current_user.id)


@router.get("/sessions", response_model=list[ChatSessionRead], summary="List chat sessions")
async def list_chat_sessions(
    current_user: CurrentUserDep, rag_chat_service: RAGChatServiceDep
) -> list[ChatSession]:
    return await rag_chat_service.list_sessions(owner_id=current_user.id)


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionDetailRead,
    summary="Get a chat session with its full message history",
)
async def get_chat_session(
    session_id: UUID, current_user: CurrentUserDep, rag_chat_service: RAGChatServiceDep
) -> ChatSession:
    return await rag_chat_service.ensure_session(session_id=session_id, owner_id=current_user.id)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[ChatMessageRead],
    summary="Get the conversation history for a session",
)
async def list_chat_messages(
    session_id: UUID, current_user: CurrentUserDep, rag_chat_service: RAGChatServiceDep
) -> list[ChatMessage]:
    return await rag_chat_service.list_messages(session_id=session_id, owner_id=current_user.id)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session and its messages",
)
async def delete_chat_session(
    session_id: UUID, current_user: CurrentUserDep, rag_chat_service: RAGChatServiceDep
) -> None:
    await rag_chat_service.delete_session(session_id=session_id, owner_id=current_user.id)


@router.post(
    "/sessions/{session_id}/query",
    response_model=ChatQueryResponse,
    summary="Ask a question grounded in the user's documents (RAG)",
)
async def query_chat_session(
    session_id: UUID,
    payload: ChatQueryRequest,
    current_user: CurrentUserDep,
    rag_chat_service: RAGChatServiceDep,
) -> ChatQueryResponse:
    """Answer a question grounded in the user's ingested documents (RAG)."""
    answer, sources, metadata = await rag_chat_service.query(
        session_id=session_id,
        owner_id=current_user.id,
        question=payload.question,
        top_k=payload.top_k,
    )
    return ChatQueryResponse(
        answer=answer, sources=sources, session_id=session_id, metadata=metadata
    )


@router.post(
    "/sessions/{session_id}/query/stream",
    summary="Ask a question and stream the answer as Server-Sent Events",
)
async def stream_chat_session_query(
    session_id: UUID,
    payload: ChatQueryRequest,
    current_user: CurrentUserDep,
    rag_chat_service: RAGChatServiceDep,
) -> StreamingResponse:
    """Stream a RAG answer as SSE `token` events, ending with a `done` event carrying sources."""
    await rag_chat_service.ensure_session(session_id=session_id, owner_id=current_user.id)

    async def event_stream() -> AsyncIterator[str]:
        async for event in rag_chat_service.stream_query(
            session_id=session_id,
            owner_id=current_user.id,
            question=payload.question,
            top_k=payload.top_k,
        ):
            event_type = event["type"]
            yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
