"""Chat request/response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ai_research_assistant.db.models.chat import MessageRole


class SourceCitation(BaseModel):
    document_id: UUID
    document_title: str | None
    chunk_index: int
    score: float
    excerpt: str


class ResponseMetadata(BaseModel):
    model: str
    latency_ms: float
    retrieved_chunk_count: int
    generated_at: datetime


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: MessageRole
    content: str
    citations: list[SourceCitation] | None = None
    response_metadata: ResponseMetadata | None = None
    created_at: datetime


class ChatSessionDetailRead(ChatSessionRead):
    messages: list[ChatMessageRead]


class ChatQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatQueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    session_id: UUID
    metadata: ResponseMetadata
