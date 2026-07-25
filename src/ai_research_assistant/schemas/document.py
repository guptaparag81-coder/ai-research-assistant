"""Document request/response schemas."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ai_research_assistant.db.models.document import DocumentStatus, DocumentType


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    title: str | None
    document_type: DocumentType
    content_type: str
    file_size_bytes: int
    status: DocumentStatus
    chunk_count: int
    error_message: str | None


class DocumentChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    chunk_index: int
    content: str
    token_count: int


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_id: UUID | None = None


class DocumentSearchResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    document_title: str | None
    chunk_index: int
    content: str
    score: float
