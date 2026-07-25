"""Document upload, search, and retrieval endpoints."""

from uuid import UUID

from fastapi import APIRouter, UploadFile, status

from ai_research_assistant.api.deps import CurrentUserDep, DocumentServiceDep, RetrieverDep
from ai_research_assistant.core.exceptions import UnsupportedFileTypeError
from ai_research_assistant.db.models.document import Document
from ai_research_assistant.schemas.document import (
    DocumentRead,
    DocumentSearchRequest,
    DocumentSearchResult,
)
from ai_research_assistant.services.retrieval.retriever import RetrievedChunk

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document for ingestion",
)
async def upload_document(
    file: UploadFile,
    current_user: CurrentUserDep,
    document_service: DocumentServiceDep,
) -> Document:
    """Upload a PDF, DOCX, TXT, or Markdown file for ingestion into the knowledge base."""
    if file.content_type is None:
        raise UnsupportedFileTypeError("Missing content type on upload")

    raw = await file.read()
    return await document_service.ingest_upload(
        owner_id=current_user.id,
        filename=file.filename or "untitled",
        content_type=file.content_type,
        raw=raw,
    )


@router.get("", response_model=list[DocumentRead], summary="List the current user's documents")
async def list_documents(
    current_user: CurrentUserDep, document_service: DocumentServiceDep
) -> list[Document]:
    return await document_service.list_documents(owner_id=current_user.id)


@router.post(
    "/search",
    response_model=list[DocumentSearchResult],
    summary="Semantic search over ingested document chunks",
)
async def search_documents(
    payload: DocumentSearchRequest,
    current_user: CurrentUserDep,
    retriever: RetrieverDep,
) -> list[RetrievedChunk]:
    """Run semantic similarity search over the current user's ingested documents."""
    return await retriever.search(
        query=payload.query,
        owner_id=current_user.id,
        document_id=payload.document_id,
        top_k=payload.top_k,
    )


@router.get("/{document_id}", response_model=DocumentRead, summary="Get a document by id")
async def get_document(
    document_id: UUID, current_user: CurrentUserDep, document_service: DocumentServiceDep
) -> Document:
    return await document_service.get_document(document_id=document_id, owner_id=current_user.id)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its embeddings",
)
async def delete_document(
    document_id: UUID, current_user: CurrentUserDep, document_service: DocumentServiceDep
) -> None:
    await document_service.delete_document(document_id=document_id, owner_id=current_user.id)
