"""Application service coordinating document upload and ingestion."""

from uuid import UUID

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import FileTooLargeError, NotFoundError
from ai_research_assistant.core.logging import get_logger
from ai_research_assistant.db.models.document import Document, DocumentStatus
from ai_research_assistant.repositories.document_repository import DocumentRepository
from ai_research_assistant.services.ingestion.loaders import resolve_document_type
from ai_research_assistant.services.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)


class DocumentService:
    """Coordinates document persistence and the ingestion pipeline."""

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        ingestion_pipeline: IngestionPipeline,
        settings: Settings,
    ) -> None:
        self._document_repository = document_repository
        self._ingestion_pipeline = ingestion_pipeline
        self._settings = settings

    async def ingest_upload(
        self,
        *,
        owner_id: UUID,
        filename: str,
        content_type: str,
        raw: bytes,
    ) -> Document:
        """Validate, persist, and ingest an uploaded file end-to-end."""
        if len(raw) > self._settings.max_upload_size_bytes:
            raise FileTooLargeError(
                f"File exceeds maximum allowed size of "
                f"{self._settings.max_upload_size_bytes} bytes"
            )

        document_type = resolve_document_type(content_type)
        document = Document(
            owner_id=owner_id,
            filename=filename,
            document_type=document_type,
            content_type=content_type,
            file_size_bytes=len(raw),
            status=DocumentStatus.PROCESSING,
        )
        await self._document_repository.create(document)

        try:
            extracted = self._ingestion_pipeline.extract(raw=raw, content_type=content_type)
            document.title = extracted.title or filename
            chunks = await self._ingestion_pipeline.process(document=document, extracted=extracted)
            await self._document_repository.add_chunks(chunks)
            document.chunk_count = len(chunks)
            document.status = DocumentStatus.READY
            await self._document_repository.save(document)
        except Exception as exc:
            document.status = DocumentStatus.FAILED
            document.error_message = str(exc)
            logger.error("document_ingestion_failed", document_id=str(document.id), error=str(exc))
            # Commit the failure state now: the exception below will unwind through
            # the request-scoped session, which rolls back on error and would
            # otherwise silently erase this FAILED row along with it.
            await self._document_repository.save(document)
            await self._document_repository.commit()
            raise

        return document

    async def get_document(self, *, document_id: UUID, owner_id: UUID) -> Document:
        document = await self._document_repository.get_by_id(document_id, owner_id)
        if document is None:
            raise NotFoundError(f"Document {document_id} not found")
        return document

    async def list_documents(self, *, owner_id: UUID) -> list[Document]:
        return await self._document_repository.list_by_owner(owner_id)

    async def delete_document(self, *, document_id: UUID, owner_id: UUID) -> None:
        document = await self.get_document(document_id=document_id, owner_id=owner_id)
        self._ingestion_pipeline.delete_document_vectors(document.id)
        await self._document_repository.delete(document)
