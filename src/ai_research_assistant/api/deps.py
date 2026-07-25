"""Centralized FastAPI dependency injection providers."""

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from ai_research_assistant.core.config import Settings, get_settings
from ai_research_assistant.core.exceptions import UnauthorizedError
from ai_research_assistant.core.security import decode_access_token
from ai_research_assistant.db.models.user import User
from ai_research_assistant.repositories.chat_repository import ChatRepository
from ai_research_assistant.repositories.document_repository import DocumentRepository
from ai_research_assistant.repositories.user_repository import UserRepository
from ai_research_assistant.services.document_service import DocumentService
from ai_research_assistant.services.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.services.ingestion.pipeline import IngestionPipeline
from ai_research_assistant.services.llm.llm_client import LLMClient
from ai_research_assistant.services.memory.conversation_memory import ConversationMemoryService
from ai_research_assistant.services.retrieval.context_builder import ContextBuilder
from ai_research_assistant.services.retrieval.prompt_builder import PromptBuilder
from ai_research_assistant.services.retrieval.rag_service import RAGChatService
from ai_research_assistant.services.retrieval.retriever import Retriever
from ai_research_assistant.services.vectorstore.chroma_store import ChromaVectorStore

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession]:
    async for session in request.app.state.db_manager.session():
        yield session


DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_vector_store(request: Request) -> ChromaVectorStore:
    return request.app.state.vector_store  # type: ignore[no-any-return]


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service  # type: ignore[no-any-return]


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client  # type: ignore[no-any-return]


VectorStoreDep = Annotated[ChromaVectorStore, Depends(get_vector_store)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]


def get_user_repository(session: DbSessionDep) -> UserRepository:
    return UserRepository(session)


def get_document_repository(session: DbSessionDep) -> DocumentRepository:
    return DocumentRepository(session)


def get_chat_repository(session: DbSessionDep) -> ChatRepository:
    return ChatRepository(session)


UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
DocumentRepositoryDep = Annotated[DocumentRepository, Depends(get_document_repository)]
ChatRepositoryDep = Annotated[ChatRepository, Depends(get_chat_repository)]


async def get_current_user(
    settings: SettingsDep,
    user_repository: UserRepositoryDep,
    token: Annotated[str | None, Depends(_oauth2_scheme)] = None,
) -> User:
    if token is None:
        raise UnauthorizedError("Missing authentication credentials")
    user_id = decode_access_token(token, settings)
    if user_id is None:
        raise UnauthorizedError("Invalid or expired access token")
    user = await user_repository.get_by_id(user_id)
    if user is None or not user.is_active:
        raise UnauthorizedError("User not found or inactive")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]


def get_ingestion_pipeline(
    embedding_service: EmbeddingServiceDep,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
) -> IngestionPipeline:
    return IngestionPipeline(
        embedding_service=embedding_service, vector_store=vector_store, settings=settings
    )


IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]


def get_document_service(
    document_repository: DocumentRepositoryDep,
    ingestion_pipeline: IngestionPipelineDep,
    settings: SettingsDep,
) -> DocumentService:
    return DocumentService(
        document_repository=document_repository,
        ingestion_pipeline=ingestion_pipeline,
        settings=settings,
    )


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


def get_retriever(
    vector_store: VectorStoreDep,
    embedding_service: EmbeddingServiceDep,
    settings: SettingsDep,
) -> Retriever:
    return Retriever(
        vector_store=vector_store, embedding_service=embedding_service, settings=settings
    )


@lru_cache
def _context_builder() -> ContextBuilder:
    return ContextBuilder()


@lru_cache
def _prompt_builder() -> PromptBuilder:
    return PromptBuilder()


def get_context_builder() -> ContextBuilder:
    return _context_builder()


def get_prompt_builder() -> PromptBuilder:
    return _prompt_builder()


def get_conversation_memory(
    chat_repository: ChatRepositoryDep, settings: SettingsDep
) -> ConversationMemoryService:
    return ConversationMemoryService(chat_repository=chat_repository, settings=settings)


RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
ContextBuilderDep = Annotated[ContextBuilder, Depends(get_context_builder)]
PromptBuilderDep = Annotated[PromptBuilder, Depends(get_prompt_builder)]
ConversationMemoryDep = Annotated[ConversationMemoryService, Depends(get_conversation_memory)]


def get_rag_chat_service(
    chat_repository: ChatRepositoryDep,
    retriever: RetrieverDep,
    context_builder: ContextBuilderDep,
    prompt_builder: PromptBuilderDep,
    conversation_memory: ConversationMemoryDep,
    llm_client: LLMClientDep,
) -> RAGChatService:
    return RAGChatService(
        chat_repository=chat_repository,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        conversation_memory=conversation_memory,
        llm_client=llm_client,
    )


RAGChatServiceDep = Annotated[RAGChatService, Depends(get_rag_chat_service)]
