"""FastAPI application entrypoint and lifespan wiring."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_research_assistant.api.router import api_router
from ai_research_assistant.core.config import get_settings
from ai_research_assistant.core.exceptions import register_exception_handlers
from ai_research_assistant.core.logging import configure_logging, get_logger
from ai_research_assistant.db.session import DatabaseSessionManager
from ai_research_assistant.services.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.services.llm.llm_client import LLMClient
from ai_research_assistant.services.vectorstore.chroma_store import (
    ChromaVectorStore,
    create_chroma_client,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    configure_logging(settings)
    logger.info("application_startup", environment=settings.environment)

    app.state.db_manager = DatabaseSessionManager(settings)
    app.state.vector_store = ChromaVectorStore(create_chroma_client(settings), settings)
    app.state.embedding_service = EmbeddingService(settings)
    app.state.llm_client = LLMClient(settings)

    yield

    logger.info("application_shutdown")
    await app.state.db_manager.close()


_OPENAPI_TAGS = [
    {"name": "health", "description": "Service liveness and database connectivity."},
    {"name": "version", "description": "Build and environment metadata."},
    {"name": "auth", "description": "User registration and authentication."},
    {"name": "documents", "description": "Upload, search, retrieve, and delete documents."},
    {
        "name": "chat",
        "description": "Chat sessions, conversation history, and retrieval-augmented queries.",
    },
]


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        description=(
            "Retrieval-augmented AI research assistant API: document ingestion, "
            "semantic search, and grounded, cited chat over your own knowledge base."
        ),
        contact={"name": "AI Research Assistant Team"},
        license_info={"name": "MIT"},
        openapi_tags=_OPENAPI_TAGS,
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    return app


app = create_app()
