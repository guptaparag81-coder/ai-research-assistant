"""Tests for app construction and lifespan wiring, with infra constructors mocked."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_research_assistant import main as main_module
from ai_research_assistant.core.config import get_settings


def test_create_app_sets_metadata() -> None:
    app = main_module.create_app()
    settings = get_settings()

    assert app.title == settings.app_name
    assert app.version == settings.app_version

    openapi_paths = set(app.openapi()["paths"])
    assert f"{settings.api_prefix}/health" in openapi_paths
    assert f"{settings.api_prefix}/version" in openapi_paths
    assert f"{settings.api_prefix}/chat/sessions/{{session_id}}/query/stream" in openapi_paths


def test_module_level_app_instance_is_configured() -> None:
    assert main_module.app.title == get_settings().app_name


@pytest.mark.asyncio
async def test_lifespan_populates_app_state_and_cleans_up_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = main_module.create_app()

    fake_db_manager = MagicMock()
    fake_db_manager.close = AsyncMock()

    monkeypatch.setattr(
        main_module, "DatabaseSessionManager", MagicMock(return_value=fake_db_manager)
    )
    monkeypatch.setattr(main_module, "create_chroma_client", MagicMock(return_value="fake-client"))
    monkeypatch.setattr(
        main_module, "ChromaVectorStore", MagicMock(return_value="fake-vector-store")
    )
    monkeypatch.setattr(
        main_module, "EmbeddingService", MagicMock(return_value="fake-embedding-service")
    )
    monkeypatch.setattr(main_module, "LLMClient", MagicMock(return_value="fake-llm-client"))

    async with main_module.lifespan(app):
        assert app.state.db_manager is fake_db_manager
        assert app.state.vector_store == "fake-vector-store"
        assert app.state.embedding_service == "fake-embedding-service"
        assert app.state.llm_client == "fake-llm-client"

    fake_db_manager.close.assert_awaited_once()
