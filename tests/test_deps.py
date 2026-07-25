"""Direct unit tests for FastAPI dependency provider functions in api/deps.py."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai_research_assistant.api import deps
from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import UnauthorizedError
from ai_research_assistant.core.security import create_access_token
from ai_research_assistant.db.models.user import User


def _fake_request(**state: object) -> MagicMock:
    request = MagicMock()
    for key, value in state.items():
        setattr(request.app.state, key, value)
    return request


async def test_get_db_session_yields_from_db_manager() -> None:
    async def fake_session_gen() -> AsyncGenerator[str]:
        yield "the-session"

    db_manager = MagicMock()
    db_manager.session = MagicMock(return_value=fake_session_gen())
    request = _fake_request(db_manager=db_manager)

    sessions = [session async for session in deps.get_db_session(request)]

    assert sessions == ["the-session"]  # type: ignore[comparison-overlap]


def test_get_vector_store_reads_app_state() -> None:
    request = _fake_request(vector_store="the-vector-store")
    assert deps.get_vector_store(request) == "the-vector-store"  # type: ignore[comparison-overlap]


def test_get_embedding_service_reads_app_state() -> None:
    request = _fake_request(embedding_service="the-embedding-service")
    assert (
        deps.get_embedding_service(request)  # type: ignore[comparison-overlap]
        == "the-embedding-service"
    )


def test_get_llm_client_reads_app_state() -> None:
    request = _fake_request(llm_client="the-llm-client")
    assert deps.get_llm_client(request) == "the-llm-client"  # type: ignore[comparison-overlap]


async def test_get_current_user_rejects_missing_token() -> None:
    settings = Settings()
    user_repository = MagicMock()

    with pytest.raises(UnauthorizedError):
        await deps.get_current_user(settings, user_repository, None)


async def test_get_current_user_rejects_invalid_token() -> None:
    settings = Settings(secret_key="s")
    user_repository = MagicMock()

    with pytest.raises(UnauthorizedError):
        await deps.get_current_user(settings, user_repository, "not-a-real-token")


async def test_get_current_user_rejects_unknown_user() -> None:
    settings = Settings(secret_key="s")
    token = create_access_token(subject=uuid4(), settings=settings)
    user_repository = MagicMock()
    user_repository.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(UnauthorizedError):
        await deps.get_current_user(settings, user_repository, token)


async def test_get_current_user_rejects_inactive_user() -> None:
    settings = Settings(secret_key="s")
    user_id = uuid4()
    token = create_access_token(subject=user_id, settings=settings)
    user_repository = MagicMock()
    user_repository.get_by_id = AsyncMock(
        return_value=User(id=user_id, email="a@b.com", hashed_password="x", is_active=False)
    )

    with pytest.raises(UnauthorizedError):
        await deps.get_current_user(settings, user_repository, token)


async def test_get_current_user_returns_active_user() -> None:
    settings = Settings(secret_key="s")
    user_id = uuid4()
    token = create_access_token(subject=user_id, settings=settings)
    expected_user = User(id=user_id, email="a@b.com", hashed_password="x", is_active=True)
    user_repository = MagicMock()
    user_repository.get_by_id = AsyncMock(return_value=expected_user)

    user = await deps.get_current_user(settings, user_repository, token)

    assert user is expected_user
