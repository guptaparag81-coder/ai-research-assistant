"""Unit tests for the async DB engine/session lifecycle manager."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ai_research_assistant.core.config import Settings
from ai_research_assistant.db import session as session_module


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> session_module.DatabaseSessionManager:
    def fake_create_async_engine(_url: str, **_kwargs: object) -> object:
        return create_async_engine("sqlite+aiosqlite:///:memory:")

    monkeypatch.setattr(session_module, "create_async_engine", fake_create_async_engine)
    return session_module.DatabaseSessionManager(Settings())


async def test_session_commits_on_clean_exit(
    manager: session_module.DatabaseSessionManager,
) -> None:
    agen = manager.session()
    session = await agen.__anext__()
    await session.execute(text("SELECT 1"))

    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()


async def test_session_rolls_back_and_reraises_on_error(
    manager: session_module.DatabaseSessionManager,
) -> None:
    agen = manager.session()
    await agen.__anext__()

    with pytest.raises(RuntimeError):
        await agen.athrow(RuntimeError("boom"))


async def test_close_disposes_engine(manager: session_module.DatabaseSessionManager) -> None:
    assert manager.engine is not None
    await manager.close()
