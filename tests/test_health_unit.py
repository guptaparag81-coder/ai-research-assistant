"""Direct unit tests for the health endpoint function, covering the failure branch."""

from unittest.mock import AsyncMock, MagicMock

from ai_research_assistant.api.v1.health import health


async def test_health_reports_ok_when_database_succeeds() -> None:
    session = MagicMock()
    session.execute = AsyncMock(return_value=None)

    result = await health(session)

    assert result.status == "ok"
    assert result.database == "ok"


async def test_health_reports_unavailable_when_database_raises() -> None:
    session = MagicMock()
    session.execute = AsyncMock(side_effect=RuntimeError("connection lost"))

    result = await health(session)

    assert result.status == "ok"
    assert result.database == "unavailable"
