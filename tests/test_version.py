"""Tests for the version endpoint."""

import pytest
from httpx import AsyncClient

from ai_research_assistant.core.config import get_settings

pytestmark = pytest.mark.asyncio


async def test_version_matches_settings(client: AsyncClient) -> None:
    settings = get_settings()

    response = await client.get("/api/v1/version")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == settings.app_name
    assert body["version"] == settings.app_version
    assert body["environment"] == settings.environment
