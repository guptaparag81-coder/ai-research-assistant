"""Unit tests for application settings."""

from ai_research_assistant.core.config import Settings, get_settings


def test_database_url_is_async_dsn() -> None:
    settings = Settings(
        postgres_user="alice",
        postgres_password="secret",
        postgres_host="db.internal",
        postgres_port=5433,
        postgres_db="research",
    )

    assert settings.database_url == "postgresql+asyncpg://alice:secret@db.internal:5433/research"


def test_sync_database_url_is_psycopg_dsn() -> None:
    settings = Settings(
        postgres_user="alice",
        postgres_password="secret",
        postgres_host="db.internal",
        postgres_port=5433,
        postgres_db="research",
    )

    assert (
        settings.sync_database_url == "postgresql+psycopg://alice:secret@db.internal:5433/research"
    )


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
