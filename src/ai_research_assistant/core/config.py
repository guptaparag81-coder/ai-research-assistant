"""Application configuration loaded from environment variables."""

import getpass
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "AI Research Assistant"
    app_version: str = "0.1.0"
    environment: Literal["local", "test", "staging", "production"] = "local"
    debug: bool = False
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Database ---
    # Defaults to the current OS user with no password, matching Homebrew
    # PostgreSQL's default superuser role (peer/trust auth). Override via
    # POSTGRES_USER / POSTGRES_PASSWORD for other setups (e.g. Docker's
    # postgres/postgres role).
    postgres_user: str = Field(default_factory=getpass.getuser)
    postgres_password: SecretStr = SecretStr("")
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ai_research_assistant"
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_echo: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:"
            f"{self.postgres_password.get_secret_value()}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_database_url(self) -> str:
        """Synchronous DSN, used by Alembic migrations."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:"
            f"{self.postgres_password.get_secret_value()}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    # --- Auth ---
    secret_key: SecretStr = SecretStr("change-me-in-production")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # --- LLM (OpenAI-compatible) ---
    llm_api_key: SecretStr = SecretStr("sk-placeholder")
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- ChromaDB ---
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "documents"
    chroma_ssl: bool = False

    # --- Ingestion ---
    max_upload_size_bytes: int = 25 * 1024 * 1024
    chunk_size: int = 1000
    chunk_overlap: int = 150
    allowed_content_types: tuple[str, ...] = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
    )

    # --- Retrieval ---
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.0
    conversation_memory_max_turns: int = 10


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide settings instance."""
    return Settings()
