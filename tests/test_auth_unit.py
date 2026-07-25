"""Direct unit tests for the auth endpoint functions (register/login)."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.security import OAuth2PasswordRequestForm

from ai_research_assistant.api.v1.auth import login, login_via_oauth2_form, register
from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import ConflictError, UnauthorizedError
from ai_research_assistant.core.security import hash_password
from ai_research_assistant.db.models.user import User
from ai_research_assistant.schemas.user import UserCreate, UserLogin


async def test_register_creates_new_user() -> None:
    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock(return_value=None)
    user_repository.create = AsyncMock(side_effect=lambda user: user)

    result = await register(
        UserCreate(email="new@example.com", password="password123"), user_repository
    )

    assert result.email == "new@example.com"
    user_repository.create.assert_awaited_once()


async def test_register_rejects_duplicate_email() -> None:
    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock(
        return_value=User(email="dupe@example.com", hashed_password="x")
    )

    with pytest.raises(ConflictError):
        await register(
            UserCreate(email="dupe@example.com", password="password123"), user_repository
        )


async def test_login_succeeds_with_correct_credentials() -> None:
    user = User(
        id=uuid4(),
        email="a@example.com",
        hashed_password=hash_password("password123"),
        is_active=True,
    )
    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock(return_value=user)
    settings = Settings(secret_key="test-secret")

    token_response = await login(
        UserLogin(email="a@example.com", password="password123"), user_repository, settings
    )

    assert token_response.access_token
    assert token_response.token_type == "bearer"


async def test_login_rejects_unknown_email() -> None:
    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock(return_value=None)
    settings = Settings(secret_key="test-secret")

    with pytest.raises(UnauthorizedError):
        await login(
            UserLogin(email="missing@example.com", password="password123"),
            user_repository,
            settings,
        )


async def test_login_rejects_wrong_password() -> None:
    user = User(email="a@example.com", hashed_password=hash_password("correct-password"))
    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock(return_value=user)
    settings = Settings(secret_key="test-secret")

    with pytest.raises(UnauthorizedError):
        await login(
            UserLogin(email="a@example.com", password="wrong-password"), user_repository, settings
        )


async def test_login_rejects_inactive_user() -> None:
    user = User(
        id=uuid4(),
        email="a@example.com",
        hashed_password=hash_password("password123"),
        is_active=False,
    )
    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock(return_value=user)
    settings = Settings(secret_key="test-secret")

    with pytest.raises(UnauthorizedError):
        await login(
            UserLogin(email="a@example.com", password="password123"), user_repository, settings
        )


async def test_oauth2_form_login_succeeds_with_correct_credentials() -> None:
    """Covers the Swagger UI Authorize dialog's code path directly (the OAuth2
    password-flow form-data endpoint), mirroring test_login_succeeds_with_correct_credentials."""
    user = User(
        id=uuid4(),
        email="a@example.com",
        hashed_password=hash_password("password123"),
        is_active=True,
    )
    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock(return_value=user)
    settings = Settings(secret_key="test-secret")
    form_data = OAuth2PasswordRequestForm(username="a@example.com", password="password123")

    token_response = await login_via_oauth2_form(form_data, user_repository, settings)

    assert token_response.access_token
    assert token_response.token_type == "bearer"


async def test_oauth2_form_login_rejects_wrong_password() -> None:
    user = User(email="a@example.com", hashed_password=hash_password("correct-password"))
    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock(return_value=user)
    settings = Settings(secret_key="test-secret")
    form_data = OAuth2PasswordRequestForm(username="a@example.com", password="wrong-password")

    with pytest.raises(UnauthorizedError):
        await login_via_oauth2_form(form_data, user_repository, settings)
