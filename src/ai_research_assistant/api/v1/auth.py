"""Authentication endpoints: registration and login."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from ai_research_assistant.api.deps import SettingsDep, UserRepositoryDep
from ai_research_assistant.core.exceptions import ConflictError, UnauthorizedError
from ai_research_assistant.core.security import create_access_token, hash_password, verify_password
from ai_research_assistant.db.models.user import User
from ai_research_assistant.repositories.user_repository import UserRepository
from ai_research_assistant.schemas.user import TokenResponse, UserCreate, UserLogin, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


async def _authenticate(*, email: str, password: str, user_repository: UserRepository) -> User:
    user = await user_repository.get_by_email(email)
    if user is None or not verify_password(password, user.hashed_password):
        raise UnauthorizedError("Invalid email or password")
    if not user.is_active:
        raise UnauthorizedError("User account is inactive")
    return user


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, user_repository: UserRepositoryDep) -> User:
    existing = await user_repository.get_by_email(payload.email)
    if existing is not None:
        raise ConflictError("A user with this email already exists")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    return await user_repository.create(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: UserLogin, user_repository: UserRepositoryDep, settings: SettingsDep
) -> TokenResponse:
    user = await _authenticate(
        email=payload.email, password=payload.password, user_repository=user_repository
    )
    token = create_access_token(subject=user.id, settings=settings)
    return TokenResponse(access_token=token)


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2 password-flow token endpoint (used by the Swagger UI Authorize dialog)",
)
async def login_via_oauth2_form(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    user_repository: UserRepositoryDep,
    settings: SettingsDep,
) -> TokenResponse:
    """Same authentication as `/login`, but accepting the OAuth2 password-flow's
    form-encoded body (`username`/`password`) instead of JSON, since that is the
    only content type the Swagger UI "Authorize" dialog is able to send."""
    user = await _authenticate(
        email=form_data.username, password=form_data.password, user_repository=user_repository
    )
    token = create_access_token(subject=user.id, settings=settings)
    return TokenResponse(access_token=token)
