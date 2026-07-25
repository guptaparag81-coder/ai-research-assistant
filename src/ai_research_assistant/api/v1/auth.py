"""Authentication endpoints: registration and login."""

from fastapi import APIRouter, status

from ai_research_assistant.api.deps import SettingsDep, UserRepositoryDep
from ai_research_assistant.core.exceptions import ConflictError, UnauthorizedError
from ai_research_assistant.core.security import create_access_token, hash_password, verify_password
from ai_research_assistant.db.models.user import User
from ai_research_assistant.schemas.user import TokenResponse, UserCreate, UserLogin, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


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
    user = await user_repository.get_by_email(payload.email)
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise UnauthorizedError("Invalid email or password")
    if not user.is_active:
        raise UnauthorizedError("User account is inactive")

    token = create_access_token(subject=user.id, settings=settings)
    return TokenResponse(access_token=token)
