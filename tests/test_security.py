"""Unit tests for password hashing and JWT helpers."""

from uuid import uuid4

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_round_trips() -> None:
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)


def test_verify_password_rejects_wrong_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert not verify_password("wrong password", hashed)


def test_hash_password_truncates_long_passwords_safely() -> None:
    long_password = "a" * 200
    hashed = hash_password(long_password)
    assert verify_password(long_password, hashed)
    assert verify_password("a" * 72, hashed)


def test_create_and_decode_access_token_round_trips() -> None:
    settings = Settings(secret_key="test-secret")
    user_id = uuid4()

    token = create_access_token(subject=user_id, settings=settings)
    decoded = decode_access_token(token, settings)

    assert decoded == user_id


def test_decode_access_token_rejects_invalid_token() -> None:
    settings = Settings(secret_key="test-secret")
    assert decode_access_token("not-a-valid-token", settings) is None


def test_decode_access_token_rejects_token_from_different_secret() -> None:
    settings_a = Settings(secret_key="secret-a")
    settings_b = Settings(secret_key="secret-b")
    token = create_access_token(subject=uuid4(), settings=settings_a)

    assert decode_access_token(token, settings_b) is None


def test_decode_access_token_rejects_non_uuid_subject() -> None:
    import jose.jwt as jose_jwt

    settings = Settings(secret_key="test-secret")
    token = jose_jwt.encode(
        {"sub": "not-a-uuid", "type": "access"},
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    assert decode_access_token(token, settings) is None


def test_decode_access_token_rejects_missing_subject() -> None:
    import jose.jwt as jose_jwt

    settings = Settings(secret_key="test-secret")
    token = jose_jwt.encode(
        {"type": "access"},
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    assert decode_access_token(token, settings) is None
