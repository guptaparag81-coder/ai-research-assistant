"""Tests for registration and login."""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_register_and_login(client: AsyncClient) -> None:
    register_response = await client.post(
        "/api/v1/auth/register",
        json={"email": "new.user@example.com", "password": "supersecret123"},
    )
    assert register_response.status_code == 201
    assert register_response.json()["email"] == "new.user@example.com"

    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": "new.user@example.com", "password": "supersecret123"},
    )
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()


async def test_register_duplicate_email_is_rejected(client: AsyncClient) -> None:
    payload = {"email": "dupe@example.com", "password": "supersecret123"}
    first = await client.post("/api/v1/auth/register", json=payload)
    second = await client.post("/api/v1/auth/register", json=payload)

    assert first.status_code == 201
    assert second.status_code == 409


async def test_login_with_wrong_password_is_unauthorized(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "wrongpass@example.com", "password": "supersecret123"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "wrongpass@example.com", "password": "not-the-password"},
    )
    assert response.status_code == 401


async def test_documents_require_authentication(client: AsyncClient) -> None:
    response = await client.get("/api/v1/documents")
    assert response.status_code == 401
