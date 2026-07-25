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


async def test_login_rejects_non_json_body_without_crashing(client: AsyncClient) -> None:
    """Regression test: a non-JSON body (e.g. what the Swagger UI "Authorize"
    dialog sends) used to embed raw request-body bytes in the validation error,
    crashing JSONResponse with `TypeError: Object of type bytes is not JSON
    serializable` instead of returning a clean 422."""
    response = await client.post(
        "/api/v1/auth/login",
        content=b"username=someone@example.com&password=whatever&grant_type=password",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    # The offending bytes must have been converted to a string, not omitted.
    errors = body["error"]["details"]["errors"]
    assert errors
    assert all(not isinstance(error.get("input"), bytes) for error in errors)


async def test_register_rejects_non_json_body_without_crashing(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        content=b"not=json",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_oauth2_token_endpoint_succeeds_with_form_credentials(
    client: AsyncClient,
) -> None:
    """This is exactly the request the Swagger UI "Authorize" dialog sends."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "swagger.user@example.com", "password": "supersecret123"},
    )

    response = await client.post(
        "/api/v1/auth/token",
        data={"username": "swagger.user@example.com", "password": "supersecret123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"


async def test_oauth2_token_endpoint_rejects_wrong_password(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "swagger.wrong@example.com", "password": "supersecret123"},
    )

    response = await client.post(
        "/api/v1/auth/token",
        data={"username": "swagger.wrong@example.com", "password": "not-the-password"},
    )

    assert response.status_code == 401


async def test_oauth2_token_endpoint_issues_a_usable_bearer_token(
    client: AsyncClient,
) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "swagger.usable@example.com", "password": "supersecret123"},
    )
    token_response = await client.post(
        "/api/v1/auth/token",
        data={"username": "swagger.usable@example.com", "password": "supersecret123"},
    )
    token = token_response.json()["access_token"]

    response = await client.get("/api/v1/documents", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200


async def test_openapi_oauth2_token_url_matches_token_endpoint(client: AsyncClient) -> None:
    """Ensures Swagger UI's "Authorize" dialog is wired to the endpoint that can
    actually accept its form-encoded request."""
    response = await client.get("/openapi.json")
    schemas = response.json()["components"]["securitySchemes"]
    oauth2_scheme = next(iter(schemas.values()))

    assert oauth2_scheme["flows"]["password"]["tokenUrl"] == "/api/v1/auth/token"
