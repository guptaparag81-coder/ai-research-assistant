"""Tests for the global exception handlers, using a minimal isolated app."""

from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ai_research_assistant.core.exceptions import NotFoundError, register_exception_handlers

pytestmark = pytest.mark.asyncio


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom/app-error")
    async def boom_app_error() -> None:
        raise NotFoundError("thing not found", details={"id": "123"})

    @app.get("/boom/unexpected")
    async def boom_unexpected() -> None:
        raise RuntimeError("something broke")

    @app.get("/boom/validation")
    async def boom_validation(required_param: int) -> dict[str, int]:
        return {"required_param": required_param}

    return app


@pytest.fixture
async def exception_client() -> AsyncGenerator[AsyncClient]:
    # `raise_app_exceptions=False` lets the real 500 response through instead of
    # re-raising, matching how a deployed server behaves (Starlette re-raises by
    # default so test runners can still surface a traceback).
    transport = ASGITransport(app=_build_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


async def test_app_error_handler_returns_structured_payload(exception_client: AsyncClient) -> None:
    response = await exception_client.get("/boom/app-error")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["message"] == "thing not found"
    assert body["error"]["details"] == {"id": "123"}


async def test_unexpected_exception_handler_returns_500(exception_client: AsyncClient) -> None:
    response = await exception_client.get("/boom/unexpected")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"


async def test_validation_error_handler_returns_422(exception_client: AsyncClient) -> None:
    response = await exception_client.get("/boom/validation")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert "errors" in body["error"]["details"]


async def test_http_exception_handler_returns_404_for_unknown_route(
    exception_client: AsyncClient,
) -> None:
    response = await exception_client.get("/this/route/does/not/exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "http_error"
