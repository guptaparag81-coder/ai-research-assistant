"""Tests for document upload and ingestion."""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_upload_txt_document_is_ingested(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    files = {
        "file": (
            "notes.txt",
            b"Photosynthesis converts light energy into chemical energy.",
            "text/plain",
        )
    }

    response = await client.post("/api/v1/documents", headers=auth_headers, files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["status"] == "ready"
    assert body["chunk_count"] >= 1


async def test_upload_rejects_unsupported_content_type(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    files = {"file": ("image.png", b"\x89PNG\r\n", "image/png")}

    response = await client.post("/api/v1/documents", headers=auth_headers, files=files)

    assert response.status_code == 415


async def test_list_documents_returns_only_owned_documents(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    files = {"file": ("a.md", b"# Title\n\nSome markdown content.", "text/markdown")}
    await client.post("/api/v1/documents", headers=auth_headers, files=files)

    response = await client.get("/api/v1/documents", headers=auth_headers)

    assert response.status_code == 200
    documents = response.json()
    assert len(documents) == 1
    assert documents[0]["filename"] == "a.md"


async def test_get_document_by_id(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    files = {"file": ("b.txt", b"Some plain text content for testing.", "text/plain")}
    upload_response = await client.post("/api/v1/documents", headers=auth_headers, files=files)
    document_id = upload_response.json()["id"]

    response = await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["id"] == document_id


async def test_get_document_not_found(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    missing_id = "00000000-0000-0000-0000-000000000000"
    response = await client.get(f"/api/v1/documents/{missing_id}", headers=auth_headers)
    assert response.status_code == 404


async def test_delete_document_removes_it(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    files = {"file": ("c.txt", b"Content to be deleted shortly.", "text/plain")}
    upload_response = await client.post("/api/v1/documents", headers=auth_headers, files=files)
    document_id = upload_response.json()["id"]

    delete_response = await client.delete(f"/api/v1/documents/{document_id}", headers=auth_headers)
    get_response = await client.get(f"/api/v1/documents/{document_id}", headers=auth_headers)

    assert delete_response.status_code == 204
    assert get_response.status_code == 404


async def test_delete_document_not_found(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    missing_id = "00000000-0000-0000-0000-000000000000"
    response = await client.delete(f"/api/v1/documents/{missing_id}", headers=auth_headers)
    assert response.status_code == 404


async def test_search_documents_returns_matching_chunks(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    files = {
        "file": (
            "search-me.txt",
            b"The mitochondria is the powerhouse of the cell.",
            "text/plain",
        )
    }
    await client.post("/api/v1/documents", headers=auth_headers, files=files)

    response = await client.post(
        "/api/v1/documents/search",
        headers=auth_headers,
        json={"query": "What powers the cell?"},
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) >= 1
    assert results[0]["document_title"] == "search-me.txt"
    assert "score" in results[0]


async def test_search_documents_respects_top_k(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    for name in ("one.txt", "two.txt", "three.txt"):
        files = {"file": (name, f"Content of {name}".encode(), "text/plain")}
        await client.post("/api/v1/documents", headers=auth_headers, files=files)

    response = await client.post(
        "/api/v1/documents/search",
        headers=auth_headers,
        json={"query": "content", "top_k": 1},
    )

    assert response.status_code == 200
    assert len(response.json()) == 1


async def test_search_documents_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/documents/search", json={"query": "anything"})
    assert response.status_code == 401
