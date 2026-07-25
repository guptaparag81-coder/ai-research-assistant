"""Health check endpoint."""

from fastapi import APIRouter
from sqlalchemy import text

from ai_research_assistant.api.deps import DbSessionDep
from ai_research_assistant.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(session: DbSessionDep) -> HealthResponse:
    """Report liveness of the API and its database connection."""
    database_status = "ok"
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        database_status = "unavailable"

    return HealthResponse(status="ok", database=database_status)
