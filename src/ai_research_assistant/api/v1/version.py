"""Version information endpoint."""

from fastapi import APIRouter

from ai_research_assistant.api.deps import SettingsDep
from ai_research_assistant.schemas.common import VersionResponse

router = APIRouter(tags=["version"])


@router.get("/version", response_model=VersionResponse)
async def version(settings: SettingsDep) -> VersionResponse:
    """Return the running application name, version, and environment."""
    return VersionResponse(
        name=settings.app_name, version=settings.app_version, environment=settings.environment
    )
