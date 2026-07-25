"""Aggregates all v1 API routers."""

from fastapi import APIRouter

from ai_research_assistant.api.v1 import auth, chat, documents, health, version

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(version.router)
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
