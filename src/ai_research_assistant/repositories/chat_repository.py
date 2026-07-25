"""Data access layer for ChatSession and ChatMessage entities."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ai_research_assistant.db.models.chat import ChatMessage, ChatSession


class ChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_session(self, chat_session: ChatSession) -> ChatSession:
        self._session.add(chat_session)
        await self._session.flush()
        return chat_session

    async def get_session(self, session_id: UUID, owner_id: UUID) -> ChatSession | None:
        result = await self._session.execute(
            select(ChatSession)
            .options(selectinload(ChatSession.messages))
            .where(ChatSession.id == session_id, ChatSession.owner_id == owner_id)
        )
        return result.scalar_one_or_none()

    async def list_sessions(self, owner_id: UUID) -> list[ChatSession]:
        result = await self._session.execute(
            select(ChatSession)
            .where(ChatSession.owner_id == owner_id)
            .order_by(ChatSession.created_at.desc())
        )
        return list(result.scalars().all())

    async def add_message(self, message: ChatMessage) -> ChatMessage:
        self._session.add(message)
        await self._session.flush()
        return message

    async def get_recent_messages(self, session_id: UUID, limit: int) -> list[ChatMessage]:
        result = await self._session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    async def get_all_messages(self, session_id: UUID) -> list[ChatMessage]:
        result = await self._session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        return list(result.scalars().all())

    async def delete_session(self, chat_session: ChatSession) -> None:
        await self._session.delete(chat_session)
        await self._session.flush()
