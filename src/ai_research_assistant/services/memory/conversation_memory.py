"""Conversation memory backed by persisted chat messages."""

from typing import Any
from uuid import UUID

from ai_research_assistant.core.config import Settings
from ai_research_assistant.db.models.chat import ChatMessage, MessageRole
from ai_research_assistant.repositories.chat_repository import ChatRepository


class ConversationMemoryService:
    """Loads and persists bounded conversation history for a chat session."""

    def __init__(self, *, chat_repository: ChatRepository, settings: Settings) -> None:
        self._chat_repository = chat_repository
        self._settings = settings

    async def load_history(self, session_id: UUID) -> list[tuple[str, str]]:
        """Return the most recent turns as (role, content) pairs, oldest first."""
        max_messages = self._settings.conversation_memory_max_turns * 2
        messages = await self._chat_repository.get_recent_messages(session_id, max_messages)
        return [(message.role.value, message.content) for message in messages]

    async def append_turn(
        self,
        *,
        session_id: UUID,
        role: MessageRole,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            citations=citations,
            response_metadata=response_metadata,
        )
        return await self._chat_repository.add_message(message)
