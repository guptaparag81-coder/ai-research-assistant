"""OpenAI-compatible chat LLM client, built on LangChain's ChatOpenAI.

Because it targets any OpenAI-compatible `base_url`, this works against the
OpenAI API itself or self-hosted/compatible gateways (vLLM, Ollama, etc.)
without code changes.
"""

from collections.abc import AsyncIterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ai_research_assistant.core.config import Settings
from ai_research_assistant.core.exceptions import LLMServiceError
from ai_research_assistant.core.logging import get_logger

logger = get_logger(__name__)


def _to_langchain_message(role: str, content: str) -> BaseMessage:
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return SystemMessage(content=content)


def _build_messages(
    *, system_prompt: str, messages: Sequence[tuple[str, str]]
) -> list[BaseMessage]:
    chat_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    chat_messages.extend(_to_langchain_message(role, content) for role, content in messages)
    return chat_messages


def _content_to_text(content: str | list[str | dict[str, object]]) -> str:
    if isinstance(content, list):
        return "".join(str(part) for part in content)
    return str(content)


class LLMClient:
    """Thin, typed wrapper around an OpenAI-compatible chat completion API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chat = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=60,
        )

    @property
    def model_name(self) -> str:
        return self._settings.llm_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def complete(self, *, system_prompt: str, messages: Sequence[tuple[str, str]]) -> str:
        """Generate a chat completion from a system prompt and (role, content) turns."""
        chat_messages = _build_messages(system_prompt=system_prompt, messages=messages)
        try:
            response = await self._chat.ainvoke(chat_messages)
        except Exception as exc:
            logger.error("llm_completion_failed", error=str(exc))
            raise LLMServiceError("Failed to obtain a response from the language model") from exc

        return _content_to_text(response.content)

    async def stream(
        self, *, system_prompt: str, messages: Sequence[tuple[str, str]]
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens as they are generated."""
        chat_messages = _build_messages(system_prompt=system_prompt, messages=messages)
        try:
            async for chunk in self._chat.astream(chat_messages):
                text = _content_to_text(chunk.content)
                if text:
                    yield text
        except Exception as exc:
            logger.error("llm_streaming_failed", error=str(exc))
            raise LLMServiceError("Failed to stream a response from the language model") from exc
