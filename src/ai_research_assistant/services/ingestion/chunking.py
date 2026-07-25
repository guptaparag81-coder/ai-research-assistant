"""Text chunking for embedding generation."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_research_assistant.core.config import Settings


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    content: str
    token_count: int


class DocumentChunker:
    """Splits extracted document text into overlapping chunks."""

    def __init__(self, settings: Settings) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str) -> list[TextChunk]:
        parts = self._splitter.split_text(text)
        return [
            TextChunk(index=index, content=part, token_count=_approximate_token_count(part))
            for index, part in enumerate(parts)
        ]


def _approximate_token_count(text: str) -> int:
    """Cheap token estimate (~4 chars/token) avoiding a tokenizer dependency."""
    return max(1, len(text) // 4)
