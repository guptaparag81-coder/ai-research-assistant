"""Assembles retrieved chunks into a single citation-annotated context block."""

from ai_research_assistant.services.retrieval.retriever import RetrievedChunk


class ContextBuilder:
    """Builds a formatted context string from retrieved chunks for prompt injection."""

    def build(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context was found in the knowledge base."

        sections: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.document_title or "Untitled document"
            sections.append(
                f"[Source {i}] {title} (chunk {chunk.chunk_index}, relevance={chunk.score:.2f})\n"
                f"{chunk.content.strip()}"
            )
        return "\n\n---\n\n".join(sections)
