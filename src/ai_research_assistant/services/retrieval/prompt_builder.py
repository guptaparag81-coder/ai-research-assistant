"""Builds the final system prompt for RAG-grounded chat completions."""

_SYSTEM_TEMPLATE = """You are an AI research assistant that answers questions strictly using \
the provided context retrieved from the user's document collection.

Rules:
- Answer only using information present in the context below.
- If the context does not contain the answer, say so explicitly instead of guessing.
- Cite sources inline using their bracketed labels, e.g. [Source 1].
- Be concise, accurate, and avoid speculation.

Context:
{context}
"""


class PromptBuilder:
    """Composes the system prompt combining retrieved context and instructions."""

    def build_system_prompt(self, context: str) -> str:
        return _SYSTEM_TEMPLATE.format(context=context)
