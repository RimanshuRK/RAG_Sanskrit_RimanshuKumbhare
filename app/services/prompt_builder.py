from typing import List
from app.models.domain import RetrievedContext


class PromptBuilder:
    """
    Builds the system prompt and user prompt for RAG.

    OpenAI messages format:
    [
        {"role": "system", "content": <system_prompt>},
        {"role": "user",   "content": <user_prompt>}
    ]
    """

    # =========================================================
    # SYSTEM PROMPT - Defines the LLM's behavior
    # This is set once and stays the same for every query
    # =========================================================
    SYSTEM_PROMPT = """You are an expert Sanskrit scholar and AI assistant specializing in ancient Indian texts, philosophy, and literature.

Your responsibilities:
1. Answer questions based STRICTLY on the provided Sanskrit document excerpts.
2. User may ask in Hindi, English, or Hinglish - answer in the same language they asked.
3. If the answer is not found in the provided context, clearly say: "The provided Sanskrit documents do not contain information about this topic."
4. When quoting Sanskrit terms or shlokas, preserve them accurately.
5. Provide clear, educational answers suitable for both scholars and general readers.
6. Always cite which document/source the information comes from.
7. Do NOT hallucinate or make up information not present in the context.

Response format:
- Give a direct answer first.
- Then provide relevant Sanskrit references if available.
- Mention the source document name.
- Keep response concise but complete."""

    # =========================================================
    # USER PROMPT TEMPLATE - Changes for each query
    # Retrieved context + the user's actual question go here
    # =========================================================
    USER_PROMPT_TEMPLATE = """Based on the following excerpts from Sanskrit documents, please answer the question.

=== RETRIEVED CONTEXT FROM SANSKRIT DOCUMENTS ===
{context}
=== END OF CONTEXT ===

Question: {question}

Please answer based only on the context provided above."""

    def build_messages(
        self,
        query: str,
        contexts: List[RetrievedContext],
    ) -> List[dict]:
        """
        Build the messages array for the OpenAI API.

        Returns:
        [
            {"role": "system", "content": "...expert Sanskrit scholar..."},
            {"role": "user",   "content": "...context + question..."}
        ]
        """
        context_text = self._format_context(contexts)
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context_text,
            question=query,
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

    def _format_context(self, contexts: List[RetrievedContext]) -> str:
        """
        Convert retrieved chunks into a readable format.

        Output example:
        [Excerpt 1 | Source: ramayana.pdf | Relevance: 0.92]
        "Dharmo rakshati rakshitah..."

        [Excerpt 2 | Source: gita.pdf | Relevance: 0.87]
        "Karmanye vadhikaraste..."
        """
        if not contexts:
            return "No relevant context found in the documents."

        parts = []
        for i, ctx in enumerate(contexts, 1):
            parts.append(
                f"[Excerpt {i} | Source: {ctx.chunk.source_file} "
                f"| Relevance: {round(ctx.score, 2)}]\n"
                f"{ctx.chunk.content}"
            )

        return "\n\n---\n\n".join(parts)

    def build_no_context_messages(self, query: str) -> List[dict]:
        """
        Use this when no document is indexed in FAISS.
        """
        user_prompt = (
            f"The user asked: {query}\n\n"
            "Note: No Sanskrit documents are currently indexed in the system. "
            "Please inform the user to upload documents first."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
