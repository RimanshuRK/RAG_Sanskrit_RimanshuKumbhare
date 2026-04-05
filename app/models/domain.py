from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocumentChunk:
    """A single chunk from a document stored in FAISS."""
    chunk_id: str
    source_file: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedContext:
    """A relevant chunk returned by the retriever with its similarity score."""
    chunk: DocumentChunk
    score: float


@dataclass
class RAGResponse:
    """The final response returned to the user."""
    answer: str
    retrieved_contexts: List[RetrievedContext]
    query: str
    model_used: str = ""
