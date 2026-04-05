from typing import List
from app.models.domain import DocumentChunk, RetrievedContext
from app.services.embedding_service import EmbeddingService
from app.repositories.vector_repository import VectorRepository
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RetrieverService:
    """
    Semantic search: query -> relevant DocumentChunks.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        top_k: int = settings.top_k_results,
    ):
        self._embedding = embedding_service
        self._repository = vector_repository
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedContext]:
        """
        Retrieve relevant contexts for a query.
        Returns: List of RetrievedContext sorted by relevance.
        """
        k = top_k or self.top_k
        logger.info(f"Retrieving top-{k} chunks for query: {query[:60]}...")

        query_vector = self._embedding.embed_query(query)
        raw_results = self._repository.search(query_vector, top_k=k)

        contexts = []
        for meta, score in raw_results:
            chunk = DocumentChunk(
                chunk_id=meta["chunk_id"],
                source_file=meta["source"],
                content=meta["content"],
                metadata=meta.get("metadata", {}),
            )
            contexts.append(RetrievedContext(chunk=chunk, score=score))

        logger.info(f"Retrieved {len(contexts)} contexts.")
        return contexts
