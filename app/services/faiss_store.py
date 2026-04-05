from typing import List
from app.models.domain import DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.repositories.vector_repository import VectorRepository
from app.core.logging import get_logger

logger = get_logger(__name__)


class FAISSStoreService:
    """
    Embeds DocumentChunks and stores them in FAISS.
    Composes EmbeddingService and VectorRepository.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
    ):
        self._embedding = embedding_service
        self._repository = vector_repository

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Take a list of DocumentChunks and index them.
        Returns: number of chunks indexed.
        """
        if not chunks:
            logger.warning("Koi chunk nahi mila indexing ke liye.")
            return 0

        texts = [chunk.content for chunk in chunks]
        vectors = self._embedding.embed_texts(texts)

        metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source_file,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        self._repository.add_vectors(vectors, metadata)
        logger.info(f"Indexed {len(chunks)} chunks successfully.")
        return len(chunks)

    def is_ready(self) -> bool:
        return not self._repository.is_empty()

    def total_indexed(self) -> int:
        return self._repository.total_vectors()
