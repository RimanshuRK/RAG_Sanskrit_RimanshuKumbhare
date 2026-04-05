import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Converts text to vectors using sentence-transformers.
    Uses a multilingual model for Sanskrit.
    Singleton pattern: loaded only once.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self._model = SentenceTransformer(
            settings.embedding_model,
            device="cpu",
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
        self._initialized = True

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into a NumPy array of vectors.
        Shape: (len(texts), dimension)
        """
        if not self._initialized:
            self.initialize()

        logger.debug(f"Embedding {len(texts)} texts...")
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        Shape: (1, dimension)
        """
        return self.embed_texts([query])
