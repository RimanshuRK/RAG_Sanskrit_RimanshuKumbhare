import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorRepository:
    """
    Abstraction layer for the FAISS vector store.
    - Saves and loads the index from disk
    - Stores metadata (chunk_ids, texts) separately in a pickle file
    """

    INDEX_FILE = "index.faiss"
    META_FILE = "metadata.pkl"

    def __init__(self, index_path: str = settings.faiss_index_path):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index: Optional[faiss.Index] = None
        self._metadata: List[dict] = []  # chunk_id, source, content per vector
        self._is_loaded = False

    def initialize(self, dimension: int):
        """
        Create a new FAISS index or load an existing one.
        `IndexFlatIP` uses inner product (normalized vectors = cosine similarity).
        """
        index_file = self.index_path / self.INDEX_FILE
        meta_file = self.index_path / self.META_FILE

        if index_file.exists() and meta_file.exists():
            logger.info("Existing FAISS index found. Loading...")
            self._index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                self._metadata = pickle.load(f)
            logger.info(f"Index loaded. Total vectors: {self._index.ntotal}")
        else:
            logger.info(f"Creating new FAISS index. Dimension: {dimension}")
            self._index = faiss.IndexFlatIP(dimension)
            self._metadata = []

        self._is_loaded = True

    def add_vectors(self, vectors: np.ndarray, metadata: List[dict]):
        """
        Add new vectors and their metadata to the index.
        vectors shape: (n, dimension)
        metadata: list of dicts with chunk_id, source, content
        """
        if not self._is_loaded:
            raise RuntimeError("VectorRepository not initialized. Call initialize() first.")

        if len(vectors) != len(metadata):
            raise ValueError("vectors aur metadata ki count match nahi karti.")

        self._index.add(vectors)
        self._metadata.extend(metadata)
        self._persist()
        logger.info(f"Added {len(vectors)} vectors. Total: {self._index.ntotal}")

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[dict, float]]:
        """
        Find the top-k most similar vectors for a query vector.
        Returns: list of (metadata_dict, score) tuples
        """
        if not self._is_loaded or self._index.ntotal == 0:
            logger.warning("FAISS index empty ya loaded nahi hai.")
            return []

        actual_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                results.append((self._metadata[idx], float(score)))

        return results

    def is_empty(self) -> bool:
        return not self._is_loaded or self._index is None or self._index.ntotal == 0

    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def _persist(self):
        """Save the index and metadata to disk."""
        faiss.write_index(self._index, str(self.index_path / self.INDEX_FILE))
        with open(self.index_path / self.META_FILE, "wb") as f:
            pickle.dump(self._metadata, f)
        logger.debug("FAISS index persisted.")
