from typing import List, Tuple
from app.models.domain import RAGResponse, RetrievedContext
from app.services.document_loader import DocumentLoader
from app.services.preprocessor import Preprocessor
from app.services.embedding_service import EmbeddingService
from app.services.faiss_store import FAISSStoreService
from app.services.retriever import RetrieverService
from app.services.generator import GeneratorService
from app.repositories.vector_repository import VectorRepository
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RAGService:
    """
    Main orchestrator - combines Retriever and Generator.
    This class works with injected dependencies (Clean Architecture).
    """

    def __init__(
        self,
        document_loader: DocumentLoader,
        preprocessor: Preprocessor,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        faiss_store: FAISSStoreService,
        retriever: RetrieverService,
        generator: GeneratorService,
    ):
        self._loader = document_loader
        self._preprocessor = preprocessor
        self._embedding = embedding_service
        self._repository = vector_repository
        self._faiss_store = faiss_store
        self._retriever = retriever
        self._generator = generator

    def startup(self):
        """
        On application startup:
        1. Load the embedding model
        2. Initialize the FAISS repository
        3. Load the LLM
        4. If FAISS is empty, index documents from the `data/` folder
        """
        logger.info("=== RAGService Startup ===")

        self._embedding.initialize()
        self._repository.initialize(dimension=self._embedding.dimension)
        self._generator.initialize()

        if self._faiss_store.is_ready():
            logger.info(f"FAISS already has {self._faiss_store.total_indexed()} vectors. Skipping re-indexing.")
        else:
            logger.info("FAISS empty. Loading documents from data/ folder...")
            self._index_data_folder()

        logger.info("=== RAGService Ready ===")

    def _index_data_folder(self):
        """Index documents from the default `data/` folder."""
        documents = self._loader.load_directory(settings.data_folder)
        if not documents:
            logger.warning("data/ folder mein koi document nahi mila.")
            return

        total = 0
        for filename, raw_text in documents:
            chunks = self._preprocessor.process(filename, raw_text)
            indexed = self._faiss_store.index_chunks(chunks)
            total += indexed

        logger.info(f"Total chunks indexed from data/: {total}")

    def index_document(self, filename: str, content: bytes) -> int:
        """
        Index a user-uploaded document.
        Returns: chunks indexed.
        """
        logger.info(f"Indexing user upload: {filename}")
        _, raw_text = self._loader.load_bytes(filename, content)
        chunks = self._preprocessor.process(filename, raw_text)
        return self._faiss_store.index_chunks(chunks)

    def query(self, question: str, top_k: int = None) -> RAGResponse:
        """
        Handle a user query:
        1. Retrieve relevant chunks
        2. Generate an answer using the LLM
        3. Return a `RAGResponse`
        """
        if not self._faiss_store.is_ready():
            return RAGResponse(
                answer="Index abhi empty hai. Pehle documents upload ya load karo.",
                retrieved_contexts=[],
                query=question,
            )

        contexts: List[RetrievedContext] = self._retriever.retrieve(
            query=question,
            top_k=top_k or settings.top_k_results,
        )
        answer = self._generator.generate(query=question, contexts=contexts)

        return RAGResponse(
            answer=answer,
            retrieved_contexts=contexts,
            query=question,
            model_used=settings.llm_model_path,
        )
