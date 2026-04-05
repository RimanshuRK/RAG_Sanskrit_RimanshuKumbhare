from functools import lru_cache
from app.services.document_loader import DocumentLoader
from app.services.preprocessor import Preprocessor
from app.services.embedding_service import EmbeddingService
from app.services.faiss_store import FAISSStoreService
from app.services.retriever import RetrieverService
from app.services.generator import GeneratorService
from app.services.prompt_builder import PromptBuilder
from app.services.rag_service import RAGService
from app.repositories.vector_repository import VectorRepository


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """
    Manually wire all services using a clean DI pattern.
    `lru_cache` ensures that only one instance is created (singleton).
    """
    embedding_service = EmbeddingService()
    vector_repository = VectorRepository()

    faiss_store = FAISSStoreService(
        embedding_service=embedding_service,
        vector_repository=vector_repository,
    )
    retriever = RetrieverService(
        embedding_service=embedding_service,
        vector_repository=vector_repository,
    )

    # Inject PromptBuilder into GeneratorService
    prompt_builder = PromptBuilder()
    generator = GeneratorService(prompt_builder=prompt_builder)

    return RAGService(
        document_loader=DocumentLoader(),
        preprocessor=Preprocessor(),
        embedding_service=embedding_service,
        vector_repository=vector_repository,
        faiss_store=faiss_store,
        retriever=retriever,
        generator=generator,
    )
