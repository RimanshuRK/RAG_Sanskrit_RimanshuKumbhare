from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # App
    app_name: str = "SanskritRAG"

    # Data & Index
    data_folder: str = "./app/data"
    faiss_index_path: str = "./faiss_index"

    # Embedding
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k_results: int = 5

    # Logging
    log_level: str = "INFO"

    # ✅ OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 1024
    openai_temperature: float = 0.3

    # ✅ Backward compat
    llm_model_path: str = "" 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        Path(self.data_folder).mkdir(parents=True, exist_ok=True)
        Path(self.faiss_index_path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()