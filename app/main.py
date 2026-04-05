from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import query, upload, websocket
from app.api.dependencies import get_rag_service
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}...")
    rag = get_rag_service()
    rag.startup()
    logger.info("System ready. Awaiting queries.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Sanskrit RAG System",
    description="Retrieve and generate answers from Sanskrit documents using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(upload.router)
app.include_router(websocket.router)


@app.get("/health", tags=["Health"])
async def health_check():
    rag = get_rag_service()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "total_indexed": rag._faiss_store.total_indexed(),
        "faiss_ready": rag._faiss_store.is_ready(),
    }