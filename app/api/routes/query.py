from fastapi import APIRouter, Depends, HTTPException
from app.schemas.request_response import QueryRequest, QueryResponse, ContextItem
from app.services.rag_service import RAGService
from app.api.dependencies import get_rag_service
from app.core.logging import get_logger

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(__name__)


@router.post("/", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """
    Retrieve a relevant answer from Sanskrit documents.

    - **question**: The user's question in Sanskrit or English
    - **top_k**: Optional number of chunks to consider
    """
    try:
        logger.info(f"REST query received: {request.question[:60]}")
        response = rag.query(
            question=request.question,
            top_k=request.top_k,
        )

        contexts = [
            ContextItem(
                source=ctx.chunk.source_file,
                content=ctx.chunk.content[:300],  # Preview truncated
                score=round(ctx.score, 4),
            )
            for ctx in response.retrieved_contexts
        ]

        return QueryResponse(
            answer=response.answer,
            query=response.query,
            contexts=contexts,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
