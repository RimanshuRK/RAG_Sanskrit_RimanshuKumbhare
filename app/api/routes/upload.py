from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.schemas.request_response import UploadResponse
from app.services.rag_service import RAGService
from app.api.dependencies import get_rag_service
from app.core.logging import get_logger

router = APIRouter(prefix="/upload", tags=["Upload"])
logger = get_logger(__name__)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag: RAGService = Depends(get_rag_service),
):
    """
    Upload a Sanskrit document and index it automatically.

    - Supported formats: PDF, TXT
    - Max size: 20MB
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing.")

    extension = file.filename.lower().split(".")[-1]
    if extension not in ("pdf", "txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF and TXT supported. Got: .{extension}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 20MB allowed.")

    try:
        logger.info(f"Processing upload: {file.filename} ({len(content)} bytes)")
        chunks_indexed = rag.index_document(
            filename=file.filename,
            content=content,
        )

        return UploadResponse(
            filename=file.filename,
            chunks_indexed=chunks_indexed,
            message=f"Document '{file.filename}' successfully indexed with {chunks_indexed} chunks.",
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
