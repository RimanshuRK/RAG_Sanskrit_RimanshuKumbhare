from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Sanskrit ya English mein sawal")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Kitne chunks retrieve karne hain")


class ContextItem(BaseModel):
    source: str
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    query: str
    contexts: List[ContextItem]
    status: str = "success"


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str = "success"
    message: str


class WebSocketMessage(BaseModel):
    type: str          # "query" | "error" | "answer" | "stream"
    content: str
    metadata: dict = {}