import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.services.rag_service import RAGService
from app.api.dependencies import get_rag_service
from app.core.logging import get_logger

router = APIRouter(tags=["WebSocket"])
logger = get_logger(__name__)


@router.websocket("/ws/query")
async def websocket_query(
    websocket: WebSocket,
    rag: RAGService = Depends(get_rag_service),
):
    """
    WebSocket endpoint for real-time queries.

    Client sends:  {"type": "query", "content": "your question"}
    Server sends:  {"type": "answer", "content": "...", "metadata": {...}}
    """
    await websocket.accept()
    client_id = websocket.client
    logger.info(f"WebSocket connected: {client_id}")

    try:
        while True:
            raw_data = await websocket.receive_text()

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid JSON format. Send: {\"type\": \"query\", \"content\": \"your question\"}",
                })
                continue

            msg_type = data.get("type", "")
            content = data.get("content", "").strip()

            if msg_type != "query" or not content:
                await websocket.send_json({
                    "type": "error",
                    "content": "Expected: {\"type\": \"query\", \"content\": \"...\"}",
                })
                continue

            logger.info(f"WS query from {client_id}: {content[:60]}")

            # Send a processing acknowledgment
            await websocket.send_json({
                "type": "processing",
                "content": "Retrieving relevant Sanskrit passages...",
            })

            try:
                response = rag.query(question=content)

                await websocket.send_json({
                    "type": "answer",
                    "content": response.answer,
                    "metadata": {
                        "query": response.query,
                        "contexts": [
                            {
                                "source": ctx.chunk.source_file,
                                "score": round(ctx.score, 4),
                                "preview": ctx.chunk.content[:200],
                            }
                            for ctx in response.retrieved_contexts
                        ],
                    },
                })

            except Exception as e:
                logger.error(f"WS query error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"Query failed: {str(e)}",
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
