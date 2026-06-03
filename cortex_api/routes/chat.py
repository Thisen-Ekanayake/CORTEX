"""Chat route — streamed RAG/CHAT/META answers over Server-Sent Events."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cortex_api.services.chat_service import stream_chat

router = APIRouter(prefix="/api", tags=["Chat"])


class ChatRequest(BaseModel):
    query: str


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Stream an answer for ``req.query`` as SSE token/done events."""
    return StreamingResponse(
        stream_chat(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for live streaming
        },
    )
