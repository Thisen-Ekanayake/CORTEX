"""Voice routes — speech-to-text (Parakeet) and text-to-speech (Coqui)."""

import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["Voice"])


class TTSRequest(BaseModel):
    text: str


@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...)) -> dict:
    """Transcribe an uploaded audio clip to text via Parakeet."""
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    from cortex_api.services.stt_service import transcribe_bytes

    try:
        text = transcribe_bytes(audio)
    except Exception as exc:  # noqa: BLE001 — surface decode/transcribe failures
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from None

    return {"text": text}


@router.post("/tts")
def text_to_speech(req: TTSRequest) -> StreamingResponse:
    """Synthesize speech from text via Coqui; returns a WAV stream."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    from cortex_api.services.tts_service import synthesize

    try:
        wav_bytes = synthesize(text)
    except Exception as exc:  # noqa: BLE001 — surface synthesis failures
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from None

    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
