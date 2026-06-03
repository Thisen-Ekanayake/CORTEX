"""Text-to-speech service backed by the local Coqui TTS model.

The engine is loaded lazily on first use and cached as a module-level
singleton. Synthesis writes to a temporary WAV file and returns its bytes.
"""

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# Coqui model identifier (override with CORTEX_TTS_MODEL).
TTS_MODEL_NAME = os.getenv("CORTEX_TTS_MODEL", "tts_models/en/ljspeech/vits")

_tts = None


def _get_tts():
    """Return the cached TextToSpeech engine, loading it on first call."""
    global _tts
    if _tts is None:
        import torch

        from text_to_speech.text_to_speech import TextToSpeech

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Coqui TTS (%s) on %s", TTS_MODEL_NAME, device)
        _tts = TextToSpeech(model_name=TTS_MODEL_NAME, device=device)
    return _tts


def synthesize(text: str) -> bytes:
    """Synthesize ``text`` to speech and return WAV bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        _get_tts().speak_to_file(text=text, file_path=wav_path)
        with open(wav_path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
