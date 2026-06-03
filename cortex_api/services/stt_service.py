"""Speech-to-text service backed by the local Parakeet (NeMo) model.

The model is loaded lazily on first use and cached as a module-level singleton
to avoid paying the (heavy) load cost at app startup.
"""

import io
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# Path to the Parakeet .nemo model (override with CORTEX_STT_MODEL).
STT_MODEL_PATH = os.getenv(
    "CORTEX_STT_MODEL",
    "models/Parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo",
)

_asr = None


def _get_asr():
    """Return the cached ParakeetASR instance, loading it on first call."""
    global _asr
    if _asr is None:
        import torch

        from speech_to_text.parakeet_asr import ParakeetASR

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Parakeet ASR on %s from %s", device, STT_MODEL_PATH)
        _asr = ParakeetASR(STT_MODEL_PATH, device=device)
    return _asr


def transcribe_bytes(audio_bytes: bytes) -> str:
    """Transcribe arbitrary audio bytes (any ffmpeg-decodable format) to text.

    The audio is normalized to 16 kHz mono WAV (what Parakeet expects) via
    ``pydub`` (requires a system ``ffmpeg``) before transcription.
    """
    from pydub import AudioSegment

    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    segment = segment.set_frame_rate(16000).set_channels(1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        segment.export(wav_path, format="wav")
        return _get_asr().transcribe(wav_path).strip()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
