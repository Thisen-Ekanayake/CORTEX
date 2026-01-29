import nemo.collections.asr as nemo_asr


class ParakeetASR:
    """
    Automatic Speech Recognition using NeMo Parakeet models.
    
    Wrapper around NeMo ASR models for transcribing audio files to text.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the ASR model.
        
        Args:
            model_path: Path to the .nemo model file.
            device: Device to load model on, "cuda" or "cpu" (default: "cuda").
        """
        print(f"🔊 Loading Parakeet model from {model_path}")
        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=model_path,
            map_location=device,
        )
        self.model.eval()

    def transcribe(self, wav_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            wav_path: Path to WAV audio file to transcribe.
        
        Returns:
            str: Transcribed text from the audio.
        """
        output = self.model.transcribe([wav_path])
        return output[0].text
