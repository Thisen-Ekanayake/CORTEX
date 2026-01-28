from TTS.api import TTS
from pathlib import Path


class TextToSpeech:
    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/vits",
        device: str = "cuda"
    ):
        """
        Text-to-Speech engine wrapper.

        Args:
            model_name: Coqui TTS model identifier
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device

        self.tts = TTS(model_name=self.model_name)
        self.tts.to(self.device)

    def speak_to_file(
        self,
        text: str,
        file_path: str | Path = "output.wav"
    ) -> str:
        """
        Convert text to speech and save to a WAV file.

        Args:
            text: Input text
            file_path: Output WAV path

        Returns:
            Path to generated audio file
        """
        file_path = Path(file_path)

        self.tts.tts_to_file(
            text=text,
            file_path=str(file_path)
        )

        return str(file_path)