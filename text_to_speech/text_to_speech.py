from pathlib import Path
import wave
from typing import Union

import numpy as np
import sounddevice as sd
from TTS.api import TTS


class TextToSpeech:
    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/vits",
        device: str = "cuda",
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
        file_path: Union[str, Path] = "output.wav",
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
            file_path=str(file_path),
        )

        return str(file_path)

    def play_audio_file(self, file_path: Union[str, Path]) -> None:
        """
        Play a WAV audio file using the local sound device.

        Args:
            file_path: Path to WAV file.
        """
        file_path = Path(file_path)

        with wave.open(str(file_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            # Only standard 16‑bit PCM is supported here
            if sampwidth != 2:
                raise ValueError(
                    f"Unsupported sample width {sampwidth * 8}‑bit; "
                    "expected 16‑bit PCM."
                )

            frames = wf.readframes(n_frames)
            audio = np.frombuffer(frames, dtype=np.int16)

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)

            sd.play(audio, framerate)
            sd.wait()

    def speak_and_play(
        self,
        text: str,
        file_path: Union[str, Path] = "output.wav",
    ) -> str:
        """
        Convenience method to synthesize text to a file and play it.

        Args:
            text: Input text to synthesize.
            file_path: Output WAV path.

        Returns:
            Path to generated audio file.
        """
        audio_path = self.speak_to_file(text=text, file_path=file_path)
        self.play_audio_file(audio_path)
        return audio_path