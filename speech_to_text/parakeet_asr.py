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
        self._disable_cuda_graph_decoder()

    def _disable_cuda_graph_decoder(self) -> None:
        """Disable the TDT CUDA-graph greedy decoder.

        On some NeMo/PyTorch combinations the CUDA-graph decoding path fails to
        compile (``ValueError: not enough values to unpack``). Falling back to
        the standard greedy loop avoids the crash with negligible cost.
        """
        try:
            from omegaconf import open_dict

            cfg = self.model.cfg.decoding
            with open_dict(cfg):
                if "greedy" in cfg:
                    cfg.greedy.use_cuda_graph_decoder = False
            self.model.change_decoding_strategy(cfg)
        except Exception as exc:  # noqa: BLE001 — best-effort; keep default decoder
            print(f"⚠️  Could not disable CUDA-graph decoder: {exc}")

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
