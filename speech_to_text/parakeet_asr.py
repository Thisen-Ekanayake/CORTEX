import nemo.collections.asr as nemo_asr


class ParakeetASR:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"🔊 Loading Parakeet model from {model_path}")
        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=model_path,
            map_location=device,
        )
        self.model.eval()

    def transcribe(self, wav_path: str) -> str:
        output = self.model.transcribe([wav_path])
        return output[0].text
