from TTS.api import TTS

tts = TTS(
    model_name="tts_models/en/ljspeech/vits"
)

tts.to("cuda")

tts.tts_to_file(
    text="Hello Thisen, your text to speech pipeline is alive.",
    file_path="output.wav"
)