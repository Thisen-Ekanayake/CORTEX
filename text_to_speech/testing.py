from text_to_speech import TextToSpeech

tts_engine = TextToSpeech(
    model_name="tts_models/en/ljspeech/vits",
    device="cuda"
)

audio_path = tts_engine.speak_to_file(
    text="Hello Thisen, your text to speech pipeline is alive.",
    file_path="logs/output.wav"
)

print("Audio saved to:", audio_path)