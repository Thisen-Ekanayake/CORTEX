from audio_recorder import AudioRecorder
from parakeet_asr import ParakeetASR


def main():
    """
    Main function: record audio and transcribe it.
    
    Interactive workflow that:
    1. Initializes audio recorder and ASR model
    2. Records audio from user input
    3. Transcribes the recorded audio
    4. Prints the transcription
    """
    recorder = AudioRecorder(sample_rate=44100, channels=1)
    asr = ParakeetASR(
        model_path="models/Parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo",
        device="cuda",
    )

    wav_file = recorder.run()
    print("🧠 Transcribing...")

    text = asr.transcribe(wav_file)
    print("\n📝 Transcription:")
    print(text)


if __name__ == "__main__":
    main()