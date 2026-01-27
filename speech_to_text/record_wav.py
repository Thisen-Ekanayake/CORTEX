import argparse
import sys
import wave

import numpy as np
import sounddevice as sd


def record_wav(duration_s: float, out_path: str, sample_rate: int, channels: int) -> None:
    if duration_s <= 0:
        raise ValueError("Duration must be > 0 seconds.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    if channels not in (1, 2):
        raise ValueError("Channels must be 1 (mono) or 2 (stereo).")

    frames = int(round(duration_s * sample_rate))

    print(f"Recording {duration_s:.2f}s @ {sample_rate} Hz, {channels} channel(s)...")
    audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="int16")
    sd.wait()
    print("Recording finished. Saving...")

    # audio is shape (frames, channels) int16
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seconds", type=float, help="Recording duration in seconds (e.g., 5 or 2.5)")
    parser.add_argument("--out", default="recording.wav", help="Output WAV filename")
    parser.add_argument("--rate", type=int, default=44100, help="Sample rate (Hz), default 44100")
    parser.add_argument("--channels", type=int, default=1, help="1=mono, 2=stereo (default 1)")
    args = parser.parse_args()

    try:
        record_wav(args.seconds, args.out, args.rate, args.channels)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()