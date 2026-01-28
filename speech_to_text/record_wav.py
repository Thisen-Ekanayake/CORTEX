#!/usr/bin/env python3
"""
Interactive audio recorder.
- Press 'r' + Enter to start recording
- Press 's' + Enter to stop and save as WAV (timestamped filename)
"""

import sounddevice as sd
import numpy as np
import wave
import queue
import sys
from datetime import datetime

SAMPLE_RATE = 44100
CHANNELS = 1

audio_queue = queue.Queue()
recording = False


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    if recording:
        audio_queue.put(indata.copy())


def save_wav(filename, audio_data, samplerate, channels):
    audio_np = np.concatenate(audio_data, axis=0)
    audio_int16 = np.int16(audio_np * 32767)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())


def timestamp_filename():
    return datetime.now().strftime("%Y%m%d_%H%M%S.wav")


def main():
    global recording

    print("🎧 Audio Recorder")
    print("r + Enter → start recording")
    print("s + Enter → stop & save")
    print("Ctrl+C → exit\n")

    recorded_chunks = []

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
    ):
        while True:
            cmd = input("> ").strip().lower()

            if cmd == "r" and not recording:
                print("🔴 Recording started...")
                recorded_chunks.clear()
                while not audio_queue.empty():
                    audio_queue.get()
                recording = True

            elif cmd == "s" and recording:
                print("🛑 Recording stopped. Saving...")
                recording = False

                while not audio_queue.empty():
                    recorded_chunks.append(audio_queue.get())

                filename = timestamp_filename()
                save_wav(
                    filename,
                    recorded_chunks,
                    SAMPLE_RATE,
                    CHANNELS,
                )
                print(f"✅ Saved as {filename}\n")

            else:
                print("ℹ️ Press 'r' to record, 's' to stop")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting.")