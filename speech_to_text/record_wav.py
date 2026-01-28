"""
Interactive audio recorder.
r + Enter → start recording
p + Enter → pause
k + Enter → resume
s + Enter → stop & save (timestamped WAV)
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
paused = False


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)

    if recording and not paused:
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
    global recording, paused

    print("🎧 Audio Recorder")
    print("r → start | p → pause | k → resume | s → stop & save")
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
                print("🔴 Recording started")
                recorded_chunks.clear()
                while not audio_queue.empty():
                    audio_queue.get()
                recording = True
                paused = False

            elif cmd == "p" and recording and not paused:
                paused = True
                print("⏸️ Recording paused")

            elif cmd == "k" and recording and paused:
                paused = False
                print("▶️ Recording resumed")

            elif cmd == "s" and recording:
                print("🛑 Recording stopped. Saving...")
                recording = False
                paused = False

                while not audio_queue.empty():
                    recorded_chunks.append(audio_queue.get())

                if not recorded_chunks:
                    print("⚠️ No audio captured\n")
                    continue

                filename = timestamp_filename()
                save_wav(filename, recorded_chunks, SAMPLE_RATE, CHANNELS)
                print(f"✅ Saved as {filename}\n")

            else:
                print("ℹ️ r=start | p=pause | k=resume | s=stop")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
