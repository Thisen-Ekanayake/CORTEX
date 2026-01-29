import sounddevice as sd
import numpy as np
import wave
import queue
import sys
from datetime import datetime


class AudioRecorder:
    """
    Audio recorder with pause/resume functionality.
    
    Records audio from the default input device and saves it as WAV files.
    Supports interactive control: start, pause, resume, and stop recording.
    """
    
    def __init__(self, sample_rate=44100, channels=1):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 44100).
            channels: Number of audio channels, 1=mono, 2=stereo (default: 1).
        """
        self.sample_rate = sample_rate
        self.channels = channels

        self.audio_queue = queue.Queue()
        self.recording = False
        self.paused = False

    # ---------- Audio ----------

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for audio input stream.
        
        Called by sounddevice for each audio buffer. Adds audio data to
        the queue if recording is active and not paused.
        
        Args:
            indata: Input audio data array.
            frames: Number of frames in the buffer.
            time_info: Timing information dict.
            status: Status flags (errors printed to stderr if present).
        """
        if status:
            print(status, file=sys.stderr)

        if self.recording and not self.paused:
            self.audio_queue.put(indata.copy())

    def _timestamp_filename(self):
        """
        Generate a timestamped WAV filename.
        
        Returns:
            str: Filename in format YYYYMMDD_HHMMSS.wav
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S.wav")

    def _save_wav(self, filename, chunks):
        """
        Save audio chunks to a WAV file.
        
        Concatenates audio chunks, converts to 16-bit PCM, and writes
        to a WAV file with the configured sample rate and channels.
        
        Args:
            filename: Output WAV file path.
            chunks: List of numpy arrays containing audio data.
        """
        audio = np.concatenate(chunks, axis=0)
        audio = np.int16(audio * 32767)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

    # ---------- Public API ----------

    def run(self):
        """
        Run interactive audio recording session.
        
        Starts an interactive loop that accepts commands:
        - 'r': Start recording
        - 'p': Pause recording
        - 'k': Resume recording
        - 's': Stop recording and save to WAV file
        
        Returns:
            str: Path to saved WAV file when recording is stopped.
        """
        print("🎧 Audio Recorder")
        print("r → start | p → pause | k → resume | s → stop & save")
        print("Ctrl+C → exit\n")

        recorded_chunks = []

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
        ):
            while True:
                cmd = input("> ").strip().lower()

                if cmd == "r" and not self.recording:
                    print("🔴 Recording started")
                    recorded_chunks.clear()
                    while not self.audio_queue.empty():
                        self.audio_queue.get()
                    self.recording = True
                    self.paused = False

                elif cmd == "p" and self.recording and not self.paused:
                    self.paused = True
                    print("⏸️ Recording paused")

                elif cmd == "k" and self.recording and self.paused:
                    self.paused = False
                    print("▶️ Recording resumed")

                elif cmd == "s" and self.recording:
                    print("🛑 Recording stopped. Saving...")
                    self.recording = False
                    self.paused = False

                    while not self.audio_queue.empty():
                        recorded_chunks.append(self.audio_queue.get())

                    if not recorded_chunks:
                        print("⚠️ No audio captured\n")
                        continue

                    filename = self._timestamp_filename()
                    self._save_wav(filename, recorded_chunks)
                    print(f"✅ Saved as {filename}\n")

                    return filename  # IMPORTANT: return saved file

                else:
                    print("ℹ️ r=start | p=pause | k=resume | s=stop")
