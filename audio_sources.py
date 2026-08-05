#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio source abstraction layer.

Feeds 16-bit mono 16 kHz PCM chunks to an STT engine from either a file or a
live microphone, so both inputs travel the same code path.

File playback can be paced in real time, which is what makes file-based latency
numbers comparable to a live deployment: the STT then overlaps with the incoming
speech instead of consuming the whole recording up front. Without pacing, the
full STT duration sits on the critical path and no amount of STT/LLM overlap can
show a benefit.
"""

from abc import ABC, abstractmethod
import queue
import sys
import threading
import time
import wave
from typing import Iterator

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    sd = None
    _HAS_SD = False

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (16-bit PCM)
DEFAULT_CHUNK_MS = 250

PACING_REALTIME = "realtime"
PACING_FAST = "fast"


class AudioSource(ABC):
    """Base class for producers of 16-bit mono 16 kHz PCM chunks."""

    def __init__(self, chunk_ms: int = DEFAULT_CHUNK_MS, stop_event=None):
        self.chunk_ms = chunk_ms
        self.duration_ms = 0
        # perf_counter timestamp of the most recently delivered chunk
        self.last_chunk_t = None
        # A never-set event keeps the delivery loops uniform when no stop
        # signal was supplied by the caller.
        self._stop_event = stop_event if stop_event is not None else threading.Event()

    @property
    def frames_per_chunk(self) -> int:
        return int(SAMPLE_RATE * self.chunk_ms / 1000)

    @abstractmethod
    def chunks(self) -> Iterator[bytes]:
        """Yield raw PCM chunks until the audio ends or the source is stopped."""
        ...

    def speech_end_t(self):
        """Return a perf_counter estimate of when the speech ended.

        This is the anchor for the TTFA metric that matters to a user: latency is
        perceived from the moment they stop talking, not from the moment
        processing starts. The last delivered chunk is the best available proxy
        in both file and microphone mode.
        """
        return self.last_chunk_t


class FileAudioSource(AudioSource):
    """Reads a mono 16 kHz WAV file, optionally paced as if spoken live."""

    def __init__(self, wav_path: str, pacing: str = PACING_REALTIME,
                 chunk_ms: int = DEFAULT_CHUNK_MS, stop_event=None):
        """Initialize the file source.

        Args:
            wav_path: Path to a mono 16 kHz 16-bit PCM WAV file.
            pacing: PACING_REALTIME delivers chunks at 1x speed, like a
                microphone would. PACING_FAST delivers them as fast as they can
                be read, which reproduces the original batch behaviour and keeps
                older measurements comparable.
            chunk_ms: Chunk length in milliseconds.
            stop_event: threading.Event that ends delivery when set.

        Raises:
            ValueError: If the WAV file is not mono 16 kHz 16-bit PCM.
        """
        super().__init__(chunk_ms, stop_event)
        self._path = str(wav_path)
        self._pacing = pacing

        with wave.open(self._path, "rb") as wf:
            if (wf.getnchannels() != 1 or wf.getframerate() != SAMPLE_RATE
                    or wf.getsampwidth() != SAMPLE_WIDTH):
                raise ValueError(
                    f"{self._path}: expected mono {SAMPLE_RATE} Hz 16-bit PCM WAV, got "
                    f"{wf.getnchannels()}ch/{wf.getframerate()}Hz/"
                    f"{wf.getsampwidth() * 8}-bit"
                )
            self.duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)

    def chunks(self) -> Iterator[bytes]:
        with wave.open(self._path, "rb") as wf:
            start_t = time.perf_counter()
            frames_read = 0
            while not self._stop_event.is_set():
                data = wf.readframes(self.frames_per_chunk)
                if not data:
                    break
                frames_read += len(data) // SAMPLE_WIDTH
                if self._pacing == PACING_REALTIME:
                    # A real capture device only hands over a block once it has
                    # been recorded, so the deadline is the end of this chunk.
                    self._wait_until(start_t + frames_read / SAMPLE_RATE)
                self.last_chunk_t = time.perf_counter()
                yield data

    def _wait_until(self, deadline: float) -> None:
        """Sleep until the deadline, waking early if the source is stopped.

        Deadlines are absolute rather than incremental, so scheduling jitter
        does not accumulate into drift. If the consumer is slower than realtime
        the deadline is already in the past and no waiting happens at all, which
        matches how a live system falls behind.
        """
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            self._stop_event.wait(remaining)


class MicAudioSource(AudioSource):
    """Captures live microphone audio, optionally mirroring it to a WAV file."""

    def __init__(self, save_path: str = None,
                 chunk_ms: int = DEFAULT_CHUNK_MS, stop_event=None):
        """Initialize the microphone source.

        Args:
            save_path: Optional WAV path to record the captured audio to.
            chunk_ms: Chunk length in milliseconds.
            stop_event: threading.Event that ends the capture when set.

        Raises:
            ImportError: If sounddevice is not installed.
        """
        super().__init__(chunk_ms, stop_event)
        if not _HAS_SD:
            raise ImportError("sounddevice is required for microphone input")
        self._save_path = save_path

    def chunks(self) -> Iterator[bytes]:
        pending = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[Audio] Input status: {status}", file=sys.stderr)
            pending.put(bytes(indata))

        save_file = None
        try:
            if self._save_path:
                save_file = wave.open(self._save_path, "wb")
                save_file.setnchannels(1)
                save_file.setsampwidth(SAMPLE_WIDTH)
                save_file.setframerate(SAMPLE_RATE)

            with sd.RawInputStream(samplerate=SAMPLE_RATE,
                                   blocksize=self.frames_per_chunk,
                                   device=None, dtype="int16",
                                   channels=1, callback=callback):
                print("[Audio] Listening on microphone (speak now, pause to trigger LLM)...")
                while not self._stop_event.is_set():
                    try:
                        data = pending.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if save_file:
                        save_file.writeframes(data)
                    self.duration_ms += int(
                        len(data) / SAMPLE_WIDTH / SAMPLE_RATE * 1000
                    )
                    self.last_chunk_t = time.perf_counter()
                    yield data
        finally:
            if save_file:
                save_file.close()
