#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STT engine abstraction layer.
Provides BaseSTTEngine interface and concrete implementations for Vosk and faster-whisper.
"""

from abc import ABC, abstractmethod
import json
import queue
import sys
import wave
from typing import Iterator

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    sd = None
    _HAS_SD = False
from vosk import KaldiRecognizer, Model

try:
    from faster_whisper import WhisperModel
    _HAS_WHISPER = True
except ImportError:
    WhisperModel = None
    _HAS_WHISPER = False


class BaseSTTEngine(ABC):
    """Abstract base class for speech-to-text engines."""

    @abstractmethod
    def transcribe(self, wav_path: str) -> str:
        """Transcribe a WAV file and return the full recognized text.

        Args:
            wav_path: Path to a mono 16kHz 16-bit PCM WAV file.

        Returns:
            The full transcription as a single string.
        """
        ...

    @abstractmethod
    def transcribe_stream(self, wav_path: str) -> Iterator[dict]:
        """Transcribe a WAV file and yield results incrementally.

        Yields dicts with keys:
            - "is_final" (bool): True if this is a finalized recognition result,
                                  False for partial/intermediate results.
            - "text" (str): The recognized text.

        Args:
            wav_path: Path to a mono 16kHz 16-bit PCM WAV file.
        """
        ...


class VoskEngine(BaseSTTEngine):
    """Vosk-based STT engine with streaming partial/final results."""

    def __init__(self, model_path: str):
        self._model = Model(model_path)
        self._sample_rate = 16000
        self._chunk_size = 4000  # frames per read

    def transcribe(self, wav_path: str) -> str:
        """Transcribe by collecting all final results from the streaming interface."""
        final_texts = []
        for result in self.transcribe_stream(wav_path):
            if result["is_final"]:
                final_texts.append(result["text"])
        return " ".join(final_texts).strip()

    def transcribe_stream(self, wav_path: str) -> Iterator[dict]:
        """Yield partial and final recognition results as audio chunks are processed.

        A fresh KaldiRecognizer is created per call, so no explicit reset is needed
        between files.
        """


        rec = KaldiRecognizer(self._model, self._sample_rate)
        rec.SetWords(True)

        if wav_path.startswith("mic"):
            
            q = queue.Queue()
            save_path = wav_path.split(":", 1)[1] if ":" in wav_path else None
            mic_wav_file = None
            if save_path:
                mic_wav_file = wave.open(save_path, "wb")
                mic_wav_file.setnchannels(1)
                mic_wav_file.setsampwidth(2)
                mic_wav_file.setframerate(self._sample_rate)

            def callback(indata, frames, time_info, status):
                if status:
                    print(f"[STT] Audio Callback Status: {status}", file=sys.stderr)
                q.put(bytes(indata))

            try:
                if not _HAS_SD:
                    raise ImportError("sounddevice module not installed")
                with sd.RawInputStream(samplerate=self._sample_rate, blocksize=8000, 
                                       device=None, dtype='int16',
                                       channels=1, callback=callback):
                    print("[STT] Listening on microphone (speak now, pause to trigger LLM)...")
                    while True:
                        if getattr(sys.modules['__main__'], 'shutdown_event', False):
                            break
                            
                        try:
                            data = q.get(timeout=0.5)
                        except queue.Empty:
                            continue
                            
                        if mic_wav_file:
                            mic_wav_file.writeframes(data)
                            
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            text = result.get("text", "")
                            if text:
                                yield {"is_final": True, "text": text}
                        else:
                            result = json.loads(rec.PartialResult())
                            text = result.get("partial", "")
                            if text:
                                yield {"is_final": False, "text": text}
            except KeyboardInterrupt:
                print("\n[STT] Microphone listening stopped by user.")
                result = json.loads(rec.FinalResult())
                text = result.get("text", "")
                if text:
                    yield {"is_final": True, "text": text}
            finally:
                if mic_wav_file:
                    mic_wav_file.close()
            return    
        else:
            wf = wave.open(wav_path, "rb")
            assert (wf.getnchannels() == 1
                    and wf.getframerate() == self._sample_rate
                    and wf.getsampwidth() == 2), \
                "Vosk requires mono 16kHz 16-bit PCM WAV input"

            try:
                while True:
                    data = wf.readframes(self._chunk_size)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "")
                        if text:
                            yield {"is_final": True, "text": text}
                    else:
                        result = json.loads(rec.PartialResult())
                        text = result.get("partial", "")
                        if text:
                            yield {"is_final": False, "text": text}

                # Flush any remaining text from the recognizer
                final = json.loads(rec.FinalResult())
                text = final.get("text", "")
                if text:
                    yield {"is_final": True, "text": text}
            finally:
                wf.close()


class WhisperEngine(BaseSTTEngine):
    """faster-whisper based STT engine. Non-streaming (processes entire file at once)."""

    def __init__(self, model_name: str = "small", device: str = "cpu",
                 compute_type: str = "int8"):
        if not _HAS_WHISPER:
            raise RuntimeError(
                "faster_whisper is not installed. Please install it to use the WhisperEngine."
            )
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: str) -> str:
        """Transcribe the entire audio file and return the full text."""
        segments, _ = self._model.transcribe(wav_path, language="en")
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_stream(self, wav_path: str) -> Iterator[dict]:
        """Yield a single final result (Whisper does not support incremental streaming)."""
        if wav_path == "mic":
            raise NotImplementedError("Live microphone streaming is currently only supported with the Vosk STT engine.")
        
        text = self.transcribe(wav_path)
        if text:
            yield {"is_final": True, "text": text}
