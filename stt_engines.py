#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STT engine abstraction layer.
Provides BaseSTTEngine interface and concrete implementations for Vosk and faster-whisper.
"""

from abc import ABC, abstractmethod
import json
import wave
from typing import Iterator


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
        from vosk import Model
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
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, self._sample_rate)
        rec.SetWords(True)

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
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: str) -> str:
        """Transcribe the entire audio file and return the full text."""
        segments, _ = self._model.transcribe(wav_path, language="en")
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_stream(self, wav_path: str) -> Iterator[dict]:
        """Yield a single final result (Whisper does not support incremental streaming)."""
        text = self.transcribe(wav_path)
        if text:
            yield {"is_final": True, "text": text}
