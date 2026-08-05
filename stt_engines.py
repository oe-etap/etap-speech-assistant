#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STT engine abstraction layer.
Provides BaseSTTEngine interface and concrete implementations for Vosk and faster-whisper.

Engines consume an iterator of raw PCM chunks (see audio_sources.py) rather than
a file path, so file and microphone input share one code path and the caller
decides how fast the audio arrives.
"""

from abc import ABC, abstractmethod
import json
from typing import Iterable, Iterator

import numpy as np
from vosk import KaldiRecognizer, Model

from audio_sources import SAMPLE_RATE

try:
    from faster_whisper import WhisperModel
    _HAS_WHISPER = True
except ImportError:
    WhisperModel = None
    _HAS_WHISPER = False


class BaseSTTEngine(ABC):
    """Abstract base class for speech-to-text engines."""

    @abstractmethod
    def transcribe_stream(self, chunks: Iterable[bytes]) -> Iterator[dict]:
        """Transcribe a stream of PCM chunks and yield results incrementally.

        Args:
            chunks: Iterable of mono 16 kHz 16-bit PCM byte chunks.

        Yields dicts with keys:
            - "is_final" (bool): True if this is a finalized recognition result,
                                  False for partial/intermediate results.
            - "text" (str): The recognized text.
        """
        ...


class VoskEngine(BaseSTTEngine):
    """Vosk-based STT engine with streaming partial/final results."""

    def __init__(self, model_path: str):
        self._model = Model(model_path)

    def transcribe_stream(self, chunks: Iterable[bytes]) -> Iterator[dict]:
        """Yield partial and final results as audio chunks arrive.

        A fresh KaldiRecognizer is created per call, so no explicit reset is
        needed between utterances or files.
        """
        rec = KaldiRecognizer(self._model, SAMPLE_RATE)
        rec.SetWords(True)

        for data in chunks:
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "")
                if text:
                    yield {"is_final": True, "text": text}
            else:
                text = json.loads(rec.PartialResult()).get("partial", "")
                if text:
                    yield {"is_final": False, "text": text}

        # Flush any remaining text from the recognizer
        text = json.loads(rec.FinalResult()).get("text", "")
        if text:
            yield {"is_final": True, "text": text}


class WhisperEngine(BaseSTTEngine):
    """faster-whisper based STT engine.

    Whisper has no incremental decoding, so the whole stream is buffered and
    transcribed in one pass once the audio ends.
    """

    def __init__(self, model_name: str = "small", device: str = "cpu",
                 compute_type: str = "int8"):
        if not _HAS_WHISPER:
            raise RuntimeError(
                "faster_whisper is not installed. Please install it to use the WhisperEngine."
            )
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe_stream(self, chunks: Iterable[bytes]) -> Iterator[dict]:
        """Buffer the whole stream, then yield a single final result."""
        pcm = b"".join(chunks)
        if not pcm:
            return

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="en")
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            yield {"is_final": True, "text": text}
