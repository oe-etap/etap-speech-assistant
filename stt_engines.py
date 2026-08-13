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
import os
import shutil
import tempfile
from pathlib import Path
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
            - "speech_end_s" (float, final results only): offset in seconds from
              the start of the stream to the end of the last recognized word.
              This is where the speaker actually stopped, which is earlier than
              the point the engine finalizes: an endpointer has to observe
              trailing silence first, and a file may carry silence past its last
              word. Omitted or None when the engine cannot report it, which
              leaves the metrics anchored on it blank rather than guessed.
        """
        ...


# Kaldi's endpointing rules fire on an OR. Rules 2, 3 and 4 each pair a silence
# length with a decoder-confidence bound, so the shipped model has no single
# "wait this long" number; writing the same value into all three gives it one.
# Rule 1 (silence with nothing recognized at all) is a different question and
# keeps its own setting; rule 5 caps utterance length and is left alone.
_ENDPOINT_RULES = ("rule2", "rule3", "rule4")


def write_endpoint_variant(model_path: str, dest: str, trailing_silence_s: float,
                           quiet_timeout_s: float = None) -> str:
    """Materialize a Vosk model whose endpointer waits `trailing_silence_s`.

    Vosk 0.3.44 has no runtime endpointer API -- `SetEndpointerDelays` arrived
    in a later release -- so the setting can only reach Kaldi through the
    model's own `conf/model.conf`, which is read once at load time. The model is
    not copied: everything but `conf/` is linked back to the original, which
    keeps a variant to a few hundred bytes. Where the filesystem refuses links
    (Windows without the privilege) it falls back to copying.

    `quiet_timeout_s` overrides rule 1, how long a silence with nothing
    recognized in it runs before the recognizer gives up on the utterance.
    Left alone by default.
    """
    src, dst = Path(model_path).resolve(), Path(dest)
    if dst.is_dir():
        shutil.rmtree(dst)
    (dst / "conf").mkdir(parents=True)

    def link(entry: Path, target: Path):
        try:
            os.symlink(entry, target)
        except (OSError, NotImplementedError):
            (shutil.copytree if entry.is_dir() else shutil.copy2)(entry, target)

    for entry in src.iterdir():
        if entry.name != "conf":
            link(entry, dst / entry.name)
    for entry in (src / "conf").iterdir():
        if entry.name != "model.conf":
            link(entry, dst / "conf" / entry.name)

    kept = [ln for ln in (src / "conf" / "model.conf").read_text().splitlines()
            if ln.strip() and not any(f"endpoint.{r}." in ln for r in _ENDPOINT_RULES)]
    kept += [f"--endpoint.{rule}.min-trailing-silence={trailing_silence_s}"
             for rule in _ENDPOINT_RULES]
    if quiet_timeout_s is not None:
        kept = [ln for ln in kept if "endpoint.rule1." not in ln]
        kept.append(f"--endpoint.rule1.min-trailing-silence={quiet_timeout_s}")
    (dst / "conf" / "model.conf").write_text("\n".join(kept) + "\n")
    return str(dst)


class VoskEngine(BaseSTTEngine):
    """Vosk-based STT engine with streaming partial/final results."""

    def __init__(self, model_path: str, endpoint_silence_ms: int = 0):
        """
        Args:
            model_path: English Vosk model directory.
            endpoint_silence_ms: Silence after the last word before the
                endpointer calls the utterance over. 0 leaves the model's own
                settings in place. Anything else is applied through a temporary
                model variant, since the value cannot be set at runtime.
        """
        self._workdir = None
        if endpoint_silence_ms:
            self._workdir = tempfile.TemporaryDirectory(prefix="vosk-endpoint-")
            model_path = write_endpoint_variant(
                model_path, os.path.join(self._workdir.name, "model"),
                endpoint_silence_ms / 1000.0)
        self._model = Model(model_path)

    def transcribe_stream(self, chunks: Iterable[bytes]) -> Iterator[dict]:
        """Yield partial and final results as audio chunks arrive.

        A fresh KaldiRecognizer is created per call, so no explicit reset is
        needed between utterances or files.

        AcceptWaveform returning True is Vosk's endpointer firing: it has seen
        enough trailing silence to call the utterance over. That happens mid
        stream and is what drives turn-taking on a live microphone.
        """
        rec = KaldiRecognizer(self._model, SAMPLE_RATE)
        rec.SetWords(True)

        for data in chunks:
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get("text"):
                    yield self._final(result)
            else:
                text = json.loads(rec.PartialResult()).get("partial", "")
                if text:
                    yield {"is_final": False, "text": text}

        # Flush any remaining text from the recognizer. Reached when the stream
        # ends before the endpointer fires, which only happens on file input.
        result = json.loads(rec.FinalResult())
        if result.get("text"):
            yield self._final(result)

    @staticmethod
    def _final(result: dict) -> dict:
        """Build a final result, carrying the word-level end time if present.

        SetWords(True) puts a per-word list in "result", each entry timed in
        seconds from the start of the stream. The last word's end is the moment
        the speaker stopped.
        """
        words = result.get("result") or []
        return {
            "is_final": True,
            "text": result["text"],
            "speech_end_s": words[-1].get("end") if words else None,
        }


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
        """Buffer the whole stream, then yield a single final result.

        Having no endpointer, this engine cannot tell that an utterance is over
        until the stream itself ends, so it never yields mid-stream. The segment
        timestamps still locate the end of the speech within the audio.
        """
        pcm = b"".join(chunks)
        if not pcm:
            return

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="en")
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            yield {
                "is_final": True,
                "text": text,
                "speech_end_s": segments[-1].end if segments else None,
            }
