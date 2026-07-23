#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS engine abstraction layer.
Provides BaseTTSEngine interface and concrete implementations for Piper and Coqui TTS.
"""

from abc import ABC, abstractmethod
import json
import os
import subprocess

import numpy as np


class BaseTTSEngine(ABC):
    """Abstract base class for text-to-speech engines."""

    @abstractmethod
    def synthesize(self, text: str) -> tuple:
        """Synthesize text to raw PCM audio.

        Args:
            text: The text to synthesize.

        Returns:
            Tuple of (pcm_bytes, sample_rate) where pcm_bytes is 16-bit signed
            integer PCM data and sample_rate is the audio sample rate in Hz.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the output sample rate in Hz."""
        ...


class PiperEngine(BaseTTSEngine):
    """Piper TTS engine with Python API and CLI executable fallback."""

    def __init__(self, voice_path: str, exe_path: str = None,
                 use_exe: bool = False):
        """Initialize the Piper engine.

        Args:
            voice_path: Path to the Piper .onnx voice model file.
            exe_path: Path to the Piper CLI executable (for fallback mode).
            use_exe: If True, use the CLI executable instead of the Python API.

        Raises:
            FileNotFoundError: If the mandatory .onnx.json sidecar config is missing.
        """
        self._voice_path = voice_path
        self._exe_path = exe_path
        self._use_exe = use_exe
        self._piper_voice = None
        self._sample_rate_val = self._read_sample_rate_from_config()

        if not use_exe:
            try:
                from piper.voice import PiperVoice
                self._piper_voice = PiperVoice.load(voice_path)
                print(f"[INFO] Piper Python API loaded (model: {voice_path})")
            except Exception as e:
                print(f"[WARN] Piper Python API failed ({e}), "
                      f"falling back to CLI executable")
                self._use_exe = True

    def _read_sample_rate_from_config(self) -> int:
        """Read sample rate from the Piper voice JSON sidecar file.

        The .onnx.json sidecar is mandatory for determining the correct
        audio sample rate.
        """
        json_path = self._voice_path + ".json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Piper voice config not found: {json_path}. "
                f"The .onnx.json sidecar file is required."
            )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                vdata = json.load(f)
                return vdata.get("audio", {}).get("sample_rate", 22050)
        except (json.JSONDecodeError, KeyError):
            return 22050

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_val

    def synthesize(self, text: str) -> tuple:
        """Synthesize text using whichever backend is available."""
        if self._piper_voice is not None:
            return self._synthesize_python(text)
        return self._synthesize_exe(text)

    def _synthesize_python(self, text: str) -> tuple:
        """Synthesize using the Piper Python API."""
        pcm_parts = []
        sr = None
        for audio_chunk in self._piper_voice.synthesize(text):
            pcm_parts.append(audio_chunk.audio_int16_bytes)
            if sr is None:
                sr = audio_chunk.sample_rate
        return b"".join(pcm_parts), sr or self._sample_rate_val

    def _synthesize_exe(self, text: str) -> tuple:
        """Synthesize using the Piper CLI executable."""
        cmd = [self._exe_path, "-m", self._voice_path, "--output_raw"]
        result = subprocess.run(
            cmd, input=text.encode("utf-8"),
            stdout=subprocess.PIPE, check=True
        )
        return result.stdout, self._sample_rate_val


class CoquiEngine(BaseTTSEngine):
    """Coqui TTS engine."""

    def __init__(self, model_name: str = "xtts_v2", language: str = "en",
                 speaker: str = "Daisy Studious"):
        from TTS.api import TTS
        self._tts = TTS(
            model_name=f"tts_models/multilingual/multi-dataset/{model_name}"
        )
        self._language = language
        self._speaker = speaker

    @property
    def sample_rate(self) -> int:
        return self._tts.synthesizer.output_sample_rate

    def synthesize(self, text: str) -> tuple:
        """Synthesize text to PCM audio via Coqui TTS."""
        wav = self._tts.tts(
            text=text, speaker=self._speaker, language=self._language
        )
        audio_data = np.array(wav, dtype=np.float32)
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767.0).astype(np.int16)
        return audio_data.tobytes(), self.sample_rate
