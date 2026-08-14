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
try:
    from piper.voice import PiperVoice
    _HAS_PIPER_API = True
except ImportError:
    PiperVoice = None
    _HAS_PIPER_API = False

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

try:
    from TTS.api import TTS
    _HAS_TTS = True
except ImportError:
    TTS = None
    _HAS_TTS = False


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

    def warmup(self) -> None:
        """Run a throwaway synthesis so the first real chunk pays no init cost.

        Model loading alone does not initialize the inference session; the first
        synthesize() call is measurably slower than the steady state.
        """
        try:
            self.synthesize("Ready.")
        except Exception as e:
            print(f"[WARN] TTS warmup failed: {e}")


class PiperEngine(BaseTTSEngine):
    """Piper TTS engine with Python API and CLI executable fallback."""

    def __init__(self, voice_path: str, exe_path: str = None,
                 use_exe: bool = False, device: str = "cpu"):
        """Initialize the Piper engine.

        Args:
            voice_path: Path to the Piper .onnx voice model file.
            exe_path: Path to the Piper CLI executable (for fallback mode).
            use_exe: If True, use the CLI executable instead of the Python API.
            device: "cpu" or "cuda", the ONNX Runtime execution provider the
                voice model runs on. Ignored when use_exe is True, which has
                no way to select one.

        Raises:
            FileNotFoundError: If the mandatory .onnx.json sidecar config is missing.
            RuntimeError: If "cuda" was asked for and could not be honoured.
        """
        self._voice_path = voice_path
        self._exe_path = exe_path
        self._use_exe = use_exe
        self._device = device
        self._piper_voice = None
        self._sample_rate_val = self._read_sample_rate_from_config()

        if not use_exe:
            if not _HAS_PIPER_API:
                raise RuntimeError(
                    "Piper Python API (piper-tts) is not installed, but use_exe is False. "
                    "Please install it or set piper-use-exe to True in your config."
                )
            if device == "cuda":
                self._preload_cuda_libraries()
            try:
                self._piper_voice = PiperVoice.load(voice_path, use_cuda=(device == "cuda"))
            except Exception as e:
                raise RuntimeError(f"Failed to load Piper Python API model: {e}")

            # A provider that fails to initialize is not an error to ONNX
            # Runtime, which drops to the next one on the list and runs. That
            # would leave a run logged as GPU while every figure in it came off
            # the CPU, so the fallback is refused rather than reported.
            providers = self._piper_voice.session.get_providers()
            if device == "cuda" and "CUDAExecutionProvider" not in providers:
                raise RuntimeError(
                    f"Piper was asked for CUDA but ONNX Runtime fell back to {providers}. "
                    f"Install onnxruntime-gpu built for this GPU's compute capability "
                    f"(see requirements_gpu.txt), or run with --piper-device cpu."
                )
            print(f"[INFO] Piper Python API loaded (model: {voice_path}, "
                  f"provider: {providers[0]})")

    @staticmethod
    def _preload_cuda_libraries() -> None:
        """Put the pip-installed CUDA libraries where the loader will find them.

        onnxruntime-gpu takes CUDA and cuDNN from the `nvidia-*` wheels, whose
        library directories are not on the loader path. Without this the CUDA
        provider's .so fails to load and ONNX Runtime silently uses the CPU.
        """
        if onnxruntime is None:
            return
        preload = getattr(onnxruntime, "preload_dlls", None)
        if preload is None:
            # Older runtimes expect the libraries to come from a system CUDA
            # install, which needs no help from us.
            return
        try:
            preload()
        except Exception as e:
            print(f"[WARN] CUDA library preload failed: {e}")

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
                 speaker: str = "Daisy Studious", device: str = "cpu"):
        """Initialize the Coqui engine.

        Args:
            model_name: Model key under tts_models/multilingual/multi-dataset/.
            language: Language code passed to the model.
            speaker: Built-in speaker id.
            device: "cpu" or "cuda". XTTS-v2 is autoregressive and does not run
                anywhere near real time on a CPU, so this is not the optional
                setting it is for Piper.

        Raises:
            RuntimeError: If Coqui is missing, or "cuda" could not be honoured.
        """
        if not _HAS_TTS:
            raise RuntimeError(
                "Coqui TTS is not installed. Install the coqui-tts package "
                "(see requirements_gpu.txt); the original `TTS` distribution "
                "caps at Python 3.11 and cannot be used here."
            )

        if device == "cuda":
            self._require_working_cuda()

        self._tts = TTS(
            model_name=f"tts_models/multilingual/multi-dataset/{model_name}"
        )
        self._tts.to(device)
        self._device = device
        self._language = language
        self._speaker = speaker
        print(f"[INFO] Coqui loaded (model: {model_name}, device: {device})")

    @staticmethod
    def _require_working_cuda() -> None:
        """Fail now if CUDA cannot actually run a kernel on this GPU.

        torch.cuda.is_available() only reports that a driver and a device are
        present. A wheel built without kernels for the device's compute
        capability passes that check and then fails on the first real op, so
        this launches one rather than trusting the flag.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "Coqui was asked for CUDA but PyTorch is not installed."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Coqui was asked for CUDA but torch.cuda.is_available() is False."
            )
        try:
            torch.zeros(8, device="cuda").add_(1).cpu()
        except Exception as e:
            cap = torch.cuda.get_device_capability(0)
            raise RuntimeError(
                f"Coqui was asked for CUDA but this PyTorch cannot run on the GPU. "
                f"Device is compute capability {cap[0]}.{cap[1]}; "
                f"torch {torch.__version__} was built for {torch.cuda.get_arch_list()}. "
                f"Install a build that covers it (see requirements_gpu.txt), "
                f"or run with --coqui-device cpu. Original error: {e}"
            )

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
