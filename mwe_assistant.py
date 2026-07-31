#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Speech Assistant MWE (EN): audio file -> STT -> LLM -> TTS.
Supports selectable engines:
  - STT: Vosk or faster-whisper
  - TTS: Piper or Coqui TTS
  - mode: CPU/GPU preset, with Whisper available on CPU too

Audio flow: batch file processing only, no microphone recording or playback.
Latency is measured per stage and saved to CSV, together with best-effort
CPU/RAM/GPU snapshots.
"""

import argparse
import csv
import queue
import threading
import yaml
import json
import os
import shutil
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

import soundfile as sf
try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    sd = None
    _HAS_SD = False

from audio_sources import (
    DEFAULT_CHUNK_MS, PACING_FAST, PACING_REALTIME,
    FileAudioSource, MicAudioSource,
)
from pipeline_channels import UtteranceMailbox
from stt_engines import VoskEngine, WhisperEngine
from llm_engine import DEFAULT_CHUNK_MAX_CHARS, DEFAULT_SYSTEM_PROMPT, OllamaEngine
from tts_engines import PiperEngine, CoquiEngine

# Optional module-level imports
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False

_GPU_BLANK_STATS = {
    "gpu_util_percent": "",
    "gpu_mem_used_mb": "",
    "gpu_mem_total_mb": "",
    "gpu_name": "",
}

# Latest reading of the nvidia-smi fallback, replaced wholesale by the sampler
# thread so that readers need no lock. Unused while NVML is active.
_gpu_stats = _GPU_BLANK_STATS

# How often the nvidia-smi fallback refreshes its reading
GPU_SAMPLE_INTERVAL_S = 1.0

# How long to wait for the pipeline workers to drain before reading their metrics
WORKER_SHUTDOWN_TIMEOUT_S = 10.0

# Silence an endpointer needs after the last word before it calls an utterance
# over. Measured against vosk-model-small-en-us-0.15, which fires at ~1100 ms
# and not at all below it. Input files with less than this never exercise the
# endpointer, so their latency figures do not carry over to live input.
MIN_TRAILING_SILENCE_MS = 1200

# What hands an utterance to the LLM.
TRIGGER_ENDPOINT = "endpoint"        # the STT calling the utterance over
TRIGGER_END_OF_FILE = "end-of-file"  # the input running out


# ---------- Helpers ----------
def is_mono_16k_pcm(path):
    """Check if a file is already mono 16kHz 16-bit PCM WAV."""
    try:
        with wave.open(str(path), "rb") as wf:
            return (wf.getnchannels() == 1 and
                    wf.getframerate() == 16000 and
                    wf.getsampwidth() == 2)
    except Exception:
        return False


def ensure_wav_mono_16k(path, out_dir=None, prefix=None):
    """Convert any audio to mono 16kHz WAV using ffmpeg. Skips if already correct format."""
    source = Path(path)
    if out_dir:
        stem = f"{prefix}_{source.stem}" if prefix else source.stem
        out_path = Path(out_dir) / f"{stem}_16k.wav"
    else:
        base, _ = os.path.splitext(path)
        out_path = Path(base + "_16k.wav")

    # Skip ffmpeg if input is already mono 16kHz 16-bit PCM WAV
    if is_mono_16k_pcm(source):
        shutil.copyfile(str(source), str(out_path))
        return str(out_path)

    cmd = ["ffmpeg", "-y", "-i", str(source), "-ac", "1", "-ar", "16000", "-f", "wav", str(out_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return str(out_path)


def load_system_prompt(path):
    """Return (variant_label, prompt_text) for the configured prompt file.

    The label identifies which variant produced a measurement, and it is simply
    the file name. That is why a prompt file must never be edited in place: a
    reworded prompt belongs in a new file, so a given label always denotes the
    same text.
    """
    if not path:
        return "(built-in default)", DEFAULT_SYSTEM_PROMPT

    prompt_file = Path(path)
    text = prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"System prompt file is empty: {path}")
    return prompt_file.name, text


def save_system_prompt(run_dir, label, text):
    """Record the prompt this run used, headed by its variant name.

    config_used.yaml only stores the path, so the prompt is copied here as well:
    the run stays reproducible even if the prompts/ directory moves on.
    """
    with open(os.path.join(run_dir, "system_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(f"{label}\n\n{text}\n")


def wav_duration_ms(wav_path):
    with sf.SoundFile(wav_path) as wav:
        return int((len(wav) / wav.samplerate) * 1000)


def _numeric_or_blank(value):
    """Return the value unchanged if it parses as a number, otherwise "".

    nvidia-smi reports fields the driver cannot supply as markers such as
    '[N/A]' or '[Not Supported]'. Blanking them keeps the CSV column numeric.
    """
    try:
        float(value)
    except (TypeError, ValueError):
        return ""
    return value


_nvml_handle = None


def _nvml_value(read, default=""):
    """Return read()'s result, or default when the driver will not supply it."""
    try:
        return read()
    except Exception:
        return default


def _read_gpu_stats_nvml():
    """Read the GPU counters through NVML.

    NVML is an in-process library call rather than a subprocess, so this is
    cheap enough to run at a stage boundary. Fields the driver does not
    support are blanked individually.
    """
    stats = dict(_GPU_BLANK_STATS)
    handle = _nvml_handle
    if handle is None:
        return stats

    name = _nvml_value(lambda: pynvml.nvmlDeviceGetName(handle))
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    stats["gpu_name"] = name

    stats["gpu_util_percent"] = _nvml_value(
        lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)

    mem = _nvml_value(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle), None)
    if mem is not None:
        stats["gpu_mem_used_mb"] = round(mem.used / (1024 * 1024))
        stats["gpu_mem_total_mb"] = round(mem.total / (1024 * 1024))

    return stats


def _refresh_gpu_stats():
    """Query nvidia-smi once and publish the reading to _gpu_stats.

    Returns True on success, False if nvidia-smi is unavailable or returned
    nothing usable.
    """
    global _gpu_stats
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return False
        first_gpu = result.stdout.strip().splitlines()[0]
        name, util, mem_used, mem_total = [part.strip() for part in first_gpu.split(",", 3)]
        _gpu_stats = {
            "gpu_name": name,
            "gpu_util_percent": _numeric_or_blank(util),
            "gpu_mem_used_mb": _numeric_or_blank(mem_used),
            "gpu_mem_total_mb": _numeric_or_blank(mem_total),
        }
        return True
    except Exception:
        return False


class _GpuSampler(threading.Thread):
    """Background thread refreshing _gpu_stats every GPU_SAMPLE_INTERVAL_S.

    Used only on the nvidia-smi fallback path. Keeps the subprocess off the
    measured pipeline: workers read the published dict instead of running
    nvidia-smi themselves.
    """

    def __init__(self, interval=GPU_SAMPLE_INTERVAL_S):
        super().__init__(daemon=True, name="gpu-sampler")
        self._interval = interval
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(self._interval):
            # A failed refresh keeps the previous reading.
            _refresh_gpu_stats()

    def stop(self):
        self._stop_event.set()


_gpu_sampler = None


def start_gpu_monitor():
    """Begin collecting GPU statistics, preferring NVML over nvidia-smi.

    NVML is read in process at each stage boundary. Where it is unavailable the
    nvidia-smi fallback starts instead, kept fresh by a background thread.
    Neither being usable leaves the GPU columns blank.
    """
    global _nvml_handle, _gpu_sampler
    if _nvml_handle is not None or _gpu_sampler is not None:
        return

    if _HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return
        except Exception:
            _nvml_handle = None

    if not _refresh_gpu_stats():
        return
    _gpu_sampler = _GpuSampler()
    _gpu_sampler.start()


def stop_gpu_monitor():
    """Release NVML and stop the nvidia-smi fallback thread."""
    global _nvml_handle, _gpu_sampler
    if _nvml_handle is not None:
        _nvml_handle = None
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    if _gpu_sampler is not None:
        _gpu_sampler.stop()
        _gpu_sampler.join(timeout=GPU_SAMPLE_INTERVAL_S * 2)
        _gpu_sampler = None


def prime_cpu_percent():
    """Open a CPU measurement window on the calling thread.

    psutil.cpu_percent(interval=None) reports load since the calling thread's
    own previous call, its state being keyed by thread id. Call this when a
    stage starts; the snapshot taken when that stage ends then covers exactly
    that stage.
    """
    if not _HAS_PSUTIL:
        return
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        pass


def collect_resource_snapshot():
    """Return the resource snapshot for the stage that has just finished.

    Call at a stage boundary, from the thread that ran the stage and opened its
    window with prime_cpu_percent(); cpu_percent then covers that stage alone.
    ram_percent and rss_mb are read directly, the GPU fields come from NVML or,
    where that is unavailable, from the background nvidia-smi sampler. Missing
    psutil/NVML/nvidia-smi leaves blanks.
    """
    stats = {
        "cpu_percent": "",
        "ram_percent": "",
        "rss_mb": "",
    }

    if _HAS_PSUTIL:
        try:
            proc = psutil.Process(os.getpid())
            stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            stats["ram_percent"] = psutil.virtual_memory().percent
            stats["rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 1)
        except Exception:
            pass

    if _nvml_handle is not None:
        stats.update(_read_gpu_stats_nvml())
    else:
        stats.update(_gpu_stats)
    return stats


def triggers_on_endpoint(args):
    """Whether an utterance is released as soon as the STT calls it over.

    A microphone has no other option, its stream never ends. File input can
    wait for the file instead, and under 'fast' pacing it always does: the
    audio is consumed faster than real time, so waiting costs nothing and the
    stages that would show a difference are not measured there anyway.
    """
    if args.input_mode == "mic":
        return True
    if args.audio_pacing != PACING_REALTIME:
        return False
    return args.file_realtime_trigger == TRIGGER_ENDPOINT


def write_timing(writer, args, item, stage, duration_ms, stats=None, extra=None):
    """Append one CSV row for a stage.

    stats is the snapshot collect_resource_snapshot() took at that stage's
    boundary; None leaves the resource columns blank.
    """
    stats = stats or {}
    writer.writerow({
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "stt_engine": args.stt_engine,
        "tts_engine": args.tts_engine,
        "input_mode": args.input_mode,
        "audio_pacing": args.audio_pacing if args.input_mode == "file" else "",
        # The behaviour that applied, not the setting that was asked for: the
        # setting only takes effect under file input with realtime pacing.
        "utterance_trigger": (TRIGGER_ENDPOINT if triggers_on_endpoint(args)
                              else TRIGGER_END_OF_FILE),
        "item": item,
        "stage": stage,
        "duration_ms": int(duration_ms),
        "cpu_percent": stats.get("cpu_percent", ""),
        "ram_percent": stats.get("ram_percent", ""),
        "rss_mb": stats.get("rss_mb", ""),
        "gpu_util_percent": stats.get("gpu_util_percent", ""),
        "gpu_mem_used_mb": stats.get("gpu_mem_used_mb", ""),
        "gpu_mem_total_mb": stats.get("gpu_mem_total_mb", ""),
        "gpu_name": stats.get("gpu_name", ""),
        "extra_json": json.dumps(extra or {}, ensure_ascii=True),
    })


# ---------- Pipeline Workers ----------
def stt_worker(stt_engine, audio_source, mailbox, stt_metrics,
               trigger_on_endpoint=False):
    """Feed audio to the STT engine and hand finalized utterances to the LLM.

    trigger_on_endpoint decides what releases an utterance:

    - True: every finalized result goes over immediately, so the answer starts
      as soon as the STT calls the utterance over. A later utterance can then
      interrupt an answer still being generated. This is the only option a live
      microphone has, since its stream never ends.
    - False: nothing is handed over until the input runs out, and the LLM is
      triggered once with everything that was recognized.

    Partial results are printed for visibility but never trigger generation:
    starting on a partial costs a full generation whenever the guess is wrong,
    which is what made the earlier experiments slower rather than faster.

    Records two instants per finalized result: when the speaker stopped, which
    is the anchor for the TTFA a user perceives, and when the transcript existed.
    The gap between them is the engine's endpointing delay.
    """
    try:
        # Opens this thread's CPU window for the span reported as stt_ms.
        prime_cpu_percent()
        t0 = time.perf_counter()
        accumulated_final = ""

        for result in stt_engine.transcribe_stream(audio_source.chunks()):
            if not result["text"]:
                continue

            if not result["is_final"]:
                print(f"[STT Partial] {result['text']}", end="\r")
                continue

            accumulated_final = (
                accumulated_final + " " + result["text"]
            ).strip()

            speech_end_s = result.get("speech_end_s")
            stt_metrics["final_t"] = time.perf_counter()
            stt_metrics["speech_end_t"] = audio_source.speech_end_t(speech_end_s)
            stt_metrics["speech_end_s"] = speech_end_s
            stt_metrics["user_text"] = accumulated_final

            if trigger_on_endpoint:
                # Carries everything recognized so far, not just this result:
                # an endpointer that fires while the speaker is only drawing
                # breath would otherwise have the LLM answer half a sentence.
                # The mailbox supersedes the earlier text and the generation
                # started on it is cancelled.
                mailbox.put(accumulated_final)

        if not trigger_on_endpoint and accumulated_final:
            mailbox.put(accumulated_final)

        t1 = time.perf_counter()
        stt_metrics["stt_ms"] = int((t1 - t0) * 1000)
        stt_metrics["user_text"] = accumulated_final
        # Covers the STT stage.
        stt_metrics["stats"] = collect_resource_snapshot()
        print(f"[STT] {accumulated_final!r}")

    except Exception as e:
        print(f"[STT] Error: {e}")
        stt_metrics["error"] = str(e)
    finally:
        mailbox.close()


def llm_worker(llm_engine, mailbox, tts_queue, llm_metrics):
    """Answer utterances from the mailbox, streaming chunks to tts_queue.

    Barge-in: if a newer utterance arrives while generation is in flight, the
    current generation is cancelled, the TTS discards its partial output, and a
    fresh generation starts from the newer text. Only finalized utterances get
    here, so a restart always reflects something the user actually finished
    saying.
    """
    try:
        while True:
            llm_metrics["is_busy"] = False
            text = mailbox.take()
            llm_metrics["is_busy"] = True

            if text is None:
                break

            # Drop any partial audio from a superseded answer and reset metrics
            llm_engine.cancel()
            tts_queue.put({"type": "cancel"})
            llm_metrics["full_assistant_text"] = ""
            llm_metrics["llm_chunk_count"] = 0
            llm_metrics["llm_ttfc_ms"] = None
            llm_metrics["llm_ttft_ms"] = None
            llm_metrics["first_chunk_chars"] = None
            llm_metrics["ttft_stats"] = None
            llm_metrics["ttfc_stats"] = None
            llm_metrics["end_stats"] = None

            print("[LLM] Starting stream...")
            # Opens this thread's CPU window; the wait on the mailbox that
            # precedes the request stays outside it.
            prime_cpu_percent()
            llm_t0 = time.perf_counter()
            llm_metrics["llm_t0"] = llm_t0

            for chunk_data in llm_engine.generate_stream(text):
                if chunk_data.get("ollama_stats"):
                    llm_metrics["ollama_stats"] = chunk_data["ollama_stats"]

                first_token_t = chunk_data.get("first_token_t")
                if first_token_t is not None and llm_metrics["llm_ttft_ms"] is None:
                    llm_metrics["llm_ttft_ms"] = int((first_token_t - llm_t0) * 1000)
                    llm_metrics["first_token_t"] = first_token_t
                    llm_metrics["ttft_stats"] = collect_resource_snapshot()

                if chunk_data.get("cancelled"):
                    break

                if mailbox.has_pending():
                    print("[LLM] Newer utterance arrived, restarting generation...")
                    llm_engine.cancel()
                    break

                chunk = chunk_data.get("text", "")
                if not chunk:
                    continue

                llm_metrics["llm_chunk_count"] += 1

                if llm_metrics["llm_ttfc_ms"] is None:
                    llm_metrics["llm_ttfc_ms"] = int(
                        (time.perf_counter() - llm_t0) * 1000
                    )
                    # Length of the opening chunk shows whether the model
                    # honoured a "start with a short sentence" instruction.
                    llm_metrics["first_chunk_chars"] = len(chunk)
                    llm_metrics["ttfc_stats"] = collect_resource_snapshot()

                print(f"[LLM Chunk {llm_metrics['llm_chunk_count']}] {chunk}")
                llm_metrics["full_assistant_text"] += chunk + " "

                tts_queue.put({"type": "text", "text": chunk})

            llm_metrics["end_stats"] = collect_resource_snapshot()

            # Let the TTS close this response's WAV before the next one starts
            tts_queue.put({"type": "end_of_response"})

    except Exception as e:
        print(f"[LLM] Error: {e}")
        llm_metrics["error"] = str(e)
    finally:
        tts_queue.put(None)  # EOF signal


def response_wav_path(out_wav, index):
    """Return the WAV path for the index-th response of this item.

    The first response keeps the plain name so file mode, which only ever has
    one, is unaffected. A mic session answers repeatedly and would otherwise
    overwrite every earlier answer.
    """
    if index <= 1:
        return out_wav
    stem, ext = os.path.splitext(out_wav)
    return f"{stem}_r{index}{ext}"


def tts_worker(tts_engine, tts_queue, out_wav, tts_metrics, playback=False):
    """Read text chunks from tts_queue and synthesize them to WAV files.

    Message types:
        - "text": synthesize and append to the current response WAV.
        - "cancel": the answer was superseded, so drop the partial WAV, flush
          any queued audio from the speakers, and reset the metrics.
        - "end_of_response": the answer is complete; close its WAV so the next
          one starts a new file.
    """
    wav_file = None
    stream = None
    response_index = 1
    tts_metrics["out_wavs"] = []

    def close_response():
        nonlocal wav_file, response_index
        if wav_file:
            wav_file.close()
            wav_file = None
            tts_metrics["out_wavs"].append(response_wav_path(out_wav, response_index))
            response_index += 1

    try:
        if playback:
            if not _HAS_SD:
                raise ImportError("sounddevice module is required for playback")
            stream = sd.RawOutputStream(
                samplerate=tts_engine.sample_rate, 
                channels=1, 
                dtype='int16'
            )
            stream.start()

        while True:
            tts_metrics["is_busy"] = False
            msg = tts_queue.get()
            tts_metrics["is_busy"] = True
            
            if msg is None:
                break

            msg_type = msg.get("type", "text")

            if msg_type == "end_of_response":
                close_response()
                continue

            if msg_type == "cancel":
                # Discard the partial WAV of the superseded answer
                partial_path = response_wav_path(out_wav, response_index)
                if wav_file:
                    wav_file.close()
                    wav_file = None
                if os.path.exists(partial_path):
                    try:
                        os.remove(partial_path)
                    except OSError:
                        pass

                # abort() drops audio already queued on the device; stop() would
                # play it out first, so the old answer would keep talking over
                # the new one.
                if stream:
                    stream.abort()
                    stream.start()

                # Reset metrics so they reflect only the answer that survives
                tts_metrics["tts_first_chunk_ms"] = None
                tts_metrics["tts_first_chunk_t"] = None
                tts_metrics["first_chunk_stats"] = None
                tts_metrics["total_tts_time"] = 0.0
                continue

            text = msg.get("text", "")
            if not text:
                continue

            if tts_metrics["tts_first_chunk_ms"] is None:
                # Opens this thread's CPU window so it covers the first
                # synthesis alone; later chunks extend it towards end_stats.
                prime_cpu_percent()

            tts_t0 = time.perf_counter()
            pcm_bytes, sample_rate = tts_engine.synthesize(text)
            tts_t1 = time.perf_counter()
            tts_metrics["total_tts_time"] += (tts_t1 - tts_t0)

            if tts_metrics["tts_first_chunk_ms"] is None:
                tts_metrics["tts_first_chunk_ms"] = int(
                    (tts_t1 - tts_t0) * 1000
                )
                tts_metrics["tts_first_chunk_t"] = tts_t1
                tts_metrics["first_chunk_stats"] = collect_resource_snapshot()

            if wav_file is None:
                wav_file = wave.open(response_wav_path(out_wav, response_index), "wb")
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)

            wav_file.writeframes(pcm_bytes)

            if stream:
                stream.write(pcm_bytes)

            tts_metrics["last_play_t"] = time.perf_counter()

    except Exception as e:
        print(f"[TTS] Error: {e}")
        tts_metrics["error"] = str(e)
    finally:
        # Taken before the WAV is closed, so it covers synthesis not teardown.
        tts_metrics["end_stats"] = collect_resource_snapshot()
        close_response()
        if stream:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Offline Speech Assistant MWE (EN, file input only)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (CLI args override it)")
    parser.add_argument("--audio", nargs="+", help="One or more audio files (wav/mp3/flac/etc.)")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu", help="Default compute preset")
    parser.add_argument("--stt-engine", choices=["vosk", "whisper"], default="whisper", help="Speech-to-text engine")
    parser.add_argument("--tts-engine", choices=["piper", "coqui"], default="piper", help="Text-to-speech engine")

    parser.add_argument("--out-dir", type=str, default="outputs", help="Where to save the audiofiles")
    parser.add_argument("--latency-csv", type=str, default=None, help="CSV file to append latency/resource logs")
    parser.add_argument("--keep-normalized", action="store_true", help="Keep ffmpeg-normalized input WAVs")
    parser.add_argument("--input-mode", choices=["file", "mic"], default="file", help="Input mode: file or live mic")
    parser.add_argument("--audio-pacing", choices=[PACING_REALTIME, PACING_FAST], default=PACING_REALTIME,
                        help="How input files are fed to the STT engine. 'realtime' simulates a "
                             "microphone at 1x speed so STT overlaps with speech (representative "
                             "latency). 'fast' reads as quickly as possible (original behaviour).")
    parser.add_argument("--file-realtime-trigger",
                        choices=[TRIGGER_ENDPOINT, TRIGGER_END_OF_FILE],
                        default=TRIGGER_ENDPOINT,
                        help="What starts the LLM under file input with realtime pacing. "
                             "'endpoint' answers as soon as the STT calls the utterance "
                             "over, the only thing a microphone can do. 'end-of-file' "
                             "waits for the whole file, so any silence past the endpoint "
                             "is added to the latency. Ignored under 'fast' pacing and in "
                             "mic mode, and has no effect with Whisper, which has no "
                             "endpointer.")
    parser.add_argument("--audio-chunk-ms", type=int, default=DEFAULT_CHUNK_MS,
                        help="Audio chunk length in milliseconds fed to the STT engine")
    parser.add_argument("--playback", action="store_true", help="Play TTS output on speakers")
    parser.add_argument("--idle-timeout", type=float, default=10.0, help="Seconds of silence before exiting mic mode")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # STT options
    parser.add_argument("--vosk-model", default="./vosk/vosk-model-small-en-us-0.15", help="Path to English Vosk model dir")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model name")
    parser.add_argument("--whisper-device", choices=["cpu", "cuda"], default=None, help="Device for faster-whisper")
    parser.add_argument("--whisper-compute-type", default=None, help="Compute type (int8, float16, etc.)")

    # TTS options
    parser.add_argument("--piper-exe", default="./piper/piper.exe", help="Path to Piper executable (piper or piper.exe)")
    parser.add_argument("--piper-voice", default="./piper/en_US-lessac-medium.onnx", help="Path to English Piper .onnx voice")
    parser.add_argument("--piper-use-exe", action="store_true", help="Use Piper CLI executable instead of Python API (slower, spawns subprocess per chunk)")
    parser.add_argument("--coqui-voice", default="xtts_v2", help="Coqui voice model key")
    parser.add_argument("--coqui-language", default="en", help="Language code for Coqui")
    parser.add_argument("--coqui-speaker", default="Daisy Studious", help="Speaker id for Coqui")

    # Shared LLM
    parser.add_argument("--ollama-model", default="phi3:mini", help="Ollama model (e.g. phi3:mini)")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama generate endpoint")
    parser.add_argument("--tts-chunk-max-chars", type=int, default=DEFAULT_CHUNK_MAX_CHARS,
                        help="Safety-net chunk length: a sentence longer than this is split at a "
                             "word boundary so TTS does not wait for the whole response. "
                             "Lower means lower TTFA, higher means better prosody.")
    parser.add_argument("--system-prompt-file", type=str, default=None,
                        help="Path to a .txt file holding the system prompt. Prompt variants live "
                             "in prompts/ and are identified by filename; a variant is never edited "
                             "in place, a change means a new file. Falls back to the built-in prompt.")
    parser.add_argument("--llm-temperature", type=float, default=0.0,
                        help="Sampling temperature. 0 is greedy decoding, which makes a run "
                             "reproducible on its own; the length-dependent stages are "
                             "otherwise noisy enough to hide whatever is under test. Raise "
                             "it only with --llm-seed set.")
    parser.add_argument("--llm-seed", type=int, default=None,
                        help="Sampling seed. Leave unset for non-deterministic sampling.")
    parser.add_argument("--llm-max-tokens", type=int, default=150,
                        help="Maximum number of tokens the LLM may generate")

    # ---- config file handling & parse ----
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=str)
    known_args, _ = initial_parser.parse_known_args()

    if known_args.config:
        with open(known_args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
            yaml_cfg = {k.replace("-", "_"): v for k, v in yaml_cfg.items()}
            parser.set_defaults(**yaml_cfg)

    args = parser.parse_args()

    if args.input_mode == "mic":
        args.audio = ["mic"]
    else:
        if not args.audio:
            parser.error("--audio is required when --input-mode is 'file' (either in CLI or config)")

        expanded_audio = []
        valid_extensions = {
            ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".m4b", ".aac", ".wma",
            ".amr", ".aiff", ".opus", ".webm", ".mp4", ".mkv", ".avi", ".mov"
        }
        for p in args.audio:
            path_obj = Path(p)
            if not path_obj.exists():
                print(f"[WARN] Input path does not exist: {p}")
                continue
            if path_obj.is_dir():
                for child in path_obj.iterdir():
                    if child.is_file() and child.suffix.lower() in valid_extensions:
                        expanded_audio.append(str(child))
            else:
                expanded_audio.append(str(path_obj))

        if not expanded_audio:
            parser.error("No valid audio files found in the provided --audio paths.")

        args.audio = sorted(expanded_audio)

    # Prompt files are local and not version controlled, so a missing one is a
    # normal setup mistake rather than a bug: report it like any bad argument.
    try:
        prompt_label, system_prompt = load_system_prompt(args.system_prompt_file)
    except (OSError, ValueError) as e:
        parser.error(f"--system-prompt-file could not be loaded: {e}")

    if args.whisper_device is None:
        args.whisper_device = "cuda" if args.mode == "gpu" else "cpu"
    if args.whisper_compute_type is None:
        args.whisper_compute_type = "float16" if args.whisper_device == "cuda" else "int8"

    # Any one-off setup cost happens here, before a stage is ever timed.
    start_gpu_monitor()

    # Pre-load (warmup) models so their initialization time isn't counted
    # in the latency of the first file
    print("[INFO] Pre-loading AI models (STT, LLM, TTS)...")

    # Initialize STT engine
    if args.stt_engine == "vosk":
        stt_engine = VoskEngine(args.vosk_model)
    elif args.stt_engine == "whisper":
        stt_engine = WhisperEngine(
            args.whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
        )

    # Initialize LLM engine
    print(f"[INFO] System prompt: {prompt_label}")
    llm_engine = OllamaEngine(
        model=args.ollama_model,
        url=args.ollama_url,
        system_prompt=system_prompt,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        seed=args.llm_seed,
        chunk_max_chars=args.tts_chunk_max_chars,
    )
    llm_engine.warmup()

    # Initialize TTS engine
    if args.tts_engine == "piper":
        tts_engine = PiperEngine(
            voice_path=args.piper_voice,
            exe_path=args.piper_exe,
            use_exe=args.piper_use_exe,
        )
    elif args.tts_engine == "coqui":
        tts_engine = CoquiEngine(
            model_name=args.coqui_voice,
            language=args.coqui_language,
            speaker=args.coqui_speaker,
        )
    tts_engine.warmup()

    run_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    if not args.latency_csv:
        args.latency_csv = os.path.join(run_dir, f"latency_log_{timestamp}.csv")

    config_to_save = vars(args).copy()

    # Filter out settings that are not used by the selected engine
    if args.stt_engine == "whisper":
        config_to_save.pop("vosk_model", None)
    elif args.stt_engine == "vosk":
        config_to_save.pop("whisper_model", None)
        config_to_save.pop("whisper_device", None)
        config_to_save.pop("whisper_compute_type", None)

    if args.tts_engine == "piper":
        config_to_save.pop("coqui_voice", None)
        config_to_save.pop("coqui_language", None)
        config_to_save.pop("coqui_speaker", None)
    elif args.tts_engine == "coqui":
        config_to_save.pop("piper_exe", None)
        config_to_save.pop("piper_voice", None)
        config_to_save.pop("piper_use_exe", None)

    config_dump_path = os.path.join(run_dir, "config_used.yaml")
    with open(config_dump_path, "w", encoding="utf-8") as f:
        yaml.dump(config_to_save, f, sort_keys=False)

    save_system_prompt(run_dir, prompt_label, system_prompt)

    # These ride along on every row so a stage is always interpretable on its
    # own: which stages carry a value, and what they include, depends on them.
    # Averaging across a mix of them is meaningless.
    fieldnames = [
        "ts_iso", "mode", "stt_engine", "tts_engine",
        "input_mode", "audio_pacing", "utterance_trigger",
        "item", "stage", "duration_ms",
        "cpu_percent", "ram_percent", "rss_mb",
        "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_name",
        "extra_json",
    ]

    trigger_on_endpoint = triggers_on_endpoint(args)

    # Signals the microphone source to stop capturing
    shutdown_event = threading.Event()

    # Prepare CSV
    csv_exists = os.path.exists(args.latency_csv)
    with open(args.latency_csv, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for item_index, audio_path in enumerate(args.audio, start=1):
            print("=" * 60)
            
            wav_path = None
            if args.input_mode == "mic":
                item_name = "live_mic"
                user_wav = os.path.join(run_dir, f"user_input_{item_index}_{item_name}.wav")
                audio_source = MicAudioSource(
                    save_path=user_wav,
                    chunk_ms=args.audio_chunk_ms,
                    stop_event=shutdown_event,
                )
                print(f"[INFO] Processing live microphone input (saving to {user_wav})...")
            else:
                audio_file = Path(audio_path)
                if not audio_file.exists():
                    raise FileNotFoundError(audio_path)

                item_name = audio_file.stem
                print(f"[INFO] Processing audio file: {audio_file} (pacing: {args.audio_pacing})")

                user_wav = os.path.join(run_dir, f"user_input_{item_index}_{audio_file.name}")
                try:
                    if os.path.abspath(audio_path) != os.path.abspath(user_wav):
                        shutil.copyfile(audio_path, user_wav)
                except Exception as e:
                    print(f"[WARN] Copy failed: {e}")

                wav_path = ensure_wav_mono_16k(audio_path, out_dir=run_dir, prefix=f"input_{item_index}")
                audio_source = FileAudioSource(
                    wav_path,
                    pacing=args.audio_pacing,
                    chunk_ms=args.audio_chunk_ms,
                    stop_event=shutdown_event,
                )

            out_wav = os.path.join(run_dir, f"assistant_{item_index}_{item_name}.wav")

            # Shared metrics dicts (single-writer per dict = thread-safe).
            # The *_stats entries hold the snapshot taken at that milestone.
            stt_metrics = {"stt_ms": None, "user_text": None, "stats": None}
            llm_metrics = {
                "llm_t0": None, "llm_ttfc_ms": None, "llm_ttft_ms": None,
                "first_chunk_chars": None,
                "full_assistant_text": "", "llm_chunk_count": 0,
                "ollama_stats": None,
                "ttft_stats": None, "ttfc_stats": None, "end_stats": None,
            }
            tts_metrics = {
                "tts_first_chunk_ms": None, "tts_first_chunk_t": None,
                "total_tts_time": 0.0,
                "first_chunk_stats": None, "end_stats": None,
            }

            # STT hands over utterances through a single-slot mailbox (only the
            # newest one matters); TTS chunks are a FIFO, so they stay a queue.
            mailbox = UtteranceMailbox()
            tts_queue = queue.Queue()

            # Opens the main thread's window, which spans the whole item.
            prime_cpu_percent()

            e2e_t0 = time.perf_counter()

            # Start worker threads
            stt_t = threading.Thread(
                target=stt_worker,
                args=(stt_engine, audio_source, mailbox, stt_metrics,
                      trigger_on_endpoint),
                daemon=True,
            )
            llm_t = threading.Thread(
                target=llm_worker,
                args=(llm_engine, mailbox, tts_queue, llm_metrics),
                daemon=True,
            )
            tts_t = threading.Thread(
                target=tts_worker,
                args=(tts_engine, tts_queue, out_wav, tts_metrics, args.playback),
                daemon=True,
            )

            stt_t.start()
            llm_t.start()
            tts_t.start()

            # Wait for completion or interrupt
            try:
                if args.input_mode == "file":
                    stt_t.join()
                    llm_t.join()
                    tts_t.join()
                else:
                    last_action_t = time.perf_counter()
                    while True:
                        time.sleep(1.0)
                        now = time.perf_counter()
                        
                        if llm_metrics.get("is_busy") or tts_metrics.get("is_busy"):
                            last_action_t = now
                        else:
                            last_action_t = max(
                                last_action_t,
                                stt_metrics.get("final_t") or e2e_t0,
                                tts_metrics.get("last_play_t") or e2e_t0
                            )
                            
                        if now - last_action_t > args.idle_timeout:
                            print(f"\n[INFO] Idle timeout reached ({args.idle_timeout}s). Stopping.")
                            shutdown_event.set()
                            break
            except KeyboardInterrupt:
                print("\n[INFO] Graceful shutdown triggered by user (Ctrl+C).")
                shutdown_event.set()

            # Let the workers drain before their metrics are read, otherwise the
            # response WAV may still be open while its duration is measured.
            for worker in (stt_t, llm_t, tts_t):
                worker.join(timeout=WORKER_SHUTDOWN_TIMEOUT_S)
                if worker.is_alive():
                    print(f"[WARN] {worker.name} did not stop within "
                          f"{WORKER_SHUTDOWN_TIMEOUT_S}s.")

            # -------- Collect metrics and write to CSV --------
            # Covers the item as a whole rather than a single stage.
            e2e_stats = collect_resource_snapshot()

            user_text = stt_metrics.get("user_text") or ""
            stt_ms = stt_metrics.get("stt_ms") or 0
            # Read after the workers finish: a mic source only knows how much
            # audio it captured once capturing has stopped.
            input_duration_ms = audio_source.duration_ms

            stt_rtf = round(stt_ms / input_duration_ms, 3) if input_duration_ms > 0 else 0.0
            stt_extra = {"input_duration_ms": input_duration_ms, "stt_rtf": stt_rtf}

            # Trailing silence is what an endpointer needs to see before it can
            # call the utterance over. Too little and the engine only finalizes
            # because the file ran out, a signal a live microphone never gets.
            speech_end_s = stt_metrics.get("speech_end_s")
            if speech_end_s is not None and input_duration_ms > 0:
                trailing_silence_ms = input_duration_ms - int(speech_end_s * 1000)
                stt_extra["trailing_silence_ms"] = trailing_silence_ms
                if (args.input_mode == "file"
                        and args.audio_pacing == PACING_REALTIME
                        and trailing_silence_ms < MIN_TRAILING_SILENCE_MS):
                    print(f"[WARN] Only {trailing_silence_ms} ms of silence after the last "
                          f"word; {MIN_TRAILING_SILENCE_MS} ms or more is needed for the "
                          f"endpointer to fire. This run measures the end-of-file path, "
                          f"which live microphone input never takes, so its TTFA is optimistic.")

            write_timing(writer, args, item_name, "stt", stt_ms,
                         stt_metrics.get("stats"), stt_extra)

            if not user_text:
                print("[Warn] No text recognized. Skipping to next file.")
                write_timing(writer, args, item_name, "e2e_response_ready",
                             int((time.perf_counter() - e2e_t0) * 1000),
                             e2e_stats,
                             {"input_duration_ms": input_duration_ms, "skipped": True})
                fcsv.flush()
                continue

            # LLM TTFT (first token) and TTFC (first chunk handed to TTS).
            # These differ by the time the chunker spends filling a chunk.
            llm_ttft_ms = llm_metrics.get("llm_ttft_ms")
            if llm_ttft_ms is not None:
                write_timing(writer, args, item_name, "llm_ttft", llm_ttft_ms,
                             llm_metrics.get("ttft_stats"))

            llm_ttfc_ms = llm_metrics.get("llm_ttfc_ms")
            if llm_ttfc_ms is not None:
                write_timing(writer, args, item_name, "llm_ttfc", llm_ttfc_ms,
                             llm_metrics.get("ttfc_stats"))

            # How long the chunker spent filling the opening chunk once the model
            # had started producing tokens. This is the slice a short opening
            # sentence is meant to remove.
            # The ttfc snapshot's CPU window is exactly this interval.
            if llm_ttfc_ms is not None and llm_ttft_ms is not None:
                write_timing(writer, args, item_name, "llm_first_chunk_fill",
                             llm_ttfc_ms - llm_ttft_ms,
                             llm_metrics.get("ttfc_stats"),
                             {"first_chunk_chars": llm_metrics.get("first_chunk_chars"),
                              "system_prompt": prompt_label})

            # TTS first chunk
            tts_first_chunk_ms = tts_metrics.get("tts_first_chunk_ms")
            if tts_first_chunk_ms is not None:
                write_timing(writer, args, item_name, "tts_first_chunk", tts_first_chunk_ms,
                             tts_metrics.get("first_chunk_stats"))

            # TTFA: from the moment the speaker stopped to the first audio out.
            # Blank whenever that instant is unknown -- no word timings from the
            # engine, or 'fast' pacing, where the audio does not advance at
            # wall-clock speed and no offset within it maps to a timestamp.
            speech_end_t = stt_metrics.get("speech_end_t")
            tts_first_t = tts_metrics.get("tts_first_chunk_t")
            if tts_first_t is not None and speech_end_t is not None:
                # Ends at the same instant as tts_first_chunk, so it shares that
                # stage's snapshot.
                write_timing(writer, args, item_name, "ttfa",
                             int((tts_first_t - speech_end_t) * 1000),
                             tts_metrics.get("first_chunk_stats"))

            # How long the STT took to decide the utterance was over, measured
            # from the actual end of speech. On file input this is the flush
            # after the stream ends; on a microphone it is the endpointer
            # waiting out the trailing silence, which dominates perceived
            # latency and is otherwise invisible.
            stt_final_t = stt_metrics.get("final_t")
            if stt_final_t is not None and speech_end_t is not None:
                write_timing(writer, args, item_name, "stt_endpoint_delay",
                             int((stt_final_t - speech_end_t) * 1000),
                             stt_metrics.get("stats"))

            # LLM server-side evaluation metrics (ground truth from Ollama)
            ollama_stats = llm_metrics.get("ollama_stats")
            if ollama_stats:
                # Reading the prompt and writing the answer are separate phases with
                # very different costs: prefill covers every prompt token in one pass,
                # decoding takes a pass per token. Ollama times them separately, so
                # they get a row each rather than one lumped LLM figure.
                #
                # Both arrive with the final message, once generation has finished.
                # The prompt was evaluated long before that, so no snapshot taken at
                # this point describes it; the one from the first token is the closest
                # boundary available.
                prompt_eval_ns = ollama_stats.get("prompt_eval_duration_ns", 0)
                if prompt_eval_ns:
                    write_timing(writer, args, item_name, "llm_prompt_eval",
                                 int(prompt_eval_ns / 1e6),
                                 llm_metrics.get("ttft_stats"),
                                 {"prompt_tokens": ollama_stats.get("prompt_eval_count", 0),
                                  "system_prompt": prompt_label})

                eval_dur_ns = ollama_stats.get("eval_duration_ns", 0)
                eval_count = ollama_stats.get("eval_count", 0)
                tokens_per_sec = round(eval_count / (eval_dur_ns / 1e9), 1) if eval_dur_ns > 0 else 0.0
                write_timing(writer, args, item_name, "llm_eval", int(eval_dur_ns / 1e6),
                             llm_metrics.get("end_stats"),
                             {"eval_tokens": eval_count,
                              "tokens_per_sec": tokens_per_sec,
                              "total_duration_ms": int(ollama_stats.get("total_duration_ns", 0) / 1e6)})

            # Log total TTS time. A response that synthesized nothing left this
            # worker's CPU window unopened, so its snapshot would read 0.0.
            write_timing(writer, args, item_name, "tts_total",
                         int(tts_metrics["total_tts_time"] * 1000),
                         tts_metrics.get("end_stats") if tts_metrics["total_tts_time"] > 0 else None)

            # Output audio duration and E2E summary. A mic session answers more
            # than once, so the reported duration covers every response WAV.
            response_wavs = [p for p in tts_metrics.get("out_wavs", []) if os.path.exists(p)]
            output_duration_ms = sum(wav_duration_ms(p) for p in response_wavs)
            full_reply = llm_metrics["full_assistant_text"].strip()

            write_timing(writer, args, item_name, "e2e_response_ready",
                         int((time.perf_counter() - e2e_t0) * 1000),
                         e2e_stats,
                         {"input_duration_ms": input_duration_ms,
                          "output_duration_ms": output_duration_ms,
                          "output_wav": response_wavs[0] if response_wavs else "",
                          "output_wav_count": len(response_wavs),
                          "full_text": full_reply,
                          "response_word_count": len(full_reply.split()) if full_reply else 0,
                          "response_char_count": len(full_reply),
                          "llm_chunk_count": llm_metrics["llm_chunk_count"]})
            print(f"[TTS] Stream finished. Saved to {', '.join(response_wavs) or '(no audio)'}.")

            # Flush CSV after each item to prevent data loss
            fcsv.flush()

            # --- Transcript Logging ---
            transcript_record = {
                "stt_text": user_text,
                "llm_text": full_reply,
            }

            # JSONL Logging
            jsonl_path = os.path.join(run_dir, "transcripts.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(transcript_record, ensure_ascii=False) + "\n")

            # YAML Logging
            yaml_path = os.path.join(run_dir, "transcripts.yaml")
            with open(yaml_path, "a", encoding="utf-8") as f_yaml:
                yaml.dump([transcript_record], f_yaml, sort_keys=False, allow_unicode=True)

            if args.input_mode == "file" and not args.keep_normalized:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

            # The stop signal is shared by every item, so a Ctrl+C or an idle
            # timeout ends the whole run instead of feeding empty audio to the
            # remaining files.
            if shutdown_event.is_set():
                print("[INFO] Stop requested, skipping any remaining items.")
                break

    stop_gpu_monitor()

    print(f"\nDone. Latency/resource log appended to: {args.latency_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
