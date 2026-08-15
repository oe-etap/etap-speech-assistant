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
import collections
import csv
import queue
import socket
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
from urllib.parse import urlparse

import soundfile as sf
try:
    import sounddevice as sd
    _HAS_SD = True
except (ImportError, OSError):
    # OSError, not just ImportError: the package can be installed while the
    # PortAudio shared library it binds to is missing, which is the normal state
    # of a headless server. File mode needs no audio device, so the pipeline
    # stays usable there and only mic input and playback report the lack.
    sd = None
    _HAS_SD = False

import aggregate_logs
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

# How often the nvidia-smi fallback refreshes its reading. A subprocess per
# tick is what keeps this slow.
GPU_SAMPLE_INTERVAL_S = 1.0

# How often the utilisation timeline is extended through NVML, which costs
# under a millisecond a call. The driver produces one sample per frame, about
# every 1/6 s, and holds them until they are collected, so this only has to run
# often enough not to fall behind that.
GPU_UTIL_SAMPLE_INTERVAL_S = 0.1

# Silence from the driver's per-process samples that means an idle device
# rather than a collection that arrived between two frames.
GPU_UTIL_IDLE_GAP_S = 0.5

# How much of the timeline to keep. At the driver's rate this is over an hour,
# which covers any window a run asks about -- including e2e_response_ready,
# which spans the whole session on a microphone.
GPU_UTIL_HISTORY_SAMPLES = 30000

# How long to wait for the pipeline workers to drain before reading their metrics
WORKER_SHUTDOWN_TIMEOUT_S = 10.0

# How long Vosk waits after the last word before calling an utterance over.
#
# Swept over 833 HeySQuAD recordings by vosk_endpoint_sweep.py, against
# vosk-model-small-en-us-0.15. Below 600 ms the endpointer starts firing inside
# sentences -- 0.69% of utterances at 500 ms, 2.95% at 300 ms -- and each of
# those sends half a question to the LLM, whose answer is then thrown away when
# the rest of the words arrive. Above 600 ms nothing improves: the one recording
# still split at 600 ms is a false start ("now", 1.7 s, then the question) that
# survives 1500 ms too.
#
# One number for every utterance is a deliberate simplification, and it is not
# free. The shipped configuration instead gates three waits on how complete the
# words already sound, which buys 220 ms on the fifth of utterances where the
# decoder is sure -- at three times the split rate, and 40 ms slower on average
# because its ceiling is 1.0 s. `500/600/600` is that mechanism with the ceiling
# brought down, and beats the shipped settings outright; it stays available
# rather than default because a split can be heard, the answer having often
# started playing before the rest of the sentence arrives to retract it.
#
# Read on a different corpus with care. These are read-aloud questions; a
# speaker composing a sentence as they go pauses longer.
DEFAULT_ENDPOINT_SILENCE_MS = (600, 600, 600)


def format_schedule(schedule):
    """How a schedule is written back to the user and into config_used.yaml."""
    if not max(schedule):
        return "model default"
    return str(schedule[0]) if len(set(schedule)) == 1 else "/".join(map(str, schedule))


def parse_endpoint_schedule(spec):
    """`600` for one wait, or `500/600/600` for Kaldi's rules 2, 3 and 4."""
    parts = tuple(int(p) for p in str(spec).split("/"))
    if len(parts) not in (1, 3):
        raise argparse.ArgumentTypeError(f"expected N or N/N/N, got {spec!r}")
    return parts * 3 if len(parts) == 1 else parts


# What the endpointer costs on top of the wait itself: the recognizer's own lag
# between the last word and the silence it scores, plus whatever is left of a
# chunk when the threshold is crossed, since it can only fire on a chunk
# boundary. Both terms are fitted over vosk_endpoint_sweep.py's output -- 32
# combinations of six chunk sizes from 50 to 250 ms with 200-1500 ms waits, over
# the 577 recordings it counts -- to
#
#     stt_endpoint_delay ~= wait + LAG_MS + LAG_CHUNK_SHARE * audio-chunk-ms
#
# fitted once for the median and once for the 99th percentile. Residuals stay
# within 35 ms in both cases, and the chunk term is linear across every size
# measured. Nothing here says it stays linear past 250 ms.
ENDPOINT_LAG_MS = 280
ENDPOINT_LAG_CHUNK_SHARE = 0.4
ENDPOINT_P99_LAG_MS = 390
ENDPOINT_P99_LAG_CHUNK_SHARE = 0.75

# The model's own settings gate three silence lengths on decoder confidence and
# so have no single wait to substitute here. Measured, they behave like this
# one: 1050 ms median and 1300 ms p99 at a 250 ms chunk, both of which the
# formulas above reproduce from 700 to within 30 ms.
STOCK_EQUIVALENT_SILENCE_MS = 700


def endpoint_delay_ms(endpoint_silence_ms, chunk_ms, worst_case=False):
    """What `stt_endpoint_delay` comes out at, typically or in the tail.

    A gated schedule is read at its ceiling, the wait that applies when the
    decoder is not persuaded the utterance is over. Its faster rules only ever
    subtract, so this is the bound the trailing-silence check needs.
    """
    wait_ms = max(endpoint_silence_ms) or STOCK_EQUIVALENT_SILENCE_MS
    if worst_case:
        return round(wait_ms + ENDPOINT_P99_LAG_MS + ENDPOINT_P99_LAG_CHUNK_SHARE * chunk_ms)
    return round(wait_ms + ENDPOINT_LAG_MS + ENDPOINT_LAG_CHUNK_SHARE * chunk_ms)


def min_trailing_silence_ms(args):
    """Silence an input file needs for the endpointer to fire before it ends.

    Below this the engine only finalizes because the file ran out, a signal a
    live microphone never gets, so the run reports a latency live input would
    never achieve. How long the endpointer takes varies from one utterance to
    the next by around 200 ms, so this is the 99th percentile rather than the
    median: a file cut to the median figure would fall short half the time.
    """
    return endpoint_delay_ms(args.vosk_endpoint_silence_ms, args.audio_chunk_ms,
                             worst_case=True)


# What hands an utterance to the LLM.
TRIGGER_ENDPOINT = "endpoint"        # the STT calling the utterance over
TRIGGER_END_OF_FILE = "end-of-file"  # the input running out


# ---------- Helpers ----------
# What is already known about a recording, from a metadata.csv beside the audio
# joined on the filename -- the layout the corpus ships in. Two things are taken
# from it: `question`, so the transcript log carries what the recording was
# supposed to say, and `speech_end_ms`, a VAD's reading of where the speech
# stops, which anchors the latency metrics. One CSV serves every recording in
# its directory, so it is read once per directory rather than once per file.
_metadata_by_dir = {}

USABLE_COLUMNS = ("question", "speech_end_ms")


def read_metadata(directory):
    """`filename` -> row, from `<directory>/metadata.csv`, or empty."""
    path = os.path.join(directory, "metadata.csv")
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {}
    except csv.Error as e:
        print(f"[WARN] {path} could not be read: {e}")
        return {}

    if not rows or "filename" not in rows[0]:
        if rows:
            print(f"[WARN] {path} has no 'filename' column, so none of it can be "
                  f"matched to an input file")
        return {}
    usable = [c for c in USABLE_COLUMNS if c in rows[0]]
    print(f"[INFO] Metadata for {len(rows)} recordings from {path}"
          f" ({', '.join(usable) if usable else 'nothing this run can use'})")
    return {r["filename"]: r for r in rows if r.get("filename")}


def metadata_for(audio_path):
    """The metadata row describing this recording, or an empty one."""
    directory = os.path.dirname(os.path.abspath(audio_path))
    if directory not in _metadata_by_dir:
        _metadata_by_dir[directory] = read_metadata(directory)
    return _metadata_by_dir[directory].get(os.path.basename(audio_path)) or {}


def ground_truth_text(row):
    """What this recording was supposed to say, or None if nothing says so."""
    return row.get("question") or None


def reference_speech_end_s(row):
    """Where a VAD put the end of the speech, in seconds into the recording.

    A better anchor for the latency metrics than the recognizer's own word
    timings, which are a by-product of decoding rather than a measurement of the
    audio, and which are missing entirely when the recognizer fails on a
    recording. Measured on the normalized mono 16 kHz audio, which is what the
    pipeline feeds the STT, so the offset lines up with the stream.
    """
    try:
        return float(row["speech_end_ms"]) / 1000.0
    except (KeyError, TypeError, ValueError):
        return None


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


def save_run_summary(run_dir, latency_csv):
    """Aggregate the log this run just wrote, into the folder it wrote it to.

    The CSV holds one row per stage per recording, which is the raw material and
    not the result: what the run measured is a distribution, and a folder left
    without its summary is one that every later reader has to remember to run
    aggregate_logs.py over. So the run does it itself, over the log it wrote to
    and nothing else - which is every row of that file, including any earlier
    run's when --latency-csv points them all at one.

    Nothing here is allowed to fail the run. The recordings have been processed
    and their measurements are on disk by this point; an unwritable folder or a
    log with nothing in it yet is worth a warning, not the loss of the run.
    """
    try:
        report = aggregate_logs.summarize_run(run_dir, logs=latency_csv)
    except Exception as e:
        print(f"[WARN] Could not summarize the run log: {e}")
        return None

    if report is None:
        print(f"[WARN] No aggregate summary written: '{latency_csv}' holds no readable rows.")
        return None

    # The same warnings the script prints, and for the same reason: they are
    # what says whether the numbers just written can be read at face value.
    for warning in report.warnings:
        print(warning if warning.startswith(" ") else f"[WARN] {warning}")

    for path in report.files.values():
        print(f"Aggregate summary saved to: {path}")
    return report


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
    """Read the point-in-time GPU counters through NVML.

    NVML is an in-process library call rather than a subprocess, so this is
    cheap enough to run at a stage boundary. Fields the driver does not
    support are blanked individually. Utilisation is not among them: it
    describes a span rather than an instant and is accumulated by the sampler.
    """
    stats = dict(_GPU_BLANK_STATS)
    handle = _nvml_handle
    if handle is None:
        return stats

    name = _nvml_value(lambda: pynvml.nvmlDeviceGetName(handle))
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    stats["gpu_name"] = name

    # NVML's original memory query defines `used` as total minus free, which
    # counts the framebuffer the driver reserves for itself -- 2192 MiB of the
    # 32 GB vGPU here, present with nothing running at all. nvidia-smi does not
    # show that in its memory figure, so the v2 query, which reports the
    # reservation separately, is what keeps the two sources of this column
    # agreeing.
    mem = None
    if hasattr(pynvml, "nvmlMemory_v2"):
        mem = _nvml_value(
            lambda: pynvml.nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2), None)
    if mem is None:
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


# The utilisation timeline, as (instant on this process's clock, percent). Each
# sample describes the interval ending at its instant, which is how the driver
# spaces its own. Appended by the sampler and read whole under the lock, since
# a deque being appended to cannot be iterated safely.
_gpu_util_samples = collections.deque(maxlen=GPU_UTIL_HISTORY_SAMPLES)
_gpu_util_lock = threading.Lock()

# The newest driver timestamp already collected, in the driver's own epoch
# microseconds. Only the sampler touches it.
_gpu_util_last_seen = 0

# Whether the driver will hand over its per-process samples, decided once at
# startup. Where it will not, the sampler falls back to the utilisation counter,
# which is a rolling average and carries the lag that goes with it.
_gpu_util_sampled = False


def _add_gpu_util_sample(instant, util):
    """Append one point of the timeline, dropping what will not convert."""
    try:
        util = float(util)
    except (TypeError, ValueError):
        return
    with _gpu_util_lock:
        _gpu_util_samples.append((instant, util))


def _sample_process_util():
    """Collect the driver's own timestamped samples since the last collection.

    These are what makes a stage's figure describe that stage. The samples are
    stamped by the driver at the instant the work happened, arrive within a
    frame of it, and stop as soon as it does -- where the utilisation counter
    reports a rolling average that starts late, outlives the work by seconds and
    so lands on whichever stage happens to be running by then.

    Utilisation is reported per process. Summing across them gives the device,
    which is what this column has always described.
    """
    global _gpu_util_last_seen
    wall, perf = time.time(), time.perf_counter()
    try:
        samples = pynvml.nvmlDeviceGetProcessUtilization(_nvml_handle, _gpu_util_last_seen)
    except pynvml.NVMLError:
        # Nothing new, which between two frames means nothing yet and over a
        # longer silence means nothing at all: the driver emits a sample every
        # frame for as long as any process holds a context. An empty timeline is
        # the same answer at the start of a run, before anything has loaded.
        if not _gpu_util_samples or perf - _gpu_util_samples[-1][0] > GPU_UTIL_IDLE_GAP_S:
            _add_gpu_util_sample(perf, 0.0)
        return

    totals = {}
    for sample in samples:
        if sample.timeStamp > _gpu_util_last_seen:
            totals[sample.timeStamp] = totals.get(sample.timeStamp, 0) + sample.smUtil
    if not totals:
        return
    _gpu_util_last_seen = max(totals)
    for stamp in sorted(totals):
        # The driver stamps these against the wall clock; the windows they are
        # measured into are on the monotonic one.
        _add_gpu_util_sample(perf + (stamp / 1e6 - wall), min(100.0, totals[stamp]))


def _supports_process_util():
    """Whether the driver will hand over per-process utilisation samples.

    'Not found' is the answer for a device with nothing running on it, which
    says nothing about support and is why it counts as a yes here: a run asks
    this before its own models are loaded. A driver without the feature reports
    it as unsupported instead.
    """
    try:
        pynvml.nvmlDeviceGetProcessUtilization(_nvml_handle, 0)
    except pynvml.NVMLError as e:
        return getattr(e, "value", None) == pynvml.NVML_ERROR_NOT_FOUND
    except Exception:
        return False
    return True


def _sample_gpu_util():
    """Extend the timeline from whichever source is available.

    On the nvidia-smi path this also refreshes the published levels, since that
    source costs a subprocess and must stay away from the workers.
    """
    if _gpu_util_sampled:
        _sample_process_util()
    elif _nvml_handle is not None:
        _add_gpu_util_sample(time.perf_counter(), _nvml_value(
            lambda: pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle).gpu))
    else:
        # A failed refresh keeps the previous reading.
        _refresh_gpu_stats()
        _add_gpu_util_sample(time.perf_counter(), _gpu_stats["gpu_util_percent"])


def gpu_util_over(window):
    """Mean utilisation across a stage, from the samples covering it.

    window is the (start, end) the stage was measured between. Each sample
    counts for the part of its own interval that falls inside, so the figure is
    an average over time rather than over samples.

    Blank where the window is not covered: before the sampler's first reading,
    or where it reaches further back than the retained timeline. The tail of a
    window can outrun the samples by up to one frame, the last of them arriving
    after the stage they describe has ended; the mean then covers the part that
    was reported, which is the whole stage bar that frame.
    """
    if not window:
        return ""
    start, end = window
    with _gpu_util_lock:
        samples = list(_gpu_util_samples)
    if len(samples) < 2 or samples[0][0] > start:
        return ""

    total, covered = 0.0, 0.0
    for (opened, _), (closed, util) in zip(samples, samples[1:]):
        overlap = min(closed, end) - max(opened, start)
        if overlap > 0:
            total += util * overlap
            covered += overlap
    return round(total / covered, 1) if covered > 0 else ""


class _GpuSampler(threading.Thread):
    """Background thread keeping the utilisation integral fed.

    Utilisation is the one GPU field that describes a span rather than an
    instant: the driver reports its own rolling average over the last fraction
    of a second, so reading it at a stage boundary describes whatever the card
    was doing around that boundary, which for overlapping stages is a different
    stage's work. Sampling it here, off the measured path, is what lets each row
    report the mean over its own span instead.

    On the nvidia-smi path the same tick also refreshes the published readings,
    since that source costs a subprocess and must stay away from the workers.
    """

    def __init__(self, interval):
        super().__init__(daemon=True, name="gpu-sampler")
        self._interval = interval
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(self._interval):
            _sample_gpu_util()

    def stop(self):
        self._stop_event.set()


_gpu_sampler = None


def start_gpu_monitor():
    """Begin collecting GPU statistics, preferring NVML over nvidia-smi.

    The point-in-time fields are read in process at each stage boundary where
    NVML is available, and come from the sampler's published reading where the
    nvidia-smi fallback is in use instead. Either way the sampler runs, since
    utilisation has to be integrated over the stage rather than probed at its
    end. Neither source being usable leaves the GPU columns blank.
    """
    global _nvml_handle, _gpu_sampler, _gpu_util_sampled
    if _gpu_sampler is not None:
        return

    interval = None
    if _HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _gpu_util_sampled = _supports_process_util()
            interval = GPU_UTIL_SAMPLE_INTERVAL_S
        except Exception:
            _nvml_handle = None

    if interval is None:
        if not _refresh_gpu_stats():
            return
        interval = GPU_SAMPLE_INTERVAL_S

    # One reading before the thread starts, so that a window opened during the
    # first tick already has an integral to be measured against. The nvidia-smi
    # path would otherwise leave the first second of a run unmeasurable.
    _sample_gpu_util()
    _gpu_sampler = _GpuSampler(interval)
    _gpu_sampler.start()


def stop_gpu_monitor():
    """Stop the sampler and release NVML.

    In that order: the sampler reads the NVML handle on every tick.
    """
    global _nvml_handle, _gpu_sampler, _gpu_util_sampled, _gpu_util_last_seen
    if _gpu_sampler is not None:
        _gpu_sampler.stop()
        _gpu_sampler.join(timeout=GPU_SAMPLE_INTERVAL_S * 2)
        _gpu_sampler = None
    if _nvml_handle is not None:
        _nvml_handle = None
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    _gpu_util_sampled = False
    _gpu_util_last_seen = 0
    with _gpu_util_lock:
        _gpu_util_samples.clear()


# The Ollama server holds the model in a runner subprocess of its own, so this
# process's RSS says nothing about what the LLM costs: on a CPU run the weights
# are the largest allocation on the machine and not one page of them appears in
# rss_mb. These are handles on the server and its runners, kept between
# snapshots so that a reading is a few /proc reads rather than a process scan.
# Rebound wholesale rather than mutated, so the workers reading them need no
# lock, as with _gpu_stats above.
_llm_procs = []

# Whether the configured server is on this machine, and so has a process here
# to be measured at all
_llm_local = False

# When _llm_procs was last rebuilt, and how often that may be retried while the
# set looks incomplete
_llm_procs_scanned_at = 0.0
LLM_PROC_RESCAN_S = 1.0

# What Ollama says the loaded model itself occupies on the GPU. Read once at
# startup: /api/ps is an HTTP round trip and has no business on a stage
# boundary. It is the smaller of the two VRAM figures -- see _read_llm_vram_mb.
_llm_model_vram_mb = ""


def _is_local_url(url):
    """Whether the URL addresses this machine, and so a process we could find."""
    host = (urlparse(url).hostname or "").lower()
    if host in ("", "localhost", "::1", "0.0.0.0"):
        return True
    return host.startswith("127.") or host == socket.gethostname().lower()


def _find_ollama_server():
    """Return the local `ollama serve` process, or None where it is not unambiguous.

    A scan of every process, which is why this is kept away from the stage
    boundaries. The listening port would identify the server exactly and cannot
    be used: mapping a socket to the process holding it needs read access to
    that process's descriptors, and the packaged service runs as its own user.
    More than one server is therefore indistinguishable, and reporting the
    memory of the wrong one is worse than reporting none.
    """
    servers = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        cmdline = proc.info["cmdline"] or []
        if len(cmdline) < 2 or cmdline[1] != "serve":
            continue
        if (proc.info["name"] or "").lower() != "ollama" and os.path.basename(cmdline[0]) != "ollama":
            continue
        servers.append(proc)
    return servers[0] if len(servers) == 1 else None


def _rebuild_llm_procs():
    """Find the Ollama server and the runner processes holding the model."""
    global _llm_procs, _llm_procs_scanned_at
    _llm_procs_scanned_at = time.monotonic()
    server = _find_ollama_server()
    if server is None:
        _llm_procs = []
        return
    try:
        _llm_procs = [server, *server.children(recursive=True)]
    except psutil.Error:
        _llm_procs = []


def _sum_llm_rss():
    """Total RSS over the cached handles, forgetting the ones that have died.

    Returns (bytes, processes read). Summing across processes counts pages
    shared between them twice; the server is tens of MB against a runner's
    gigabytes, and the figure that would not double count -- PSS -- needs the
    ptrace access the packaged service does not grant to the user running this.
    """
    global _llm_procs
    total, alive = 0, []
    for proc in _llm_procs:
        try:
            total += proc.memory_info().rss
            alive.append(proc)
        except psutil.Error:
            pass
    _llm_procs = alive
    return total, len(alive)


def start_llm_memory_monitor(url, placement):
    """Begin accounting for the memory the LLM holds outside this process.

    Call once the model is loaded: the runner holding the weights is started by
    the first request, so before warmup() there is nothing there to find.
    placement is report_llm_placement()'s reading of /api/ps, or None where the
    server could not be asked.
    """
    global _llm_local, _llm_model_vram_mb
    if placement is not None:
        _llm_model_vram_mb = round(placement["size_vram_bytes"] / (1024 * 1024))
    _llm_local = _HAS_PSUTIL and _is_local_url(url)
    if not _llm_local:
        return
    _rebuild_llm_procs()
    if not _llm_procs:
        print("[WARN] No single local 'ollama serve' process to attribute memory "
              "to; llm_rss_mb will be blank")
    elif len(_llm_procs) < 2:
        # The server on its own, with no runner under it -- which is what a
        # server inside a container looks like from out here, its children
        # living in a pid namespace of their own.
        print("[WARN] Found the Ollama server but no runner beneath it; "
              "llm_rss_mb will not account for the model")


def _read_llm_vram_mb():
    """GPU memory the driver attributes to the LLM's processes.

    Read from NVML over the same pids llm_rss_mb sums, so the two describe the
    same processes. This is the larger of the two VRAM figures and the one
    nvidia-smi shows: a runner that has loaded a 2467 MiB model appears here at
    2992 MiB, the difference being the CUDA context, the kernels and the
    libraries the runtime brings with it. llm_model_vram_mb carries what Ollama
    says the model alone occupies, which is what decides whether it fits.

    Blank where NVML is not the source of the GPU columns, or where the driver
    will not attribute memory per process -- which happens on some cards and
    under some virtualization, and cannot be told apart from an idle GPU.
    """
    if _nvml_handle is None:
        return ""
    procs = _nvml_value(lambda: pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml_handle), None)
    if not procs:
        return ""
    pids = {proc.pid for proc in _llm_procs}
    ours = [proc for proc in procs if proc.pid in pids]
    if any(proc.usedGpuMemory is None for proc in ours):
        return ""
    # No entry for our pids is a model that is not on the GPU at all, which is
    # zero rather than unknown: the driver answered, and did not name them.
    return round(sum(proc.usedGpuMemory for proc in ours) / (1024 * 1024))


def read_llm_memory():
    """What the LLM holds, in host RAM and on the GPU, as snapshot fields.

    llm_rss_mb covers the Ollama server and the runners holding the model, and
    is blank where the server is not on this machine or no process here could be
    identified as it.
    """
    stats = {
        "llm_rss_mb": "",
        "llm_vram_mb": "",
        "llm_model_vram_mb": _llm_model_vram_mb,
    }
    if not _llm_local:
        return stats

    total, alive = _sum_llm_rss()
    # A runner is torn down when keep_alive expires, and the next request starts
    # a fresh one under a new pid. Both halves of that appear here as a set with
    # no runner left in it, and only a scan recovers it. Rate limited because a
    # model that is not resident stays that way until something asks for it, and
    # a scan at every stage boundary would be paid for in the latency figures.
    if alive < 2 and time.monotonic() - _llm_procs_scanned_at >= LLM_PROC_RESCAN_S:
        _rebuild_llm_procs()
        total, alive = _sum_llm_rss()
    if alive:
        stats["llm_rss_mb"] = round(total / (1024 * 1024), 1)
        stats["llm_vram_mb"] = _read_llm_vram_mb()
    return stats


# Where each worker's utilisation window was opened, keyed by thread the way
# psutil keys its own cpu_percent state, so that concurrent stages do not close
# each other's windows.
_stage_window = threading.local()


def prime_cpu_percent():
    """Open the rate measurement windows on the calling thread.

    psutil.cpu_percent(interval=None) reports load since the calling thread's
    own previous call, its state being keyed by thread id. GPU utilisation is
    given the same treatment against the sampler's integral, so both columns
    describe the same span. Call this when a stage starts; the snapshot taken
    when that stage ends then covers exactly that stage.
    """
    _open_gpu_util_window()
    if not _HAS_PSUTIL:
        return
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        pass


def _open_gpu_util_window():
    """Mark where the calling thread's utilisation window starts."""
    _stage_window.gpu = time.perf_counter()


def _close_gpu_util_window():
    """Return the window this thread just finished, and open the next one.

    The window rather than a figure: the samples covering its last frame have
    not necessarily arrived yet, and the row is not written until the item is
    over. Resolving it there costs nothing and covers the whole stage.
    """
    opened = getattr(_stage_window, "gpu", None)
    now = time.perf_counter()
    _stage_window.gpu = now
    return None if opened is None else (opened, now)


def collect_resource_snapshot():
    """Return the resource snapshot for the stage that has just finished.

    Call at a stage boundary, from the thread that ran the stage and opened its
    window with prime_cpu_percent(). cpu_percent then covers that stage alone,
    and gpu_util_window carries the same span for the utilisation the row is
    written with; both reopen for whatever the thread measures next. ram_percent
    and rss_mb are read directly, and so are the remaining GPU fields, from NVML
    or, where that is unavailable, from the sampler's latest nvidia-smi reading.
    Missing psutil/NVML/nvidia-smi leaves blanks.

    rss_mb covers this process alone, which is the pipeline without its LLM: the
    model lives in the Ollama server's runner and is accounted for separately by
    the llm_ fields.
    """
    stats = {
        "cpu_percent": "",
        "ram_percent": "",
        "rss_mb": "",
        **read_llm_memory(),
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
    # Not a reading but the span to average one over, resolved when the row is
    # written; see _close_gpu_util_window().
    stats["gpu_util_window"] = _close_gpu_util_window()
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
        "llm_rss_mb": stats.get("llm_rss_mb", ""),
        "llm_vram_mb": stats.get("llm_vram_mb", ""),
        "llm_model_vram_mb": stats.get("llm_model_vram_mb", ""),
        "gpu_util_percent": gpu_util_over(stats.get("gpu_util_window")),
        "gpu_mem_used_mb": stats.get("gpu_mem_used_mb", ""),
        "gpu_mem_total_mb": stats.get("gpu_mem_total_mb", ""),
        "gpu_name": stats.get("gpu_name", ""),
        "extra_json": json.dumps(extra or {}, ensure_ascii=True),
    })


# ---------- Pipeline Workers ----------
def stt_worker(stt_engine, audio_source, mailbox, stt_metrics,
               trigger_on_endpoint=False, reference_speech_end_s=None):
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

    reference_speech_end_s, when a metadata.csv supplies one, replaces the
    engine's own word timing as that anchor. It is the better of the two: a
    measurement of the audio rather than a by-product of decoding it, the same
    for every finalized result rather than moving with each, and present even
    where the recognizer heard nothing. Which one was used is reported on the
    `stt` row, since runs anchored differently are not comparable.
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

            if reference_speech_end_s is not None:
                speech_end_s = reference_speech_end_s
                stt_metrics["speech_end_source"] = "metadata"
            else:
                speech_end_s = result.get("speech_end_s")
                stt_metrics["speech_end_source"] = "stt_word_timings"
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
                raise ImportError(
                    "playback needs sounddevice and the PortAudio library it "
                    "binds to (Debian/Ubuntu: apt install libportaudio2)"
                )
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


def report_llm_placement(llm_engine, model_name):
    """Print where Ollama put the model, and warn when it is not all on the GPU.

    A model that does not fit is not an error to Ollama: the layers that overflow
    run on the CPU, with nothing in the log to say so. It is only reported, since a
    partly-offloaded model is a configuration somebody may well mean to measure.

    Returns the reading, which is also where llm_vram_mb comes from, or None
    where the server could not be asked.
    """
    placement = llm_engine.placement()
    if placement is None:
        print("[WARN] Could not read Ollama's model placement; GPU residency unverified")
        return None

    fraction = placement["vram_fraction"]
    size_gb = placement["size_bytes"] / 1e9
    print(f"[INFO] Ollama model {model_name}: {size_gb:.1f} GB resident, "
          f"{fraction * 100:.0f}% on GPU")
    if fraction < 0.999:
        print(f"[WARN] {(1 - fraction) * 100:.0f}% of {model_name} is on the CPU. "
              f"Generation runs several times slower than a GPU-resident model. "
              f"Lower --llm-num-ctx, free VRAM, or pick a smaller model.")
    return placement


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
    parser.add_argument("--no-summary", dest="summary", action="store_false",
                        help="Skip the aggregate report the run otherwise leaves in its own "
                             "folder. The CSV is written either way, and aggregate_logs.py "
                             "produces the same report from it afterwards.")
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
    parser.add_argument("--vosk-endpoint-silence-ms", type=parse_endpoint_schedule,
                        default=DEFAULT_ENDPOINT_SILENCE_MS,
                        help="Silence after the last word before Vosk calls the utterance "
                             "over. Lower answers sooner but risks firing on a breath and "
                             "sending half a sentence to the LLM. `0` keeps whatever the "
                             "model ships with; `500/600/600` restores its habit of leaving "
                             "early when the words already parse as a finished utterance. "
                             "See vosk_endpoint_sweep.py for how the default was arrived at.")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model name")
    parser.add_argument("--whisper-device", choices=["cpu", "cuda"], default=None, help="Device for faster-whisper")
    parser.add_argument("--whisper-compute-type", default=None, help="Compute type (int8, float16, etc.)")

    # TTS options
    parser.add_argument("--piper-exe", default="./piper/piper.exe", help="Path to Piper executable (piper or piper.exe)")
    parser.add_argument("--piper-voice", default="./piper/en_US-lessac-medium.onnx", help="Path to English Piper .onnx voice")
    parser.add_argument("--piper-use-exe", action="store_true", help="Use Piper CLI executable instead of Python API (slower, spawns subprocess per chunk)")
    parser.add_argument("--piper-device", choices=["cpu", "cuda"], default=None,
                        help="ONNX Runtime execution provider for the Piper voice model. "
                             "Defaults to the --mode preset. 'cuda' needs an onnxruntime-gpu "
                             "matching the GPU's compute capability, and is refused rather "
                             "than silently served from the CPU. Ignored with --piper-use-exe.")
    parser.add_argument("--coqui-voice", default="xtts_v2", help="Coqui voice model key")
    parser.add_argument("--coqui-language", default="en", help="Language code for Coqui")
    parser.add_argument("--coqui-speaker", default="Daisy Studious", help="Speaker id for Coqui")
    parser.add_argument("--coqui-device", choices=["cpu", "cuda"], default=None,
                        help="Device for Coqui XTTS-v2, following the --mode preset when "
                             "unset. Unlike Piper this is not a tuning knob: XTTS-v2 is "
                             "autoregressive and runs slower than real time on a CPU. "
                             "Refused rather than silently served from the CPU.")

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
    parser.add_argument("--llm-num-ctx", type=int, default=1024,
                        help="KV cache size in tokens. The server otherwise picks "
                             "from the VRAM present, which on a 32 GB card means "
                             "32768 -- 12 GB of cache for a single-turn exchange "
                             "that uses a few hundred tokens. Costs nothing in "
                             "speed while the model fits; buys the headroom that "
                             "keeps a large one fitting. 0 follows the server.")
    parser.add_argument("--llm-keep-alive", type=str, default="30m",
                        help="How long Ollama holds the model in VRAM after a "
                             "request, in its duration syntax ('30m', '-1' for "
                             "forever). A reload costs about 8 s for an 8B q4_K_M "
                             "and lands inside a measured utterance. Pair a long "
                             "value with OLLAMA_MAX_LOADED_MODELS=1 so the "
                             "previous model is evicted rather than crowding the "
                             "next one off the GPU. Empty follows the server.")
    parser.add_argument("--llm-num-gpu", type=int, default=None,
                        help="Layers offloaded to the GPU. Unset lets Ollama fit "
                             "the model itself. 99 forces every layer, which turns "
                             "a model that does not fit into a loud failure "
                             "instead of a quiet CPU spill several times slower.")
    parser.add_argument("--llm-num-batch", type=int, default=None,
                        help="Prompt-processing batch size, default 512 server-side. "
                             "Prompts here are far shorter than one batch, so this "
                             "is a knob for completeness rather than for latency.")
    parser.add_argument("--llm-num-thread", type=int, default=None,
                        help="CPU threads for the Ollama runner. Unset takes one per core.")

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

    # A YAML value reaches argparse through set_defaults, which skips the `type`
    # conversion a command line goes through, so the schedule is normalized here
    # rather than in one of the two places it can arrive from.
    args.vosk_endpoint_silence_ms = parse_endpoint_schedule(args.vosk_endpoint_silence_ms) \
        if not isinstance(args.vosk_endpoint_silence_ms, tuple) \
        else args.vosk_endpoint_silence_ms

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
    if args.piper_device is None:
        args.piper_device = "cuda" if args.mode == "gpu" else "cpu"
    if args.coqui_device is None:
        args.coqui_device = "cuda" if args.mode == "gpu" else "cpu"

    # Any one-off setup cost happens here, before a stage is ever timed.
    start_gpu_monitor()

    # Pre-load (warmup) models so their initialization time isn't counted
    # in the latency of the first file
    print("[INFO] Pre-loading AI models (STT, LLM, TTS)...")

    # Initialize STT engine
    if args.stt_engine == "vosk":
        stt_engine = VoskEngine(args.vosk_model, args.vosk_endpoint_silence_ms)
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
        # 0 and "" are how a command line asks for "leave the server alone",
        # there being no way to pass a null through argparse.
        num_ctx=args.llm_num_ctx or None,
        keep_alive=args.llm_keep_alive or None,
        num_gpu=args.llm_num_gpu,
        num_batch=args.llm_num_batch,
        num_thread=args.llm_num_thread,
    )
    llm_engine.warmup()
    placement = report_llm_placement(llm_engine, args.ollama_model)
    start_llm_memory_monitor(args.ollama_url, placement)

    # Initialize TTS engine
    if args.tts_engine == "piper":
        tts_engine = PiperEngine(
            voice_path=args.piper_voice,
            exe_path=args.piper_exe,
            use_exe=args.piper_use_exe,
            device=args.piper_device,
        )
    elif args.tts_engine == "coqui":
        tts_engine = CoquiEngine(
            model_name=args.coqui_voice,
            language=args.coqui_language,
            speaker=args.coqui_speaker,
            device=args.coqui_device,
        )
    tts_engine.warmup()

    run_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    if not args.latency_csv:
        args.latency_csv = os.path.join(run_dir, f"latency_log_{timestamp}.csv")

    config_to_save = vars(args).copy()
    config_to_save["vosk_endpoint_silence_ms"] = format_schedule(args.vosk_endpoint_silence_ms)

    # Filter out settings that are not used by the selected engine
    if args.stt_engine == "whisper":
        config_to_save.pop("vosk_model", None)
        config_to_save.pop("vosk_endpoint_silence_ms", None)
    elif args.stt_engine == "vosk":
        config_to_save.pop("whisper_model", None)
        config_to_save.pop("whisper_device", None)
        config_to_save.pop("whisper_compute_type", None)

    if args.tts_engine == "piper":
        config_to_save.pop("coqui_voice", None)
        config_to_save.pop("coqui_language", None)
        config_to_save.pop("coqui_speaker", None)
        config_to_save.pop("coqui_device", None)
    elif args.tts_engine == "coqui":
        config_to_save.pop("piper_exe", None)
        config_to_save.pop("piper_voice", None)
        config_to_save.pop("piper_use_exe", None)
        config_to_save.pop("piper_device", None)

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
        "cpu_percent", "ram_percent", "rss_mb", "llm_rss_mb",
        "llm_vram_mb", "llm_model_vram_mb",
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
                # Live speech has nothing describing it in advance: no text it
                # was supposed to be, and no measured end to anchor on.
                item_filename, ori_text, ref_speech_end_s = None, None, None
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
                item_filename = audio_file.name
                metadata = metadata_for(audio_path)
                ori_text = ground_truth_text(metadata)
                ref_speech_end_s = reference_speech_end_s(metadata)
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

            # Setting up a recognizer is not recognition: doing it here keeps
            # its cost out of both windows opened below, the way the model load
            # is kept out by pre-loading before the loop.
            stt_engine.warmup()

            # Opens the main thread's window, which spans the whole item.
            prime_cpu_percent()

            e2e_t0 = time.perf_counter()

            # Start worker threads
            stt_t = threading.Thread(
                target=stt_worker,
                args=(stt_engine, audio_source, mailbox, stt_metrics,
                      trigger_on_endpoint, ref_speech_end_s),
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
            # Which reading of the end of speech everything anchored on. Runs
            # that differ here are measuring from different instants, so the
            # figures must not be pooled across them.
            if stt_metrics.get("speech_end_source"):
                stt_extra["speech_end_source"] = stt_metrics["speech_end_source"]
            if speech_end_s is not None and input_duration_ms > 0:
                trailing_silence_ms = input_duration_ms - int(speech_end_s * 1000)
                stt_extra["trailing_silence_ms"] = trailing_silence_ms
                needed_ms = min_trailing_silence_ms(args)
                if (args.input_mode == "file"
                        and args.audio_pacing == PACING_REALTIME
                        and args.stt_engine == "vosk"   # the only engine that endpoints
                        and trailing_silence_ms < needed_ms):
                    print(f"[WARN] Only {trailing_silence_ms} ms of silence after the last "
                          f"word; {needed_ms} ms or more is needed for the endpointer to "
                          f"fire at --vosk-endpoint-silence-ms "
                          f"{format_schedule(args.vosk_endpoint_silence_ms)}. This run "
                          f"measures the end-of-file path, which live microphone input "
                          f"never takes, so its TTFA is optimistic.")

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
                "filename": item_filename,
                "ori_text": ori_text,
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

    # Last, and outside the CSV's `with`: the log has to be closed and complete
    # before anything reads it back.
    if args.summary:
        save_run_summary(run_dir, args.latency_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
