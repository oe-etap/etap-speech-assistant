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
from llm_engine import DEFAULT_CHUNK_MAX_CHARS, OllamaEngine
from tts_engines import PiperEngine, CoquiEngine

# Optional module-level imports
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# nvidia-smi cache (timestamp, stats_dict)
_gpu_cache = (0.0, None)
_GPU_CACHE_TTL = 2.0

# How long to wait for the pipeline workers to drain before reading their metrics
WORKER_SHUTDOWN_TIMEOUT_S = 10.0


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


def wav_duration_ms(wav_path):
    with sf.SoundFile(wav_path) as wav:
        return int((len(wav) / wav.samplerate) * 1000)


def _query_gpu_stats():
    """Query nvidia-smi with caching to avoid subprocess overhead on every call."""
    global _gpu_cache
    now = time.monotonic()
    if _gpu_cache[1] is not None and (now - _gpu_cache[0]) < _GPU_CACHE_TTL:
        return _gpu_cache[1]

    gpu_stats = {
        "gpu_util_percent": "",
        "gpu_mem_used_mb": "",
        "gpu_mem_total_mb": "",
        "gpu_name": "",
    }
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            first_gpu = result.stdout.strip().splitlines()[0]
            name, util, mem_used, mem_total = [part.strip() for part in first_gpu.split(",", 3)]
            gpu_stats["gpu_name"] = name
            gpu_stats["gpu_util_percent"] = util
            gpu_stats["gpu_mem_used_mb"] = mem_used
            gpu_stats["gpu_mem_total_mb"] = mem_total
    except Exception:
        pass

    _gpu_cache = (now, gpu_stats)
    return gpu_stats


def collect_resource_snapshot():
    """Best-effort resource snapshot. Missing psutil/nvidia-smi leaves blanks."""
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

    stats.update(_query_gpu_stats())
    return stats


def write_timing(writer, args, item, stage, duration_ms, extra=None):
    stats = collect_resource_snapshot()
    writer.writerow({
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "stt_engine": args.stt_engine,
        "tts_engine": args.tts_engine,
        "item": item,
        "stage": stage,
        "duration_ms": int(duration_ms),
        "cpu_percent": stats["cpu_percent"],
        "ram_percent": stats["ram_percent"],
        "rss_mb": stats["rss_mb"],
        "gpu_util_percent": stats["gpu_util_percent"],
        "gpu_mem_used_mb": stats["gpu_mem_used_mb"],
        "gpu_mem_total_mb": stats["gpu_mem_total_mb"],
        "gpu_name": stats["gpu_name"],
        "extra_json": json.dumps(extra or {}, ensure_ascii=True),
    })


# ---------- Pipeline Workers ----------
def stt_worker(stt_engine, audio_source, mailbox, stt_metrics, input_mode="file"):
    """Feed audio to the STT engine and hand finalized utterances to the LLM.

    - In 'mic' mode: every finalized utterance is handed over immediately, so a
      new utterance can interrupt an answer that is still being generated.
    - In 'file' mode: the LLM is triggered once, at the end of the file.

    Partial results are printed for visibility but never trigger generation:
    starting on a partial costs a full generation whenever the guess is wrong,
    which is what made the earlier experiments slower rather than faster.

    Records speech_end_t on every finalized result: this is the anchor for the
    TTFA metric a user actually perceives, which starts when they stop talking
    rather than when processing begins.
    """
    try:
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

            stt_metrics["last_speech_t"] = time.perf_counter()
            stt_metrics["speech_end_t"] = audio_source.speech_end_t()
            stt_metrics["user_text"] = accumulated_final

            if input_mode == "mic":
                mailbox.put(accumulated_final)

        if input_mode == "file" and accumulated_final:
            mailbox.put(accumulated_final)

        t1 = time.perf_counter()
        stt_metrics["stt_ms"] = int((t1 - t0) * 1000)
        stt_metrics["user_text"] = accumulated_final
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

            print("[LLM] Starting stream...")
            llm_t0 = time.perf_counter()
            llm_metrics["llm_t0"] = llm_t0

            for chunk_data in llm_engine.generate_stream(text):
                if chunk_data.get("ollama_stats"):
                    llm_metrics["ollama_stats"] = chunk_data["ollama_stats"]

                first_token_t = chunk_data.get("first_token_t")
                if first_token_t is not None and llm_metrics["llm_ttft_ms"] is None:
                    llm_metrics["llm_ttft_ms"] = int((first_token_t - llm_t0) * 1000)
                    llm_metrics["first_token_t"] = first_token_t

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

                print(f"[LLM Chunk {llm_metrics['llm_chunk_count']}] {chunk}")
                llm_metrics["full_assistant_text"] += chunk + " "

                tts_queue.put({"type": "text", "text": chunk})

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
                tts_metrics["total_tts_time"] = 0.0
                continue

            text = msg.get("text", "")
            if not text:
                continue

            tts_t0 = time.perf_counter()
            pcm_bytes, sample_rate = tts_engine.synthesize(text)
            tts_t1 = time.perf_counter()
            tts_metrics["total_tts_time"] += (tts_t1 - tts_t0)

            if tts_metrics["tts_first_chunk_ms"] is None:
                tts_metrics["tts_first_chunk_ms"] = int(
                    (tts_t1 - tts_t0) * 1000
                )
                tts_metrics["tts_first_chunk_t"] = tts_t1

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

    if args.whisper_device is None:
        args.whisper_device = "cuda" if args.mode == "gpu" else "cpu"
    if args.whisper_compute_type is None:
        args.whisper_compute_type = "float16" if args.whisper_device == "cuda" else "int8"

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
    llm_engine = OllamaEngine(
        model=args.ollama_model,
        url=args.ollama_url,
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

    fieldnames = [
        "ts_iso", "mode", "stt_engine", "tts_engine", "item", "stage", "duration_ms",
        "cpu_percent", "ram_percent", "rss_mb",
        "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_name",
        "extra_json",
    ]

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

            # Shared metrics dicts (single-writer per dict = thread-safe)
            stt_metrics = {"stt_ms": None, "user_text": None}
            llm_metrics = {
                "llm_t0": None, "llm_ttfc_ms": None, "llm_ttft_ms": None,
                "full_assistant_text": "", "llm_chunk_count": 0,
                "ollama_stats": None,
            }
            tts_metrics = {
                "tts_first_chunk_ms": None, "tts_first_chunk_t": None,
                "total_tts_time": 0.0,
            }

            # STT hands over utterances through a single-slot mailbox (only the
            # newest one matters); TTS chunks are a FIFO, so they stay a queue.
            mailbox = UtteranceMailbox()
            tts_queue = queue.Queue()

            e2e_t0 = time.perf_counter()

            # Start worker threads
            stt_t = threading.Thread(
                target=stt_worker,
                args=(stt_engine, audio_source, mailbox, stt_metrics,
                      args.input_mode),
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
                                stt_metrics.get("last_speech_t", e2e_t0),
                                tts_metrics.get("last_play_t", e2e_t0)
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
            if args.input_mode == "mic":
                e2e_t0 = stt_metrics.get("last_speech_t", e2e_t0)
                
            user_text = stt_metrics.get("user_text") or ""
            stt_ms = stt_metrics.get("stt_ms") or 0
            # Read after the workers finish: a mic source only knows how much
            # audio it captured once capturing has stopped.
            input_duration_ms = audio_source.duration_ms

            stt_rtf = round(stt_ms / input_duration_ms, 3) if input_duration_ms > 0 else 0.0
            write_timing(writer, args, item_name, "stt", stt_ms,
                         {"input_duration_ms": input_duration_ms, "stt_rtf": stt_rtf})

            if not user_text:
                print("[Warn] No text recognized. Skipping to next file.")
                write_timing(writer, args, item_name, "e2e_response_ready",
                             int((time.perf_counter() - e2e_t0) * 1000),
                             {"input_duration_ms": input_duration_ms, "skipped": True})
                fcsv.flush()
                continue

            # LLM TTFT (first token) and TTFC (first chunk handed to TTS).
            # These differ by the time the chunker spends filling a chunk.
            llm_ttft_ms = llm_metrics.get("llm_ttft_ms")
            if llm_ttft_ms is not None:
                write_timing(writer, args, item_name, "llm_ttft", llm_ttft_ms)

            llm_ttfc_ms = llm_metrics.get("llm_ttfc_ms")
            if llm_ttfc_ms is not None:
                write_timing(writer, args, item_name, "llm_ttfc", llm_ttfc_ms)

            # TTS first chunk
            tts_first_chunk_ms = tts_metrics.get("tts_first_chunk_ms")
            if tts_first_chunk_ms is not None:
                write_timing(writer, args, item_name, "tts_first_chunk", tts_first_chunk_ms)

            # TTFA (wall-clock time from start of processing to first audio ready)
            tts_first_t = tts_metrics.get("tts_first_chunk_t")
            if tts_first_t is not None:
                ttfa_ms = int((tts_first_t - e2e_t0) * 1000)
                write_timing(writer, args, item_name, "ttfa", ttfa_ms)

                # TTFA anchored at the end of the speech, which is what a user
                # actually perceives. Only meaningful with realtime pacing; with
                # 'fast' pacing the audio ends as soon as it is read.
                speech_end_t = stt_metrics.get("speech_end_t")
                if speech_end_t is not None:
                    write_timing(writer, args, item_name, "ttfa_from_speech_end",
                                 int((tts_first_t - speech_end_t) * 1000),
                                 {"audio_pacing": args.audio_pacing})

            # LLM server-side evaluation metrics (ground truth from Ollama)
            ollama_stats = llm_metrics.get("ollama_stats")
            if ollama_stats:
                eval_dur_ns = ollama_stats.get("eval_duration_ns", 0)
                eval_count = ollama_stats.get("eval_count", 0)
                tokens_per_sec = round(eval_count / (eval_dur_ns / 1e9), 1) if eval_dur_ns > 0 else 0.0
                write_timing(writer, args, item_name, "llm_eval", int(eval_dur_ns / 1e6), {
                    "eval_tokens": eval_count,
                    "tokens_per_sec": tokens_per_sec,
                    "prompt_tokens": ollama_stats.get("prompt_eval_count", 0),
                    "prompt_eval_ms": int(ollama_stats.get("prompt_eval_duration_ns", 0) / 1e6),
                    "total_duration_ms": int(ollama_stats.get("total_duration_ns", 0) / 1e6),
                })

            # Log total TTS time
            write_timing(writer, args, item_name, "tts_total",
                         int(tts_metrics["total_tts_time"] * 1000))

            # Output audio duration and E2E summary. A mic session answers more
            # than once, so the reported duration covers every response WAV.
            response_wavs = [p for p in tts_metrics.get("out_wavs", []) if os.path.exists(p)]
            output_duration_ms = sum(wav_duration_ms(p) for p in response_wavs)
            full_reply = llm_metrics["full_assistant_text"].strip()

            write_timing(writer, args, item_name, "e2e_response_ready",
                         int((time.perf_counter() - e2e_t0) * 1000),
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

    print(f"\nDone. Latency/resource log appended to: {args.latency_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
