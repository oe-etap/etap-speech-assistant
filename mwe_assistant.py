#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Speech Assistant MWE (EN): audio file -> STT -> LLM -> TTS.
Supports selectable engines via streaming pipeline architecture:
  - STT: Vosk or faster-whisper
  - TTS: Piper or Coqui TTS
  - LLM: Ollama
  - mode: CPU/GPU preset, with Whisper available on CPU too
"""

import argparse
import csv
import yaml
import json
import os
import shutil
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path
import queue
import threading

import numpy as np
import soundfile as sf

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Import engines
from stt_engines import VoskEngine, WhisperEngine
from tts_engines import PiperEngine, CoquiEngine
from llm_engine import OllamaEngine

# nvidia-smi cache (timestamp, stats_dict)
_gpu_cache = (0.0, None)
_GPU_CACHE_TTL = 2.0


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
    if hasattr(writer, "stream"):
        writer.stream.flush()

def chunked_audio_reader(wav_path, chunk_size=4000):
    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(chunk_size)
            if len(data) == 0:
                break
            yield data

# ---------- Pipeline Workers ----------
def stt_worker(wav_path, stt_engine, llm_queue, stt_metrics, debounce_sec=0.3):
    t0 = time.perf_counter()
    audio_iter = chunked_audio_reader(wav_path)
    last_partial_text = ""
    last_partial_time = 0.0

    for result in stt_engine.transcribe_stream(audio_iter):
        t_now = time.perf_counter()
        text = result.get("text", "").strip()
        if not text:
            continue

        if not result.get("is_final", True):
            # Partial STT output
            word_count = len(text.split())
            if word_count >= 3 and text != last_partial_text:
                if (t_now - last_partial_time) >= debounce_sec:
                    last_partial_text = text
                    last_partial_time = t_now
                    print(f"[STT Partial] {text}")
                    llm_queue.put({"type": "partial", "text": text})
        else:
            # Final STT output (speech pause / VAD boundary)
            if stt_metrics["stt_ms"] is None:
                stt_metrics["stt_ms"] = int((t_now - t0) * 1000)
            if stt_metrics["stt_first_final_t"] is None:
                stt_metrics["stt_first_final_t"] = t_now
            stt_metrics["stt_end_t"] = t_now
            stt_metrics["full_user_text"] += text + " "
            print(f"[STT Final] {text}")
            last_partial_text = ""
            llm_queue.put({"type": "final", "text": text})

    llm_queue.put(None) # Signal EOF to LLM

def llm_worker(llm_engine, llm_queue, tts_queue, llm_metrics):
    generating = False
    next_msg = None  # Pre-fetched message from mid-stream interruption

    while True:
        # Use pre-fetched message if available, otherwise block for next
        if next_msg is not None:
            msg = next_msg
            next_msg = None
        else:
            msg = llm_queue.get()

        if msg is None:
            tts_queue.put(None)  # Signal EOF to TTS
            break

        # Drain older messages: keep only the newest non-EOF message
        while True:
            try:
                drain_msg = llm_queue.get_nowait()
            except queue.Empty:
                break
            if drain_msg is None:
                # Put the EOF sentinel back so it is processed on the next iteration
                llm_queue.put(None)
                break
            msg = drain_msg

        if isinstance(msg, str):
            msg = {"type": "final", "text": msg}

        msg_type = msg.get("type", "final")

        if msg_type == "cancel":
            llm_engine.cancel()
            tts_queue.put({"type": "cancel"})
            generating = False
            continue

        if msg_type in ("partial", "final"):
            # Cancel any prior active LLM stream before starting a new one
            if generating:
                llm_engine.cancel()
                tts_queue.put({"type": "cancel"})
            llm_metrics["full_assistant_text"] = ""
            llm_metrics["llm_chunk_count"] = 0
            llm_metrics["llm_ttfc_ms"] = None

        text_chunk = msg.get("text", "")
        if not text_chunk:
            continue

        generating = True
        llm_t0 = time.perf_counter()
        for chunk_data in llm_engine.generate_stream(text_chunk):
            if chunk_data.get("cancelled"):
                print("[LLM] Generation cancelled mid-stream")
                tts_queue.put({"type": "cancel"})
                break

            # Check if a newer STT message arrived while generating
            try:
                pending = llm_queue.get_nowait()
                if pending is None:
                    # EOF sentinel — re-queue so the outer loop sees it after
                    # the current generation finishes naturally
                    llm_queue.put(None)
                else:
                    # A non-EOF message arrived — cancel and hand it directly
                    # to the outer loop via next_msg (avoids re-queue ordering issues)
                    print("[LLM] Newer STT message arrived, cancelling active stream")
                    llm_engine.cancel()
                    tts_queue.put({"type": "cancel"})
                    next_msg = pending
                    break
            except queue.Empty:
                pass

            text = chunk_data.get("text", "")
            stats = chunk_data.get("ollama_stats")

            if stats:
                llm_metrics["ollama_stats"] = stats

            if text:
                if llm_metrics["llm_ttfc_ms"] is None:
                    llm_metrics["llm_ttfc_ms"] = int((time.perf_counter() - llm_t0) * 1000)
                llm_metrics["full_assistant_text"] += text + " "
                llm_metrics["llm_chunk_count"] += 1
                print(f"[LLM Chunk] {text}")
                tts_queue.put({"type": "text", "text": text})

        generating = False

def tts_worker(tts_engine, tts_queue, out_wav, e2e_t0, tts_metrics):
    while True:
        is_eof = False
        is_cancelled = False

        def text_generator():
            nonlocal is_eof, is_cancelled
            while True:
                item = tts_queue.get()
                if item is None:
                    is_eof = True
                    break
                if isinstance(item, str):
                    yield item
                elif isinstance(item, dict):
                    msg_type = item.get("type", "text")
                    if msg_type == "cancel":
                        print("[TTS] Cancel signal received, resetting TTS generator")
                        is_cancelled = True
                        break
                    elif msg_type == "text":
                        yield item.get("text", "")

        final_wav = None
        for pcm_bytes, sample_rate in tts_engine.synthesize_stream(text_generator()):
            tts_t1 = time.perf_counter()

            if tts_metrics["tts_t1"] is None:
                tts_metrics["tts_t1"] = tts_t1

            if final_wav is None:
                final_wav = wave.open(out_wav, "wb")
                final_wav.setnchannels(1)
                final_wav.setsampwidth(2)
                final_wav.setframerate(sample_rate)
            final_wav.writeframes(pcm_bytes)

        # Close any open WAV handle before deciding what to do next
        if final_wav is not None:
            final_wav.close()
            final_wav = None

        if is_cancelled:
            if os.path.exists(out_wav):
                try:
                    os.remove(out_wav)
                except Exception:
                    pass
            tts_metrics["tts_t1"] = None
            continue

        if is_eof:
            break

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # STT options
    parser.add_argument("--vosk-model", default="./vosk/vosk-model-small-en-us-0.15", help="Path to English Vosk model dir")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model name")
    parser.add_argument("--whisper-device", choices=["cpu", "cuda"], default=None, help="Device for faster-whisper")
    parser.add_argument("--whisper-compute-type", default=None, help="Compute type (int8, float16, etc.)")
    parser.add_argument("--stt-debounce-sec", type=float, default=0.5, help="Debounce interval in seconds for partial STT streaming")

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

    if not args.audio:
        parser.error("the following arguments are required: --audio (either in CLI or config)")

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

    print("[INFO] Pre-loading AI models (STT, LLM, TTS)...")
    
    # Init LLM
    llm_engine = OllamaEngine(model_name=args.ollama_model, url=args.ollama_url)

    # Init STT
    if args.stt_engine == "vosk":
        stt_engine = VoskEngine(args.vosk_model)
    elif args.stt_engine == "whisper":
        stt_engine = WhisperEngine(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)

    # Init TTS
    if args.tts_engine == "piper":
        tts_engine = PiperEngine(args.piper_exe, args.piper_voice, args.piper_use_exe)
    elif args.tts_engine == "coqui":
        tts_engine = CoquiEngine(args.coqui_voice, args.coqui_language, args.coqui_speaker)

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

    csv_exists = os.path.exists(args.latency_csv)
    with open(args.latency_csv, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for item_index, audio_path in enumerate(args.audio, start=1):
            print("=" * 60)
            audio_file = Path(audio_path)
            if not audio_file.exists():
                raise FileNotFoundError(audio_path)

            item_name = audio_file.stem
            print(f"[INFO] Processing audio file: {audio_file}")

            user_wav = os.path.join(run_dir, f"user_input_{item_index}_{audio_file.name}")
            try:
                if os.path.abspath(audio_path) != os.path.abspath(user_wav):
                    shutil.copyfile(audio_path, user_wav)
            except Exception as e:
                print(f"[WARN] Copy failed: {e}")

            wav_path = ensure_wav_mono_16k(audio_path, out_dir=run_dir, prefix=f"input_{item_index}")
            input_duration_ms = wav_duration_ms(wav_path)
            
            stt_engine.reset()
            e2e_t0 = time.perf_counter()
            out_wav = os.path.join(run_dir, f"assistant_{item_index}_{item_name}.wav")
            
            # --- Threading Setup ---
            llm_queue = queue.Queue()
            tts_queue = queue.Queue()
            
            stt_metrics = {"stt_ms": None, "stt_first_final_t": None, "stt_end_t": None, "full_user_text": ""}
            llm_metrics = {"llm_ttfc_ms": None, "ollama_stats": None, "full_assistant_text": "", "llm_chunk_count": 0}
            tts_metrics = {"tts_t1": None, "total_tts_time": 0.0}

            # Start threads
            t_stt = threading.Thread(target=stt_worker, args=(wav_path, stt_engine, llm_queue, stt_metrics, args.stt_debounce_sec))
            t_llm = threading.Thread(target=llm_worker, args=(llm_engine, llm_queue, tts_queue, llm_metrics))
            t_tts = threading.Thread(target=tts_worker, args=(tts_engine, tts_queue, out_wav, e2e_t0, tts_metrics))
            
            t_stt.start()
            t_llm.start()
            t_tts.start()
            
            t_stt.join()
            t_llm.join()
            t_tts.join()

            # --- Log metrics ---
            e2e_ms = int((time.perf_counter() - e2e_t0) * 1000)
            
            stt_ms = stt_metrics.get("stt_ms") or 0
            stt_rtf = round(stt_ms / input_duration_ms, 3) if input_duration_ms > 0 else 0.0
            write_timing(writer, args, item_name, "stt", stt_ms, {"input_duration_ms": input_duration_ms, "stt_rtf": stt_rtf})
            
            if llm_metrics["llm_ttfc_ms"]:
                write_timing(writer, args, item_name, "llm_ttfc", llm_metrics["llm_ttfc_ms"])
            
            if tts_metrics.get("tts_t1") and stt_metrics.get("stt_first_final_t"):
                ttfa_ms = int((tts_metrics["tts_t1"] - stt_metrics["stt_first_final_t"]) * 1000)
                write_timing(writer, args, item_name, "ttfa", ttfa_ms)
            
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
            
            output_duration_ms = wav_duration_ms(out_wav) if os.path.exists(out_wav) else 0
            
            write_timing(writer, args, item_name, "e2e_response_ready", e2e_ms,
                         {"input_duration_ms": input_duration_ms,
                          "output_duration_ms": output_duration_ms,
                          "output_wav": out_wav,
                          "full_text": llm_metrics["full_assistant_text"].strip(),
                          "response_word_count": len(llm_metrics["full_assistant_text"].split()),
                          "response_char_count": len(llm_metrics["full_assistant_text"].strip()),
                          "llm_chunk_count": llm_metrics["llm_chunk_count"]})
            print(f"[TTS] Stream finished. Saved to {out_wav}.")

            # --- Transcript Logging ---
            transcript_record = {
                "stt_text": stt_metrics["full_user_text"].strip(),
                "llm_text": llm_metrics["full_assistant_text"].strip()
            }
            jsonl_path = os.path.join(run_dir, "transcripts.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(transcript_record, ensure_ascii=False) + "\n")
            yaml_path = os.path.join(run_dir, "transcripts.yaml")
            with open(yaml_path, "a", encoding="utf-8") as f_yaml:
                yaml.dump([transcript_record], f_yaml, sort_keys=False, allow_unicode=True)

            if not args.keep_normalized:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    print(f"\nDone. Latency/resource log appended to: {args.latency_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
