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
import yaml
import json
import os
import shutil
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

# Optional module-level imports
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    pass

try:
    from faster_whisper import WhisperModel
except ImportError:
    pass

try:
    from piper.voice import PiperVoice
except ImportError:
    pass

try:
    from TTS.api import TTS
except ImportError:
    pass

_requests_session = requests.Session()

_vosk_model = None
_vosk_recognizer = None
_whisper_model = None
_coqui_tts = None
_piper_voice = None  # PiperVoice Python API singleton

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


def transcribe_with_vosk(wav_path):
    """STT from WAV file using Vosk (expects mono 16 kHz 16-bit PCM)."""
    wf = wave.open(wav_path, "rb")
    assert wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2, \
        "Use mono/16 kHz/16-bit PCM WAV for Vosk"
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if _vosk_recognizer.AcceptWaveform(data):
            results.append(json.loads(_vosk_recognizer.Result()))
    results.append(json.loads(_vosk_recognizer.FinalResult()))
    text = " ".join([r.get("text", "") for r in results]).strip()
    wf.close()
    return text


def stt_whisper(wav_path):
    """STT from WAV file using faster-whisper.
    Requires _whisper to be pre-loaded via main()."""
    segments, _ = _whisper_model.transcribe(wav_path, language="en")
    text = " ".join([s.text.strip() for s in segments]).strip()
    return text


def stream_llm_ollama_chat(user_text, model_name="phi3:mini", url="http://localhost:11434/api/generate"):
    """Stream LLM response from Ollama, yielding sentence-level chunks as dicts.
    
    Yields dicts with keys:
      - 'text': the synthesizable text chunk (str)
      - 'ollama_stats': None for text chunks; dict with server-side timing for the final metadata chunk
    
    The last yielded dict may have empty 'text' and non-None 'ollama_stats' containing
    Ollama's server-side performance metrics (eval_count, eval_duration, etc.).
    """
    system_prompt = (
        "You are a concise, factual but friendly voice assistant. "
        "Answer in English in 1-3 medium length sentences."
    )
    prompt = f'{system_prompt}\n\nThe user said: "{user_text}"\n\nAnswer:'
    try:
        r = _requests_session.post(url, json={"model": model_name, "prompt": prompt, "stream": True, "options": {
        "num_predict": 150,
        "temperature": 0.7
            }
        }, timeout=120)
        r.raise_for_status()
        
        buffer = ""
        terminators = {".", "?", "!", ":", ";", "\n"}
        ollama_stats = None
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                piece = data.get("response", "")
                buffer += piece
                
                # Capture Ollama server-side stats from the final message
                if data.get("done", False):
                    ollama_stats = {
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration_ns": data.get("eval_duration", 0),
                        "total_duration_ns": data.get("total_duration", 0),
                    }
                
                if any(t in piece for t in terminators):
                    if len(buffer.strip()) > 2:
                        yield {"text": buffer.strip(), "ollama_stats": None}
                        buffer = ""
        if buffer.strip():
            yield {"text": buffer.strip(), "ollama_stats": None}
        # Yield final metadata-only entry with Ollama server-side stats
        if ollama_stats:
            yield {"text": "", "ollama_stats": ollama_stats}
    except Exception as e:
        yield {"text": f"(LLM call failed: {e})", "ollama_stats": None}


def tts_piper_python(text):
    """Returns raw 16-bit PCM bytes and sample rate from Piper using the Python API."""
    pcm_parts = []
    sample_rate = None
    for audio_chunk in _piper_voice.synthesize(text):
        pcm_parts.append(audio_chunk.audio_int16_bytes)
        if sample_rate is None:
            sample_rate = audio_chunk.sample_rate
    return b"".join(pcm_parts), sample_rate or 22050


def tts_piper_exe(text, piper_exe, piper_voice, sample_rate=16000):
    """Returns raw 16-bit PCM bytes and sample rate from Piper CLI executable.
    Fallback for when the Python API is not available."""
    cmd = [piper_exe, "-m", piper_voice, "--output_raw"]
    result = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, check=True)
    return result.stdout, sample_rate


def tts_coqui_stream(text, language="en", speaker="Daisy Studious"):
    """Returns raw 16-bit PCM bytes and sample rate from Coqui."""
    wav = _coqui_tts.tts(text=text, speaker=speaker, language=language)
    
    audio_data = np.array(wav, dtype=np.float32)
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_data = (audio_data * 32767.0).astype(np.int16)
    
    sample_rate = _coqui_tts.synthesizer.output_sample_rate
    return audio_data.tobytes(), sample_rate


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
    parser.add_argument("--vosk-model", default="./vosk-model-small-en-us-0.15", help="Path to English Vosk model dir")
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

    if args.whisper_device is None:
        args.whisper_device = "cuda" if args.mode == "gpu" else "cpu"
    if args.whisper_compute_type is None:
        args.whisper_compute_type = "float16" if args.whisper_device == "cuda" else "int8"

    # Pre-load Piper sample rate to avoid disk I/O on every chunk (only needed for --piper-use-exe fallback)
    piper_sample_rate = 16000
    if args.tts_engine == "piper":
        json_path = args.piper_voice + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    vdata = json.load(f)
                    piper_sample_rate = vdata.get("audio", {}).get("sample_rate", 16000)
            except Exception:
                pass

    # Pre-load (warmup) models so their initialization time isn't counted in the latency of the first file
    print("[INFO] Pre-loading AI models (STT, LLM, TTS)...")
    
    # Ollama Warmup (betöltés VRAM/RAM-ba)
    try:
        _requests_session.post(args.ollama_url, json={
            "model": args.ollama_model,
            "prompt": "",
            "options": {"num_predict": 1}
        }, timeout=120)
    except Exception as e:
        print(f"[WARN] Failed to warmup Ollama: {e}")

    global _vosk_model, _vosk_recognizer, _whisper_model, _coqui_tts, _piper_voice
    if args.stt_engine == "vosk":
        _vosk_model = Model(args.vosk_model)
        _vosk_recognizer = KaldiRecognizer(_vosk_model, 16000)
        _vosk_recognizer.SetWords(True)
    elif args.stt_engine == "whisper":
        _whisper_model = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)

    if args.tts_engine == "piper" and not args.piper_use_exe:
        try:
            _piper_voice = PiperVoice.load(args.piper_voice)
            print(f"[INFO] Piper Python API loaded (model: {args.piper_voice})")
        except Exception as e:
            print(f"[WARN] Piper Python API failed ({e}), falling back to CLI executable")
            args.piper_use_exe = True

    if args.tts_engine == "coqui":
        _coqui_tts = TTS(model_name=f"tts_models/multilingual/multi-dataset/{args.coqui_voice}")

    run_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

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

    if not args.latency_csv:
        args.latency_csv = os.path.join(run_dir, f"latency_log_{timestamp}.csv")

    fieldnames = [
        "ts_iso", "mode", "stt_engine", "tts_engine", "item", "stage", "duration_ms",
        "cpu_percent", "ram_percent", "rss_mb",
        "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_name",
        "extra_json",
    ]

    # Prepare CSV
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
            
            if args.stt_engine == "vosk":
                _vosk_recognizer.Reset()

            e2e_t0 = time.perf_counter()

            # -------- STT --------
            t0 = time.perf_counter()
            if args.stt_engine == "vosk":
                user_text = transcribe_with_vosk(wav_path)
            else:
                user_text = stt_whisper(wav_path)
            t1 = time.perf_counter()
            stt_ms = int((t1 - t0) * 1000)
            stt_rtf = round(stt_ms / input_duration_ms, 3) if input_duration_ms > 0 else 0.0
            write_timing(writer, args, item_name, "stt", stt_ms,
                         {"input_duration_ms": input_duration_ms, "stt_rtf": stt_rtf})
            print(f"[STT] {user_text!r}")

            if not user_text:
                print("[Warn] No text recognized. Skipping to next file.")
                write_timing(writer, args, item_name, "e2e_response_ready", int((time.perf_counter() - e2e_t0) * 1000),
                             {"input_duration_ms": input_duration_ms, "skipped": True})
                continue

            # -------- LLM & TTS Streaming --------
            print(f"[LLM] Starting stream...")
            out_wav = os.path.join(run_dir, f"assistant_{item_index}_{item_name}.wav")
            first_chunk_ts = None
            final_wav = None
            llm_t0 = time.perf_counter()
            
            total_tts_time = 0.0
            tts_first_chunk_ms = None
            llm_ttfc_ms = None
            full_reply = ""
            ollama_stats = None
            chunk_count = 0
            
            for chunk_data in stream_llm_ollama_chat(user_text, model_name=args.ollama_model, url=args.ollama_url):
                # Handle dict-based chunk from generator
                if isinstance(chunk_data, dict):
                    chunk = chunk_data.get("text", "")
                    if chunk_data.get("ollama_stats"):
                        ollama_stats = chunk_data["ollama_stats"]
                else:
                    chunk = chunk_data  # fallback for plain string
                
                if not chunk:
                    continue
                chunk_count += 1
                
                # LLM Time to First (synthesizable) Chunk
                if first_chunk_ts is None:
                    first_chunk_ts = time.perf_counter()
                    llm_ttfc_ms = int((first_chunk_ts - llm_t0) * 1000)
                    write_timing(writer, args, item_name, "llm_ttfc", llm_ttfc_ms)
                    
                print(f"[LLM Chunk {chunk_count}] {chunk}")
                full_reply += chunk + " "
                
                tts_t0 = time.perf_counter()
                if args.tts_engine == "piper":
                    if _piper_voice is not None:
                        pcm_bytes, sample_rate = tts_piper_python(chunk)
                    else:
                        pcm_bytes, sample_rate = tts_piper_exe(
                            chunk, args.piper_exe, args.piper_voice, sample_rate=piper_sample_rate
                        )
                else:
                    pcm_bytes, sample_rate = tts_coqui_stream(
                        chunk, args.coqui_language, args.coqui_speaker
                    )
                tts_t1 = time.perf_counter()
                chunk_tts_ms = int((tts_t1 - tts_t0) * 1000)
                total_tts_time += (tts_t1 - tts_t0)
                
                # Track TTS first chunk time separately
                if tts_first_chunk_ms is None:
                    tts_first_chunk_ms = chunk_tts_ms
                    write_timing(writer, args, item_name, "tts_first_chunk", tts_first_chunk_ms)
                
                # Közvetlenül a nyitva tartott fájl pufferébe írunk (valódi streaming)
                if final_wav is None:
                    final_wav = wave.open(out_wav, "wb")
                    final_wav.setnchannels(1)
                    final_wav.setsampwidth(2) # 16-bit
                    final_wav.setframerate(sample_rate)
                
                final_wav.writeframes(pcm_bytes)

            if final_wav:
                final_wav.close()
            
            # TTFA (Time to First Audio) = STT + LLM TTFC + TTS first chunk
            if llm_ttfc_ms is not None and tts_first_chunk_ms is not None:
                ttfa_ms = stt_ms + llm_ttfc_ms + tts_first_chunk_ms
                write_timing(writer, args, item_name, "ttfa", ttfa_ms)
            
            # LLM server-side evaluation metrics (ground truth from Ollama)
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
            write_timing(writer, args, item_name, "tts_total", int(total_tts_time * 1000))
            
            # Output audio duration
            output_duration_ms = wav_duration_ms(out_wav) if os.path.exists(out_wav) else 0
            
            write_timing(writer, args, item_name, "e2e_response_ready", int((time.perf_counter() - e2e_t0) * 1000),
                         {"input_duration_ms": input_duration_ms,
                          "output_duration_ms": output_duration_ms,
                          "output_wav": out_wav,
                          "full_text": full_reply.strip(),
                          "response_word_count": len(full_reply.split()),
                          "response_char_count": len(full_reply.strip()),
                          "llm_chunk_count": chunk_count})
            print(f"[TTS] Stream finished. Saved to {out_wav}.")

            if not args.keep_normalized:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    print(f"\nDone. Latency/resource log appended to: {args.latency_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
