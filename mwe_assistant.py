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
import time
import csv
import os
import subprocess
import shutil
from pathlib import Path

import soundfile as sf
import requests
import json as _json

import wave, json
from datetime import datetime

# Optional lazy singletons
_vosk = None
_whisper = None
_coqui_tts = None


# ---------- Helpers ----------
def ensure_wav_mono_16k(path, out_dir=None, prefix=None):
    """Convert any audio to mono 16kHz WAV using ffmpeg."""
    source = Path(path)
    if out_dir:
        stem = f"{prefix}_{source.stem}" if prefix else source.stem
        out_path = Path(out_dir) / f"{stem}_16k.wav"
    else:
        base, _ = os.path.splitext(path)
        out_path = Path(base + "_16k.wav")
    cmd = ["ffmpeg", "-y", "-i", str(source), "-ac", "1", "-ar", "16000", "-f", "wav", str(out_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return str(out_path)


def wav_duration_ms(wav_path):
    with sf.SoundFile(wav_path) as wav:
        return int((len(wav) / wav.samplerate) * 1000)


def collect_resource_snapshot():
    """Best-effort resource snapshot. Missing psutil/nvidia-smi leaves blanks."""
    stats = {
        "cpu_percent": "",
        "ram_percent": "",
        "rss_mb": "",
        "gpu_util_percent": "",
        "gpu_mem_used_mb": "",
        "gpu_mem_total_mb": "",
        "gpu_name": "",
    }

    try:
        import psutil

        proc = psutil.Process(os.getpid())
        stats["cpu_percent"] = psutil.cpu_percent(interval=None)
        stats["ram_percent"] = psutil.virtual_memory().percent
        stats["rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        pass

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
            stats["gpu_name"] = name
            stats["gpu_util_percent"] = util
            stats["gpu_mem_used_mb"] = mem_used
            stats["gpu_mem_total_mb"] = mem_total
    except Exception:
        pass

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


def transcribe_with_vosk(wav_path, model_dir):
    """STT from WAV file using Vosk (expects mono 16 kHz 16-bit PCM)."""
    global _vosk
    from vosk import Model, KaldiRecognizer

    if _vosk is None:
        _vosk = {"model": Model(model_dir)}

    wf = wave.open(wav_path, "rb")
    assert wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2, \
        "Use mono/16 kHz/16-bit PCM WAV for Vosk"
    rec = KaldiRecognizer(_vosk["model"], wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    text = " ".join([r.get("text", "") for r in results]).strip()
    wf.close()
    return text


def stt_whisper(wav_path, whisper_model_name="small", device="cpu", compute_type="int8"):
    global _whisper
    cache_key = (whisper_model_name, device, compute_type)
    if _whisper is None or _whisper.get("key") != cache_key:
        from faster_whisper import WhisperModel
        _whisper = {
            "key": cache_key,
            "model": WhisperModel(whisper_model_name, device=device, compute_type=compute_type),
        }
    segments, _ = _whisper["model"].transcribe(wav_path, language="en")
    text = " ".join([s.text.strip() for s in segments]).strip()
    return text


def llm_ollama_chat(user_text, model_name="phi3:mini", url="http://localhost:11434/api/generate"):
    system_prompt = (
        "You are a concise, factual but friendly voice assistant. "
        "Answer in English in 1-3 medium length sentences."
    )
    prompt = f'{system_prompt}\n\nThe user said: "{user_text}"\n\nAnswer:'
    try:
        r = requests.post(url, json={"model": model_name, "prompt": prompt, "stream": False, "options": {
        "num_predict": 150,
        "temperature": 0.7
            }
        }, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"(LLM call failed: {e})"


def tts_piper(text, piper_exe, piper_voice, output_file):
    cmd = [piper_exe, "-m", piper_voice, "-f", output_file]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def tts_coqui(text, voice_model="xtts_v2", language="en",
              speaker="Daisy Studious", out_wav="out.wav"):
    global _coqui_tts
    if _coqui_tts is None:
        from TTS.api import TTS
        _coqui_tts = TTS(model_name=f"tts_models/multilingual/multi-dataset/{voice_model}")
    _coqui_tts.tts_to_file(
        text=text,
        file_path=out_wav,
        speaker=speaker,
        language=language
    )


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Offline Speech Assistant MWE (EN, file input only)")
    parser.add_argument("--audio", nargs="+", required=True, help="One or more audio files (wav/mp3/flac/etc.)")
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
    parser.add_argument("--coqui-voice", default="xtts_v2", help="Coqui voice model key")
    parser.add_argument("--coqui-language", default="en", help="Language code for Coqui")
    parser.add_argument("--coqui-speaker", default="Daisy Studious", help="Speaker id for Coqui")

    # Shared LLM
    parser.add_argument("--ollama-model", default="phi3:mini", help="Ollama model (e.g. phi3:mini)")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama generate endpoint")

    # ---- parse ----
    args = parser.parse_args()

    if args.whisper_device is None:
        args.whisper_device = "cuda" if args.mode == "gpu" else "cpu"
    if args.whisper_compute_type is None:
        args.whisper_compute_type = "float16" if args.whisper_device == "cuda" else "int8"

    run_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

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
            e2e_t0 = time.perf_counter()

            # -------- STT --------
            t0 = time.perf_counter()
            if args.stt_engine == "vosk":
                user_text = transcribe_with_vosk(wav_path, args.vosk_model)
            else:
                user_text = stt_whisper(
                    wav_path=wav_path,
                    whisper_model_name=args.whisper_model,
                    device=args.whisper_device,
                    compute_type=args.whisper_compute_type
                )
            t1 = time.perf_counter()
            write_timing(writer, args, item_name, "stt", int((t1 - t0) * 1000),
                         {"input_duration_ms": input_duration_ms})
            print(f"[STT] {user_text!r}")

            if not user_text:
                print("[Warn] No text recognized. Skipping to next file.")
                write_timing(writer, args, item_name, "e2e_response_ready", int((time.perf_counter() - e2e_t0) * 1000),
                             {"input_duration_ms": input_duration_ms, "skipped": True})
                continue

            # -------- LLM --------
            t0 = time.perf_counter()
            reply = llm_ollama_chat(user_text=user_text, model_name=args.ollama_model, url=args.ollama_url)
            t1 = time.perf_counter()
            write_timing(writer, args, item_name, "llm", int((t1 - t0) * 1000))
            print(f"[LLM] {reply}")

            # -------- TTS --------
            out_wav = os.path.join(run_dir, f"assistant_{item_index}_{item_name}.wav")
            t0 = time.perf_counter()
            if args.tts_engine == "piper":
                tts_piper(text=reply, piper_exe=args.piper_exe, piper_voice=args.piper_voice, output_file=out_wav)
            else:
                tts_coqui(
                    text=reply,
                    voice_model=args.coqui_voice,
                    language=args.coqui_language,
                    speaker=args.coqui_speaker,
                    out_wav=out_wav)
            t1 = time.perf_counter()
            write_timing(writer, args, item_name, "tts", int((t1 - t0) * 1000))
            write_timing(writer, args, item_name, "e2e_response_ready", int((time.perf_counter() - e2e_t0) * 1000),
                         {"input_duration_ms": input_duration_ms, "output_wav": out_wav})
            print(f"[TTS] Saved to {out_wav}.")

            if not args.keep_normalized:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    print(f"\nDone. Latency/resource log appended to: {args.latency_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
