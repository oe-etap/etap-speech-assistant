#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Speech Assistant MWE (DE): Mic -> STT -> LLM -> TTS
Supports two modes:
  - cpu: Vosk (STT) + Ollama (LLM) + Piper (TTS)
  - gpu: Whisper (faster-whisper) + Ollama (LLM) + Coqui TTS (XTTS-v2)

Platform: Windows or Ubuntu (also WSL with audio support)
Audio flow: Turn-based: press ENTER to record for N seconds, then processing, then playback.
Latency is measured per stage and saved to CSV.
"""

import argparse
import time
import csv
import os
import subprocess
import shutil
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
import json as _json

import wave, json
from vosk import Model, KaldiRecognizer
from datetime import datetime

# Optional lazy singletons
_vosk = None
_whisper = None
_coqui_tts = None


# ---------- Helpers ----------
def ensure_wav_mono_16k(path):
    """Convert any audio to mono 16kHz WAV using ffmpeg."""
    base, _ = os.path.splitext(path)
    out_path = base + "_16k.wav"
    cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-f", "wav", out_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return out_path

def transcribe_with_vosk(wav_path, model_dir):
    """STT from WAV file using Vosk (expects mono 16 kHz 16-bit PCM)."""
    wf = wave.open(wav_path, "rb")
    assert wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2, \
        "Use mono/16 kHz/16-bit PCM WAV for Vosk"
    rec = KaldiRecognizer(Model(model_dir), wf.getframerate())
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
    return text

def record_from_mic(seconds=5, samplerate=16000, channels=1, dtype='int16'):
    print(f"[Mic] Recording {seconds}s at {samplerate} Hz ...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype=dtype, blocking=True)
    sd.wait()
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    return audio

def save_wav(path, audio_int16, samplerate=16000):
    sf.write(path, audio_int16, samplerate)
    print(f"[File] Saved WAV: {path}")

def play_wav(path):
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, sr)
    sd.wait()

def stt_vosk(audio_int16, vosk_model_dir, samplerate=16000):
    global _vosk
    if _vosk is None:
        _vosk = {"model": Model(vosk_model_dir)}
    rec = KaldiRecognizer(_vosk["model"], samplerate)
    rec.SetWords(True)
    ok = rec.AcceptWaveform(audio_int16.tobytes())
    if ok:
        result = _json.loads(rec.Result())
    else:
        result = _json.loads(rec.FinalResult())
    return (result.get("text") or "").strip()

def stt_whisper(audio_int16, whisper_model_name="medium", device="cuda", compute_type="float16", samplerate=16000):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = {"model": WhisperModel(whisper_model_name, device=device, compute_type=compute_type)}
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_int16, samplerate)
        tmp_path = tmp.name
    segments, _ = _whisper["model"].transcribe(tmp_path, language="de")
    text = " ".join([s.text for s in segments]).strip()
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return text

def llm_ollama_chat(user_text, model_name="phi3:mini", url="http://localhost:11434/api/generate"):
    system_prompt = (
        "Du bist ein knapper, sachlicher Assistent. "
        "Antworte kurz und präzise in einem einzigen Satz."
    )
    prompt = f"{system_prompt}\n\nNutzer sagte: \"{user_text}\"\n\nAntwort:"
    try:
        r = requests.post(url, json={"model": model_name, "prompt": prompt, "stream": False,"options": {
        "num_predict": 50,
        "temperature": 0.7
            }
        }, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"(Fehler beim LLM-Aufruf: {e})"

def tts_piper(text, piper_exe, piper_voice, output_file):
    cmd = [piper_exe, "-m", piper_voice, "-f", output_file]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)

def tts_coqui(text, voice_model="xtts_v2", language="de",
              speaker="Aaron Dreschner", out_wav="out.wav"):
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
    parser = argparse.ArgumentParser(description="Offline Speech Assistant MWE (DE)")
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True, help="cpu: Vosk+Piper; gpu: Whisper+Coqui")
    parser.add_argument("--turns", type=int, default=None, help="How many turns to run")
    parser.add_argument("--rec-seconds", type=float, default=5.0, help="Recording duration per turn (seconds)")
    parser.add_argument("--samplerate", type=int, default=16000, help="Microphone sample rate")
    parser.add_argument("--save-inputs", action="store_true", help="Save recorded user WAVs")

    parser.add_argument("--out-dir", type=str, default="outputs", help="Where to save the audiofiles")
    parser.add_argument("--latency-csv", type=str, default=None, help="CSV file to append latency logs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser.add_argument("--audio", type=str, default=None, help="Path to an audio file (wav/mp3/flac)")

    # CPU mode options
    parser.add_argument("--vosk-model", default="./vosk-model-small-de-0.15", help="Path to Vosk German model dir")
    parser.add_argument("--piper-exe", default="./piper", help="Path to Piper executable (piper or piper.exe)")
    parser.add_argument("--piper-voice", default="./de_DE-thorsten_high.onnx", help="Path to Piper .onnx German voice")

    # GPU mode options
    parser.add_argument("--whisper-model", default=None, help="faster-whisper model name")
    parser.add_argument("--whisper-device", default="cpu", help="Device for faster-whisper (cuda/cpu)")
    parser.add_argument("--whisper-compute-type", default="float16", help="Compute type (float16, int8_float16, etc.)")
    parser.add_argument("--coqui-voice", default="xtts_v2", help="Coqui voice model key")
    parser.add_argument("--coqui-language", default="de", help="Language code for Coqui (e.g., de)")
    parser.add_argument("--coqui-speaker", default="Aaron Dreschner", help="Speaker id for Coqui")
    # parser.add_argument(
    # "--coqui-voice",
    # default="thorsten",
    # help="Coqui GPU voice: 'thorsten' for native German or 'xtts_v2' for multispeaker")


    # Shared LLM
    parser.add_argument("--ollama-model", default="phi3:mini", help="Ollama model (e.g., phi3:mini)")

    # ---- parse ----
    args = parser.parse_args()

    # turns default: file módban 1, különben 3
    if args.turns is None:
        args.turns = 1 if args.audio else 3

    run_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    if not args.latency_csv:
        args.latency_csv = os.path.join(run_dir, f"latency_log_{timestamp}.csv")

    # Prepare CSV
    csv_exists = os.path.exists(args.latency_csv)
    with open(args.latency_csv, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if not csv_exists:
            writer.writerow(["ts_iso", "mode", "turn", "stage", "duration_ms"])

        for turn in range(1, args.turns + 1):
            print("=" * 60)

            # -------- Input: mic or file --------
            if args.audio:
                audio = None
                user_wav = os.path.join(run_dir, f"user_input_t{turn}.wav")
                try:
                    if os.path.abspath(args.audio) != os.path.abspath(user_wav):
                        shutil.copyfile(args.audio, user_wav)
                except Exception as e:
                    print(f"[WARN] Copy failed: {e}")
            else:
                input(f"Press ENTER to speak... (will record for {args.rec_seconds}s)")
                t0 = time.perf_counter()
                audio = record_from_mic(seconds=args.rec_seconds,
                                        samplerate=args.samplerate,
                                        channels=1, dtype="int16")
                t1 = time.perf_counter()
                writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), args.mode, turn, "record", int((t1 - t0) * 1000)])
                if args.save_inputs:
                    user_wav = os.path.join(run_dir, f"user_t{turn}.wav")
                    save_wav(user_wav, audio, args.samplerate)

            # -------- STT --------
            if args.audio:
                print(f"[INFO] Processing audio file: {args.audio}")
                wav_path = ensure_wav_mono_16k(args.audio)

                t0 = time.perf_counter()

                if args.mode == "cpu":

                    user_text = transcribe_with_vosk(wav_path, args.vosk_model)
                else:
                    audio_data, sr = sf.read(wav_path, dtype="int16")
                    user_text = stt_whisper(
                        audio_int16=audio_data,
                        whisper_model_name=args.whisper_model,
                        device=args.whisper_device,
                        compute_type=args.whisper_compute_type,
                        samplerate=16000
                    )
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()

                if args.whisper_model:
                    user_text = stt_whisper(
                        audio_int16=audio,
                        whisper_model_name=args.whisper_model,
                        device=args.whisper_device,
                        compute_type=args.whisper_compute_type,
                        samplerate=args.samplerate
                    )
                else:
                    user_text = stt_vosk(
                        audio_int16=audio,
                        vosk_model_dir=args.vosk_model,
                        samplerate=args.samplerate
                    )

            t1 = time.perf_counter()

            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), args.mode, turn, "stt", int((t1 - t0) * 1000)])
            print(f"[STT] {user_text!r}")

            if not user_text:
                print("[Warn] No text recognized. Skipping to next turn.")
                continue

            # -------- LLM --------
            t0 = time.perf_counter()
            reply = llm_ollama_chat(user_text=user_text, model_name=args.ollama_model)
            t1 = time.perf_counter()
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), args.mode, turn, "llm", int((t1 - t0) * 1000)])
            print(f"[LLM] {reply}")

            # -------- TTS --------
            out_wav = os.path.join(run_dir, f"assistant_t{turn}.wav")
            # if args.mode == "cpu":
            t0 = time.perf_counter()
            tts_piper(text=reply, piper_exe=args.piper_exe, piper_voice=args.piper_voice, output_file=out_wav)
            t1 = time.perf_counter()
            # else:
            #     t0 = time.perf_counter()
            #     tts_coqui(
            #     text=reply,
            #     voice_model=args.coqui_voice,
            #     language=args.coqui_language,
            #     speaker=args.coqui_speaker,
            #     out_wav=out_wav)

            # t1 = time.perf_counter()
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), args.mode, turn, "tts", int((t1 - t0) * 1000)])
            print(f"[TTS] Saved to {out_wav}. Playing...")

            # -------- Playback --------
            play_wav(out_wav)

            if args.audio:
                break

    print(f"\nDone. Latency log appended to: {args.latency_csv}")
    print("Tip: open the CSV in a spreadsheet to analyze per-stage timings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
