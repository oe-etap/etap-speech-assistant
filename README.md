# Offline Speech Assistant MWE (DE) – CPU/GPU

Mic → STT → LLM → TTS, fully offline (except for Ollama model pulls). The script is turn-based
and logs per-stage latency.

---

## Features
- **CPU mode**: Vosk (STT) + Ollama (LLM) + Piper (TTS)
- **GPU mode**: faster-whisper (STT) + Ollama (LLM) + Coqui TTS XTTS‑v2 (TTS)
- **Audio I/O**: microphone recording (sounddevice) + WAV playback (soundfile)
- **File input**: you can process an existing audio file via `--audio <file>`
- **Robust audio handling**: any input file is converted to mono 16 kHz WAV using **ffmpeg**
- **Outputs**: every run creates `outputs/<timestamp>/` with all generated WAVs
- **Latency CSV**: per-stage timings are appended to a CSV (path can be customized)

---

## Requirements

### Common (both modes)
- Python **3.10+**
- **Ollama** running locally (`ollama serve`), with a pulled model (e.g. `phi3:mini`)
- **ffmpeg** accessible in your PATH (the script calls the `ffmpeg` CLI)
- Packages: see the respective `requirements_*.txt`

### Additional assets to download
- **Vosk German model** (for CPU mode), e.g. `vosk-model-small-de-0.15`
- **Piper** binary and a **German voice** (e.g. `de_DE-thorsten_high.onnx`)
- **CUDA runtime + PyTorch CUDA build** (for GPU mode)

> Tip (Windows): put `ffmpeg.exe`, `piper.exe`, and your `.onnx` voice into a known folder and add that folder to PATH,
> or pass explicit paths to `--piper-exe`/`--piper-voice`.

---

## Install

### 1) Common
```bash
# start the local LLM
ollama serve
# e.g., pull a model
ollama pull phi3:mini
```

### 2) CPU‑only setup
```bash
pip install -r requirements_cpu.txt
```
Download:
- Vosk German model (e.g. `vosk-model-small-de-0.15`) and unpack it somewhere.
- Piper binary (`piper` / `piper.exe`) + a German voice ONNX (e.g. `de_DE-thorsten_high.onnx`).
- ffmpeg (ensure `ffmpeg` is on PATH).

**Run (Windows paths example):**
```bash
python mwe_assistant.py --mode cpu --turns 3 --rec-seconds 5 ^
  --vosk-model .\vosk-model-small-de-0.15 ^
  --piper-exe .\piper\piper.exe ^
  --piper-voice .\piper\de_DE-thorsten_high.onnx ^
  --ollama-model phi3:mini
```

**File input**
```bash
python mwe_assistant.py --mode cpu --audio .\budapest_wetter.wav ^
  --vosk-model .\vosk-model-small-de-0.15 ^
  --piper-exe .\piper\piper.exe ^
  --piper-voice .\piper\de_DE-thorsten_high.onnx ^
  --ollama-model phi3:mini
```

POSIX example:
```bash
python mwe_assistant.py --mode cpu --turns 3 --rec-seconds 5   --vosk-model ./vosk-model-small-de-0.15   --piper-exe ./piper   --piper-voice ./de_DE-thorsten_high.onnx   --ollama-model phi3:mini
```

## 3) GPU Mode (Whisper GPU + Coqui XTTS-v2)

---

**1. GPU Dependencies**

Install the required versions:

```bash
pip install TTS==0.22.0
pip install transformers==4.35.2
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

These combinations are verified to work with:

* **faster-whisper GPU**
* **Coqui TTS XTTS-v2**
* **CUDA 12.1 runtime on Windows**

---

**2. GPU Whisper STT**

Whisper will run on GPU when:

```bash
--whisper-device cuda
```

Example:

```bash
python mwe_assistant.py --mode gpu \
  --whisper-device cuda \
  --whisper-model medium \
  --whisper-compute-type float16
```

---

**3. GPU TTS – Coqui XTTS-v2**

Coqui XTTS-v2 provides multilingual, multi-speaker text-to-speech. To use it on GPU:

* `--coqui-voice xtts_v2`
* `--coqui-language de`
* `--coqui-speaker "Aaron Dreschner"` *(default)*

Example command:

```bash
python mwe_assistant.py --mode gpu \
  --whisper-device cuda \
  --whisper-model medium \
  --ollama-model phi3:mini
```

**Listing available speaker names**

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print(tts.speaker_manager.speakers.keys())
```

Choose any speaker and pass it to `--coqui-speaker`.

---

## Arguments (highlights)

- `--mode {cpu,gpu}` (required)
- `--turns N` (default: 3 for mic; 1 when `--audio` is used)
- `--rec-seconds SEC` microphone record duration per turn (default 5.0)
- `--samplerate` mic sample rate (default 16000)
- `--save-inputs` also save recorded user WAVs
- `--out-dir DIR` base folder for run outputs (default: `outputs`)
- `--latency-csv PATH` path to CSV; default is auto‑generated inside `outputs/<timestamp>/`

**CPU mode options**
- `--audio PATH`           input audio file;
- `--vosk-model PATH`      Vosk model directory
- `--piper-exe PATH`       Piper executable
- `--piper-voice PATH`     Piper German voice `.onnx`

**GPU mode options**
- `--audio PATH`           input audio file;
- `--whisper-model NAME`   faster-whisper model name (e.g. `medium`)
- `--whisper-device`       `cuda`/`cpu` (default: `cuda`)
- `--whisper-compute-type` e.g. `float16`, `int8_float16`
- `--coqui-voice`          Coqui voice key (e.g. `xtts_v2`)
- `--coqui-language`       e.g. `de`
- `--coqui-speaker`        e.g. `Aaron Dreschner`

**LLM**
- `--ollama-model NAME`    e.g. `phi3:mini`, `llama3:8b-instruct-q4_K_M`

---

## Outputs
Each run creates `outputs/<YYYYMMDD_HHMMSS>/` containing:
- `assistant_t{turn}.wav`  — synthesized reply audio
- (optional) `user_t{turn}.wav` if `--save-inputs` is used
- `latency_log_<timestamp>.csv` with rows: `ts_iso,mode,turn,stage,duration_ms`

---

## Troubleshooting
- **ffmpeg not found**: install ffmpeg and ensure the `ffmpeg` CLI is on PATH.
- **No text recognized**: check mic selection/levels; try a longer `--rec-seconds`.
- **Piper voice errors**: verify the voice `.onnx` path; use a voice matching your Piper build.
- **Ollama errors**: ensure `ollama serve` is running and the specified model is pulled.
- **GPU mode**: verify CUDA drivers and that your PyTorch build matches your CUDA version.

---

## Useful downloads (direct links)
- **Ollama**: https://ollama.com/download
- **Vosk German model** (small 0.15): https://alphacephei.com/vosk/models
- **Piper releases**: https://github.com/rhasspy/piper/releases
  - German Thorsten voice pack: https://github.com/rhasspy/piper/blob/master/VOICES.md
- **ffmpeg**: https://ffmpeg.org/download.html  (Windows static builds: https://www.gyan.dev/ffmpeg/builds/)
- **faster-whisper**: https://github.com/SYSTRAN/faster-whisper
- **Coqui TTS** (XTTS‑v2): https://github.com/coqui-ai/TTS
- **PyTorch (CUDA install selector)**: https://pytorch.org/get-started/locally/
