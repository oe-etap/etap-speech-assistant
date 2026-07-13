# Offline Speech Assistant MWE (EN) - File Input, CPU/GPU

Existing audio file(s) -> STT -> LLM -> TTS, mostly offline. The only network-like dependency is the local Ollama HTTP API, and Ollama model downloads happen outside this script.

The script is batch-oriented: it does not record from the microphone. It processes one or more pre-recorded English audio files, writes generated response WAVs, and logs latency plus best-effort CPU/RAM/GPU statistics.

## Features

- File-only input via `--audio file1.wav file2.mp3 ...`
- English-only STT prompt flow and English TTS defaults
- Selectable STT engine: `--stt-engine vosk` or `--stt-engine whisper`
- Selectable TTS engine: `--tts-engine piper` or `--tts-engine coqui`
- CPU/GPU preset via `--mode cpu|gpu`
- Whisper can run on CPU or CUDA via `--whisper-device cpu|cuda`
- Input audio is normalized to mono 16 kHz WAV using `ffmpeg`
- Per-stage timing: `stt`, `llm`, `tts`
- End-to-end timing: `e2e_response_ready`, measured from the end of the input audio being available to the response audio file being ready
- Best-effort resource stats in CSV: CPU %, RAM %, process RSS, and NVIDIA GPU utilization/memory when available

## Requirements

Common:

- Python 3.10
- `ffmpeg` on PATH
- Ollama running locally, for example `ollama serve`
- A pulled Ollama model, for example `ollama pull phi3:mini`

CPU package setup:

```bash
pip install -r requirements_cpu.txt
```

GPU package setup:

```bash
pip install -r requirements_gpu.txt
```

Additional assets:

- For Vosk: an English model, for example `vosk-model-small-en-us-0.15`
- For Piper: an English voice, for example `en_US-lessac-medium.onnx`
- For Coqui XTTS-v2: the model is loaded through the `TTS` package
- For GPU stats: `nvidia-smi` must be available

## Examples

Whisper on CPU with Piper:

```bash
python mwe_assistant.py --audio .\sample_en.wav ^
  --mode cpu ^
  --stt-engine whisper ^
  --whisper-device cpu ^
  --whisper-model small ^
  --whisper-compute-type int8 ^
  --tts-engine piper ^
  --piper-exe .\piper\piper.exe ^
  --piper-voice .\piper\en_US-lessac-medium.onnx ^
  --ollama-model phi3:mini
```

Vosk on CPU with Piper:

```bash
python mwe_assistant.py --audio .\sample_en.wav ^
  --mode cpu ^
  --stt-engine vosk ^
  --vosk-model .\vosk-model-small-en-us-0.15 ^
  --tts-engine piper ^
  --piper-exe .\piper\piper.exe ^
  --piper-voice .\piper\en_US-lessac-medium.onnx
```

Whisper on CUDA with Coqui:

```bash
python mwe_assistant.py --audio .\sample_en.wav ^
  --mode gpu ^
  --stt-engine whisper ^
  --whisper-device cuda ^
  --whisper-model medium ^
  --whisper-compute-type float16 ^
  --tts-engine coqui ^
  --coqui-language en ^
  --coqui-speaker "Daisy Studious"
```

Multiple files:

```bash
python mwe_assistant.py --audio .\a.wav .\b.mp3 .\c.flac --stt-engine whisper --tts-engine piper
```

## Important Arguments

- `--audio PATH [PATH ...]`: required; one or more input files
- `--mode {cpu,gpu}`: default device preset
- `--stt-engine {vosk,whisper}`: speech-to-text engine
- `--tts-engine {piper,coqui}`: text-to-speech engine
- `--out-dir DIR`: base output directory, default `outputs`
- `--latency-csv PATH`: custom CSV path
- `--keep-normalized`: keep the intermediate mono 16 kHz WAV files
- `--vosk-model PATH`: English Vosk model directory
- `--whisper-model NAME`: faster-whisper model name, default `small`
- `--whisper-device {cpu,cuda}`: explicit Whisper device
- `--whisper-compute-type TYPE`: for example `int8` on CPU or `float16` on CUDA
- `--piper-exe PATH`: Piper executable
- `--piper-voice PATH`: English Piper `.onnx` voice
- `--coqui-voice NAME`: default `xtts_v2`
- `--coqui-language CODE`: default `en`
- `--coqui-speaker NAME`: default `Daisy Studious`
- `--ollama-model NAME`: default `phi3:mini`
- `--ollama-url URL`: default `http://localhost:11434/api/generate`

## Outputs

Each run creates `outputs/<YYYYMMDD_HHMMSS>/` containing:

- Copied input files as `input_<n>_<name>`
- Generated assistant responses as `assistant_<n>_<input-stem>.wav`
- `latency_log_<timestamp>.csv`

The CSV contains:

- `stage`: `stt`, `llm`, `tts`, or `e2e_response_ready`
- `duration_ms`: stage duration in milliseconds
- `cpu_percent`, `ram_percent`, `rss_mb`
- `gpu_util_percent`, `gpu_mem_used_mb`, `gpu_mem_total_mb`, `gpu_name`
- `extra_json`: additional metadata, including input duration and output path where relevant

## Notes

- CPU/RAM stats require `psutil`; if unavailable, those fields stay blank.
- GPU stats are read from `nvidia-smi`; if unavailable or no NVIDIA GPU is present, those fields stay blank.
- The end-to-end metric is measured in batch mode after the full input audio file is available, so it represents processing latency from input-audio end to response-audio readiness.


# Metric Descriptions

## Timeline

```
User finishes speaking
  │
  ├─── STT ───────┬─── LLM TTFC ───┬──── TTS₁ ──┐
  │    (stt)      │   (llm_ttfc)   │(tts_first) │
  │               │                │            │
  │               │                │   TTFA ◄───┘
  │               │                │
  │               │  ┌─── LLM eval ──────────────┐  ← Ollama server-side
  │               │  │   (llm_eval)              │
  │               │  └───────────────────────────┘
  │               │                │
  │               ├─── TTS₂ ──┬─── TTS₃ ──┐
  │               │           │           │
  │               └───────────┴───────────┘
  │                  (tts_total = Σ TTS)
  │                                        │
  ├────────────────────────────────────────┘
  │              e2e_response_ready
```

---

## CSV Stages

### `stt` – Speech-to-Text processing time

| | |
|---|---|
| **What it measures** | The net time to convert the input audio to text (Vosk or Whisper). |
| **Measurement point** | Before → after the STT function call. |
| **duration_ms** | STT net processing time. |
| **extra_json** | `input_duration_ms` – length of input audio (ms); `stt_rtf` – Real-Time Factor (stt / input_duration; <1.0 = faster than real-time). |

---

### `llm_ttfc` – LLM Time to First (synthesizable) Chunk

| | |
|---|---|
| **What it measures** | The time elapsed from starting the LLM request until the first synthesizable text chunk (sentence) arrives at the client side. Includes sending the HTTP request, Ollama server prompt processing, and the tokens of the first complete sentence. |
| **Measurement point** | Starting LLM streaming request → arrival of the first chunk from the generator. |
| **duration_ms** | Prompt → first sentence client-side latency. |
| **extra_json** | – |

---

### `tts_first_chunk` – TTS first chunk synthesis time

| | |
|---|---|
| **What it measures** | The time to convert the first LLM sentence into speech. This is the last component of TTFA: after this, the user would first hear a response. |
| **Measurement point** | Before → after the TTS function call (for the first chunk only). |
| **duration_ms** | TTS synthesis time of the first sentence. |
| **extra_json** | – |

---

### `ttfa` – Time to First Audio ⭐

| | |
|---|---|
| **What it measures** | The most important metric from a user perspective: how long they have to wait to hear the first word of the response. Calculated metric: `stt + llm_ttfc + tts_first_chunk`. |
| **Measurement point** | Calculated (not directly measured). |
| **duration_ms** | `stt + llm_ttfc + tts_first_chunk` |
| **extra_json** | – |

---

### `llm_eval` – LLM server-side token generation time

| | |
|---|---|
| **What it measures** | The net token generation time (eval_duration) measured on the Ollama server. This is ground truth: it does not include network latency, client-side processing, or TTS time. |
| **Measurement point** | Read from Ollama's `done: true` message (server-side measurement). |
| **duration_ms** | Net token generation time on the server. |
| **extra_json** | `eval_tokens` – number of generated tokens; `tokens_per_sec` – generation speed (tok/s); `prompt_tokens` – number of prompt tokens; `prompt_eval_ms` – prompt processing time (ms); `total_duration_ms` – total Ollama server-side time (load + prompt eval + eval + overhead). |

---

### `tts_total` – TTS total synthesis time

| | |
|---|---|
| **What it measures** | The sum of the synthesis times of all TTS chunks. Net TTS time, does not include LLM waiting or I/O. |
| **Measurement point** | Σ (Before → after TTS call) for every chunk. |
| **duration_ms** | Total TTS synthesis time. |
| **extra_json** | – |

---

### `e2e_response_ready` – End-to-End processing time

| | |
|---|---|
| **What it measures** | The total processing time: from the availability of the normalized audio to the closing of the final response WAV file. In batch mode, this is the "processing latency". |
| **Measurement point** | After audio normalization → closing WAV file + final write_timing calls. |
| **duration_ms** | Total STT + LLM streaming + TTS streaming time. |
| **extra_json** | `input_duration_ms` – length of input audio; `output_duration_ms` – length of output (response) audio; `output_wav` – path of the generated WAV file; `full_text` – full text response of the LLM; `response_word_count` – number of words in the response; `response_char_count` – number of characters in the response; `llm_chunk_count` – number of sentence-level chunks. |

---

## Relationships Between Metrics

| Relationship | Always true? |
|-------------|-------------|
| `ttfa = stt + llm_ttfc + tts_first_chunk` | ✅ By definition |
| `e2e >= ttfa` | ✅ E2E includes TTFA + the processing of the remaining chunks |
| `tts_total >= tts_first_chunk` | ✅ The total is the sum of all chunks |
| `llm_eval < total_duration_ms` (extra_json) | ✅ The total also includes prompt eval and overhead |
| `llm_ttfc >= llm_eval` | ❌ Not always! TTFC measures until the first sentence arrives, eval is the generation time of **all** tokens. For short responses TTFC ≈ eval; for long responses TTFC < eval. |
| `tokens_per_sec ≈ eval_tokens / (llm_eval / 1000)` | ✅ By definition |
| `stt_rtf = stt / input_duration_ms` | ✅ By definition |
