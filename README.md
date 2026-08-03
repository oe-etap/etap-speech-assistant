# Offline Speech Assistant MWE (EN) - File Input, CPU/GPU

Existing audio file(s) -> STT -> LLM -> TTS, mostly offline. The only network-like dependency is the local Ollama HTTP API, and Ollama model downloads happen outside this script.

The script is batch-oriented: it does not record from the microphone. It processes one or more pre-recorded English audio files, writes generated response WAVs, and logs latency plus best-effort CPU/RAM/GPU statistics.

## Features

- Configuration via YAML files (`--config`) or CLI arguments
- Logs conversation transcripts (STT and LLM text) to `transcripts.jsonl` and `transcripts.yaml`
- Saves the exact runtime configuration to `config_used.yaml`
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
- System prompts: `prompts/*.txt`, referenced by `system-prompt-file`. These are
  local and not version controlled, so prompt wording can be iterated on freely.
  Without a prompt file the built-in default applies. The prompt actually used is copied to `<out-dir>/system_prompt.txt`, headed by its file name, so a
  measurement stays identifiable.

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

- `--config PATH`: optional; path to a YAML configuration file (CLI arguments override YAML values)
- `--audio PATH [PATH ...]`: required (unless provided in config); one or more input files
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

- Copied input files as `user_input_<n>_<name>`
- Generated assistant responses as `assistant_<n>_<input-stem>.wav`
- `config_used.yaml`: The exact runtime configuration used for the run
- `transcripts.jsonl` and `transcripts.yaml`: Conversation logs containing the recognized STT text and the LLM response
- `latency_log_<timestamp>.csv`

The CSV contains:

- `stage`: one of `stt`, `stt_endpoint_delay`, `llm_ttft`, `llm_first_chunk_fill`, `llm_ttfc`, `tts_first_chunk`, `ttfa`, `llm_eval`, `tts_total`, `e2e_response_ready`
- `duration_ms`: stage duration in milliseconds
- `input_mode`, `audio_pacing`, `utterance_trigger`: how the audio reached the pipeline and what released it to the LLM. Which stages carry a value, and what they include, depends on these, so runs that differ in them must not be pooled. `utterance_trigger` records the behaviour that applied, not the setting that was requested.
- `cpu_percent`: average system-wide CPU load over the stage the row belongs to
- `ram_percent`, `rss_mb`: system RAM in use and this process's resident set, read at the moment the stage finished
- `gpu_util_percent`, `gpu_mem_used_mb`, `gpu_mem_total_mb`, `gpu_name`
- `extra_json`: additional metadata, including input duration and output path where relevant

### TTFA and the end of speech

`ttfa` is the headline latency figure: from the moment the speaker stopped to the first audio out. Everything hangs on locating that moment, which is neither the end of the file nor the point the STT finalizes:

- A recording carries silence past its last word.
- An endpointer has to hear that silence before it can decide the utterance is over, so a microphone keeps delivering audio the whole time it deliberates.

The anchor therefore comes from the STT's own word timings, which both engines report. `extra_json` on the `stt` row carries `trailing_silence_ms`, the gap between the last word and the end of the audio.

`stt_endpoint_delay` measures from the end of speech to the finished transcript. On file input that is the flush once the stream ends; on a microphone it is the endpointer waiting out the silence, which is usually the largest single component of what a user perceives.

Both stages are **blank under `fast` pacing**: the audio does not advance at wall-clock speed there, so no offset within it corresponds to an instant. Use `realtime` for any latency claim; `fast` remains useful for throughput and for the per-stage costs (`stt`, `llm_ttfc`, `tts_first_chunk`, `e2e_response_ready`), which stay valid.

### Trailing silence in input files

For `realtime` pacing to stand in for a microphone, input files need enough silence after the last word for the endpointer to fire — measured at roughly 1100 ms with `vosk-model-small-en-us-0.15`. Below that the engine only finalizes because the file ran out, a signal live input never gets, and the run reports an optimistic TTFA. A warning is printed when this happens.

How much more than that to allow depends on `file-realtime-trigger`:

| `file-realtime-trigger` | what starts the LLM | trailing silence to aim for |
|---|---|---|
| `end-of-file` (default) | the whole file has been read | 1.2–1.5 s, and no more — silence past the endpoint is added to `ttfa` in full |
| `endpoint` | the STT calls the utterance over, as a microphone would | at least 1.2 s, with no upper bound |

`stt_endpoint_delay` is unaffected either way and stays around 1100 ms however long the tail runs.

The setting applies only to file input with realtime pacing. Mic mode always triggers on the endpoint, having no stream end to wait for; `fast` pacing always waits for the file, where doing so costs nothing. It has no effect with Whisper, which has no endpointer and finalizes only when the stream ends.

Under `endpoint`, a second end-of-speech signal within one input carries **everything recognized so far**, not just the new sentence, and supersedes the answer already in flight. An endpointer that fires while the speaker is only drawing breath would otherwise have the LLM answer half a sentence. Note that this is not conversation memory: each request to Ollama is independent, carries no `context` and no message history, and the assistant's own previous replies are never fed back.

## Notes

- CPU/RAM stats require `psutil`; if unavailable, those fields stay blank.
- Resource columns are captured by the worker that owns the stage, at the moment that stage finishes. `psutil.cpu_percent(interval=None)` reports load since its own previous call and keeps that state per thread, so each stage opens its own measurement window with a priming call when it starts.
- A row's CPU window matches its `duration_ms` exactly for `stt`, `llm_ttft`, `llm_first_chunk_fill`, `tts_first_chunk` and `e2e_response_ready`. The rest are approximations: `llm_ttfc` is timed from the start of the request but its CPU window starts at the first token; `ttfa` reuses the `tts_first_chunk` snapshot and `stt_endpoint_delay` the `stt` one; `llm_eval` and `tts_total` carry the snapshot taken when their worker finished.
- GPU stats are read from `nvidia-smi`; if unavailable or no NVIDIA GPU is present, those fields stay blank. A background sampler refreshes them every second, which keeps the subprocess off the measured path. Fields the driver reports as `[N/A]` are stored blank so the numeric columns stay numeric.
- `e2e_response_ready` runs from the start of processing to the complete response, in every mode. On file input it therefore includes the delivery of the audio; on a microphone it covers the whole session. It is a wall-clock span, not a latency — for latency use `ttfa`.
- `stt_rtf` is `stt` divided by the audio duration. Under `realtime` pacing `stt` includes waiting for the audio to arrive, so the ratio exceeds 1 and no longer expresses a real-time factor. Only the `fast` figures do.
- Response length varies enough between runs to hide the effect under test: `llm_eval`, `tts_total` and `e2e_response_ready` scale with it. Pin `llm-temperature: 0` or an `llm-seed` before comparing configurations. `ttfa` and `stt_endpoint_delay` are unaffected, as both conclude before the response length is known.


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
