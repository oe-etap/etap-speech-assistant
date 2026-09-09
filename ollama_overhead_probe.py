#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate what a request costs the model from what it costs to reach it.

Ollama times every request server-side and reports the parts: `load_duration`,
`prompt_eval_duration` and `eval_duration`, which together should account for
`total_duration`. On a model that is already resident `load_duration` is meant to
be noise -- there is nothing left to load. On Ollama 0.32.9, which is what
`outputs/sub1` was measured on, it was not: subtracting `prompt_eval` from the
logged `llm_ttft` left 711-843 ms for every llama and qwen configuration and
1629-1632 ms for every gemma3 one, flat across model sizes from 1B to 32B. A
constant that large sits inside every `ttfa` in that series, and it belongs to
none of the models under test.

This script measures it two ways and prints both:

  - through Ollama, client-side, streaming: the wall time from sending the
    request to the first token arriving, next to the server's own breakdown of
    where that time went.
  - to the llama.cpp runner Ollama started, directly on its port, with the same
    prompt while the same model is resident. Same process, same weights, same
    warm KV cache; the only thing removed is Ollama in front of it.

The difference is what a request pays to get to a model that is already loaded.
On a GRID V100DX-32C it was 813 ms for llama3.2:1b, 1780 ms for gemma3:1b and
1764 ms for gemma3:27b under Ollama 0.32.9 -- in each case larger than the
model's own time to first token -- and 19, 17 and -40 ms under 0.33.3, which is
run-to-run noise. The defect was fixed upstream, so this is a regression check
rather than a standing finding: run it after a server upgrade, and before
trusting a llm_ttft figure measured across one.

Two things the direct figure is not. It skips the chat template Ollama applies,
which is string work on a couple of kilobytes rather than hundreds of
milliseconds, and it skips Ollama's scheduling, which is the thing being
measured. Treat it as the floor the runner is capable of, not as a drop-in
replacement for what Ollama does.

What the overhead did not depend on, all measured on 0.32.9: prompt length (a
37-token prompt cost the same as a 320-token one), `num_ctx` at 1024, 2048 or
4096, whether options are sent at all, streaming or not, and whether the model
is on the GPU or on the CPU. Nothing a caller controls moved it, which is why
this compares transports rather than settings.

The prompt_eval column is worth reading on its own, and it outlived the fix.
Where the prefix cache works -- llama and qwen -- a long system prompt is
evaluated once and costs nothing on later requests, so prompt length is free.
gemma3 re-evaluates it every time, on both versions.

Usage:
    python ollama_overhead_probe.py llama3.2:1b-instruct-q4_K_M gemma3:27b-it-q4_K_M
    python ollama_overhead_probe.py --unload qwen2.5:7b-instruct-q4_K_M
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time

import requests

DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_PROMPT_FILE = "./prompts/short-opener.txt"
# Any utterance does; this one is the first recording of the evaluation set, so
# the numbers line up with what the latency logs hold for it.
DEFAULT_QUESTION = "what does the name of the campus tv station"


def build_prompt(system_prompt_file, question):
    """The framing assistant.py sends, so the measurement matches a real run.

    Reproduced here rather than imported: this script has to run against a
    server, not a pipeline, and importing llm_engine would pull in the whole
    engine to reuse three lines of string formatting.
    """
    system_prompt = open(system_prompt_file, encoding="utf-8").read()
    return (f'{system_prompt}\n\n'
            f'The user said: "{question}"\n\n'
            f'Answer:')


def via_ollama(url, model, prompt, options, keep_alive):
    """Client-measured TTFT and the server's breakdown, from one streamed call.

    Streaming rather than a plain request because that is what the pipeline
    does, and because the first token's arrival is the quantity of interest;
    a non-streaming call only returns once the whole answer is written.
    """
    started = time.perf_counter()
    first_token_ms = None
    final = {}
    r = requests.post(url, stream=True, timeout=600,
                      json={"model": model, "prompt": prompt, "stream": True,
                            "keep_alive": keep_alive, "options": options})
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        message = json.loads(line)
        if message.get("response") and first_token_ms is None:
            first_token_ms = (time.perf_counter() - started) * 1000
        if message.get("done"):
            final = message
    return {
        "ttft_ms": first_token_ms,
        "load_ms": final.get("load_duration", 0) / 1e6,
        "prompt_eval_ms": final.get("prompt_eval_duration", 0) / 1e6,
        "prompt_tokens": final.get("prompt_eval_count", 0),
        "eval_ms": final.get("eval_duration", 0) / 1e6,
    }


def runner_ports():
    """Ports of the llama-server processes Ollama has running, newest last.

    Ollama starts one per resident model and passes it `--port`. With
    OLLAMA_MAX_LOADED_MODELS=1 there is at most one, which is the case this can
    resolve without guessing; more than one and the caller is told to name the
    port, since nothing on the command line says which model a runner holds
    except a blob digest this script has no business resolving.
    """
    listing = subprocess.run(["pgrep", "-a", "llama-server"],
                             capture_output=True, text=True).stdout
    ports = []
    for line in listing.splitlines():
        found = re.search(r"--port (\d+)", line)
        if found:
            ports.append(found.group(1))
    return ports


def direct(port, prompt, num_predict):
    """TTFT from the runner itself, over llama.cpp's own completion endpoint."""
    started = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/completion", stream=True,
                      timeout=120,
                      json={"prompt": prompt, "n_predict": num_predict,
                            "temperature": 0, "stream": True})
    r.raise_for_status()
    for line in r.iter_lines():
        # Server-sent events: one "data: {...}" per token.
        if line and line.startswith(b"data: "):
            if json.loads(line[6:]).get("content"):
                return (time.perf_counter() - started) * 1000
    return None


def unload(url, model):
    """Drop the model from VRAM, for a probe run on a card someone else shares."""
    try:
        requests.post(url, timeout=120,
                      json={"model": model, "prompt": "", "stream": False,
                            "keep_alive": 0})
    except requests.RequestException as e:
        print(f"[WARN] Could not unload {model}: {e}")


def probe(args, model, prompt, options):
    print(f"\n{model}")

    # The first call loads the model and fills the prefix cache. Both are
    # one-off costs that would otherwise land in the first sample.
    try:
        via_ollama(args.url, model, prompt, options, args.keep_alive)
    except requests.RequestException as e:
        print(f"  [WARN] {model} did not answer: {e}")
        return
    through = [via_ollama(args.url, model, prompt, options, args.keep_alive)
               for _ in range(args.repeats)]

    median = lambda key: statistics.median(sample[key] for sample in through)
    print(f"  ollama   TTFT {median('ttft_ms'):7.0f} ms"
          f"   load {median('load_ms'):6.0f}"
          f"   prompt_eval {median('prompt_eval_ms'):6.0f}"
          f" over {through[0]['prompt_tokens']} tokens"
          f"   eval {median('eval_ms'):6.0f}")

    port = args.runner_port
    if port is None:
        ports = runner_ports()
        if not ports:
            print("  runner   no llama-server found; "
                  "is the server on another machine?")
            return
        if len(ports) > 1:
            print(f"  runner   {len(ports)} runners are up ({', '.join(ports)}); "
                  f"pass --runner-port to say which one holds this model")
            return
        port = ports[0]

    try:
        direct(port, prompt, args.num_predict)
        bare = statistics.median(direct(port, prompt, args.num_predict)
                                 for _ in range(args.repeats))
    except requests.RequestException as e:
        print(f"  runner   port {port} did not answer: {e}")
        return

    print(f"  runner   TTFT {bare:7.0f} ms   (port {port}, Ollama not in front)")
    print(f"  overhead      {median('ttft_ms') - bare:7.0f} ms per request")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="Ollama model tags to probe")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--system-prompt-file", default=DEFAULT_PROMPT_FILE)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--repeats", type=int, default=3,
                    help="samples per transport, after a warm-up call (default 3)")
    ap.add_argument("--num-ctx", type=int, default=1024)
    ap.add_argument("--num-predict", type=int, default=150)
    ap.add_argument("--keep-alive", default="5m",
                    help="how long the model stays resident after the probe")
    ap.add_argument("--runner-port",
                    help="port of the llama-server to query directly; found "
                         "automatically when exactly one is running")
    ap.add_argument("--unload", action="store_true",
                    help="drop each model from VRAM when its probe finishes")
    args = ap.parse_args()

    try:
        prompt = build_prompt(args.system_prompt_file, args.question)
    except OSError as e:
        sys.exit(f"Could not read the system prompt: {e}")

    options = {"temperature": 0, "num_predict": args.num_predict,
               "num_ctx": args.num_ctx}

    for model in args.models:
        probe(args, model, prompt, options)
        if args.unload:
            unload(args.url, model)


if __name__ == "__main__":
    main()
