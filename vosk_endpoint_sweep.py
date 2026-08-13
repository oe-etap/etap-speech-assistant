#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep Vosk's endpointer silence threshold over a corpus of utterances.

The endpointer decides when the speaker has stopped. Set it too high and every
answer is late by the difference; set it too low and it fires on a breath, the
half-sentence goes to the LLM, and the generation has to be thrown away and
restarted once the rest of the words arrive. This script measures where that
boundary sits for a given corpus.

Vosk 0.3.44 has no runtime endpointer API (`SetEndpointerDelays` arrived later),
so the threshold comes from the model's own `conf/model.conf`, read at load
time. Each swept value therefore gets a model directory of its own: every
subdirectory symlinked back to the real model, with a rewritten `model.conf`.
Kaldi's four silence rules fire on an OR, so all three that require decoded
speech are pinned to the same value, which collapses them into the single
"t_end" knob this script sweeps. Rule 1 (silence with nothing recognized at all)
keeps its default, and rule 5 (utterance length cap) is lifted out of the way.

A firing is judged against two independent references:

  - acoustic: Silero VAD (bundled with faster-whisper, runs offline) locates the
    speech. A firing before the last speech segment ends is a cut, whatever the
    recognizer thought it heard.
  - recognizer: more than one non-empty final result means Vosk itself split the
    utterance in two.

The two disagree on recordings the recognizer fails on, which is why both are
reported -- and why the headline figures cover only the files a second engine
confirms Vosk was following. See the note above coverage().

Input files need enough trailing silence for the endpointer to fire before the
audio runs out: about `t_end + 400 ms` at a 250 ms chunk. `never_fired` in the
summary counts the files that fell short, and a value with a nonzero count there
is measuring the end of the file rather than the endpointer.

Usage:
    python vosk_endpoint_sweep.py --audio-dir audios/HeySQuAD_5000/audio \
        --vosk-model vosk/vosk-model-small-en-us-0.15 --include-stock
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Keep the per-worker BLAS/ONNX thread pools at one thread: the parallelism here
# is one process per file, and nested pools only fight over the same cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

# min-trailing-silence for the reference decode. Any value past the longest file
# disables endpointing outright, which is what the reference needs: one segment,
# one word list, covering the whole recording.
NO_ENDPOINT_S = 1000.0

DEFAULT_T_END_MS = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1500]

# Silero segments shorter than this are dropped before the speech boundaries are
# read off. Isolated blips are usually a click or a breath, and letting one
# stand as the end of speech would mark a correct firing as a cut.
VAD_MIN_SPEECH_MS = 100


# --------------------------------------------------------------------------
# model variants
# --------------------------------------------------------------------------

def build_model_variant(src_model: str, dest: str, t_end_s: float) -> str:
    """Materialize a model directory whose endpointer waits `t_end_s` seconds.

    Everything except `conf/` is symlinked, so a variant costs a few hundred
    bytes rather than a copy of the acoustic model.
    """
    src = Path(src_model).resolve()
    dst = Path(dest)
    if dst.is_dir():
        shutil.rmtree(dst)
    (dst / "conf").mkdir(parents=True)

    for entry in src.iterdir():
        if entry.name != "conf":
            os.symlink(entry, dst / entry.name)
    for entry in (src / "conf").iterdir():
        if entry.name != "model.conf":
            os.symlink(entry, dst / "conf" / entry.name)

    base = (src / "conf" / "model.conf").read_text().splitlines()
    lines = [ln for ln in base
             if "endpoint.rule" not in ln and ln.strip()]
    lines += [
        # Rules 2-4 differ only in how confident the decoder has to be; pinning
        # them together makes the endpointer fire on t_end of silence and
        # nothing else.
        f"--endpoint.rule2.min-trailing-silence={t_end_s}",
        f"--endpoint.rule3.min-trailing-silence={t_end_s}",
        f"--endpoint.rule4.min-trailing-silence={t_end_s}",
        # Not under test: without this, a long recording would be endpointed on
        # its length alone and counted as a cut.
        "--endpoint.rule5.min-utterance-length=10000.0",
    ]
    if t_end_s >= NO_ENDPOINT_S:
        # The reference decode wants no endpointing whatsoever, including the
        # "nothing recognized yet" rule that leading silence would otherwise
        # trip.
        lines.append(f"--endpoint.rule1.min-trailing-silence={t_end_s}")
    (dst / "conf" / "model.conf").write_text("\n".join(lines) + "\n")
    return str(dst)


# --------------------------------------------------------------------------
# stage 1: audio + VAD
# --------------------------------------------------------------------------

_VAD_OPTS = None


def _vad_init():
    global _VAD_OPTS
    from faster_whisper.vad import VadOptions
    _VAD_OPTS = VadOptions(threshold=0.5, min_speech_duration_ms=VAD_MIN_SPEECH_MS,
                           min_silence_duration_ms=0, speech_pad_ms=0)


def analyze_audio(path: str) -> dict:
    """Locate the speech in one file and describe the silence around it."""
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps

    with wave.open(path) as wf:
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    pcm = np.frombuffer(raw, dtype=np.int16)
    duration_s = len(pcm) / SAMPLE_RATE

    # The appended pad is exact digital silence, so the run of zero samples at
    # the end separates it from whatever silence the speaker left behind.
    nonzero = np.nonzero(pcm)[0]
    pad_s = (len(pcm) - 1 - nonzero[-1]) / SAMPLE_RATE if len(nonzero) else duration_s

    audio = pcm.astype(np.float32) / 32768.0
    segments = [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE)
                for t in get_speech_timestamps(audio, _VAD_OPTS)]

    gaps = [round(b[0] - a[1], 3) for a, b in zip(segments, segments[1:])]
    return {
        "file": os.path.basename(path),
        "duration_s": round(duration_s, 3),
        "trailing_zero_pad_s": round(pad_s, 3),
        "vad_n_segments": len(segments),
        "vad_speech_start_s": round(segments[0][0], 3) if segments else None,
        "vad_speech_end_s": round(segments[-1][1], 3) if segments else None,
        "vad_speech_total_s": round(sum(e - s for s, e in segments), 3),
        "vad_max_gap_s": max(gaps) if gaps else 0.0,
        "vad_gaps_s": gaps,
        "vad_segments_s": [(round(s, 3), round(e, 3)) for s, e in segments],
    }


# --------------------------------------------------------------------------
# stage 2/3: decode
# --------------------------------------------------------------------------

_MODEL = None
_CHUNK_FRAMES = None


def _decode_init(model_dir: str, chunk_ms: int):
    global _MODEL, _CHUNK_FRAMES
    from vosk import Model, SetLogLevel
    SetLogLevel(-1)
    _MODEL = Model(model_dir)
    _CHUNK_FRAMES = int(SAMPLE_RATE * chunk_ms / 1000)


def decode(path: str) -> dict:
    """Run one file through the recognizer, recording every endpoint firing.

    The chunk loop mirrors `VoskEngine.transcribe_stream`: `AcceptWaveform`
    returning True *is* the endpointer, and the position reached in the audio
    when it does is when a live microphone would have released the utterance.
    Word timestamps stay absolute across firings — Vosk carries a frame offset
    over the reset — so times from different segments are directly comparable.
    """
    from vosk import KaldiRecognizer

    rec = KaldiRecognizer(_MODEL, SAMPLE_RATE)
    rec.SetWords(True)

    segments = []
    pos_s = 0.0
    with wave.open(path) as wf:
        while True:
            data = wf.readframes(_CHUNK_FRAMES)
            if not data:
                break
            pos_s += len(data) / SAMPLE_WIDTH / SAMPLE_RATE
            if rec.AcceptWaveform(data):
                segments.append(_segment(json.loads(rec.Result()), pos_s, False))
    segments.append(_segment(json.loads(rec.FinalResult()), pos_s, True))

    spoken = [s for s in segments if s["text"]]
    words = [w for s in spoken for w in s["words"]]
    return {
        "file": os.path.basename(path),
        "audio_end_s": round(pos_s, 3),
        "segments": segments,
        "n_spoken_segments": len(spoken),
        "text": " ".join(s["text"] for s in spoken),
        "first_word_start_s": spoken[0]["first_word_start_s"] if spoken else None,
        "last_word_end_s": spoken[-1]["last_word_end_s"] if spoken else None,
        "n_words": len(words),
        "word_duration_s": round(sum(w[2] - w[1] for w in words), 3),
        "word_gaps_s": [round(b[1] - a[2], 3) for a, b in zip(words, words[1:])],
    }


def _segment(result: dict, pos_s: float, is_flush: bool) -> dict:
    words = result.get("result") or []
    return {
        "at_s": round(pos_s, 3),          # where the audio stood when this landed
        "flush": is_flush,                # end of stream, not the endpointer
        "text": result.get("text", ""),
        "n_words": len(words),
        "first_word_start_s": words[0]["start"] if words else None,
        "last_word_end_s": words[-1]["end"] if words else None,
        "words": [(w["word"], w["start"], w["end"]) for w in words],
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(run: dict, audio: dict) -> dict:
    """Judge one decode against the acoustic and recognizer references."""
    fires = [s for s in run["segments"] if not s["flush"]]
    spoken_fires = [s for s in fires if s["text"]]

    speech_end = audio["vad_speech_end_s"]
    # A firing before the last speech segment ends had speech still to come.
    acoustic_cuts = [s for s in fires
                     if speech_end is not None and s["at_s"] < speech_end - 1e-6]

    # The firing that closed the utterance, and how long after the last word it
    # came: this is what `stt_endpoint_delay` costs in the live pipeline.
    closing = spoken_fires[-1] if spoken_fires else None
    delay_ms = None
    if closing and closing["last_word_end_s"] is not None:
        delay_ms = round((closing["at_s"] - closing["last_word_end_s"]) * 1000, 1)

    # How much text a premature firing would have shipped to the LLM. One or two
    # words is a false start the pipeline could reject on its own; eight words
    # is half a question, and retracting that is the expensive case.
    words_at_cut = None
    if acoustic_cuts:
        upto = run["segments"].index(acoustic_cuts[0]) + 1
        words_at_cut = sum(s["n_words"] for s in run["segments"][:upto])

    return {
        "file": run["file"],
        "n_fires": len(fires),
        "n_spoken_segments": run["n_spoken_segments"],
        "asr_split": run["n_spoken_segments"] > 1,
        "acoustic_cut": bool(acoustic_cuts),
        "first_cut_s": acoustic_cuts[0]["at_s"] if acoustic_cuts else None,
        "words_at_first_cut": words_at_cut,
        "endpointer_fired": closing is not None,
        "endpoint_delay_ms": delay_ms,
        "text": run["text"],
    }


# --------------------------------------------------------------------------
# which recordings the question applies to
# --------------------------------------------------------------------------
#
# A firing is only premature if the recognizer was following the speech to begin
# with. Where it is not, it splits wherever it loses the thread, at every
# threshold, and those files pile up as a floor the sweep can never push through
# -- on this corpus they held the apparent cut rate at 4% when the real figure
# was 0.17%. Keeping them in does not overstate the risk conservatively, it
# hides the knee entirely, so the subset has to be drawn before anything is
# read off the sweep.

def coverage(audio: dict, reference: dict) -> float:
    """How much of the speech the VAD found ended up inside a recognized word.

    Catches the recordings Vosk gives up on outright -- Silero hears seconds of
    speech, Vosk accounts for a fraction of it. It does *not* catch the ones
    where Vosk stays confident and wrong, emitting a stream of short filler
    words that covers the speech densely enough to pass. Those need a second
    opinion, which is what `agrees_with_whisper` is for.
    """
    if not audio["vad_speech_total_s"]:
        return 0.0
    return round(reference["word_duration_s"] / audio["vad_speech_total_s"], 3)


def agrees_with_whisper(reference: dict, whisper: dict, max_wer: float) -> bool:
    """Whether a second engine heard the same words.

    Whisper has no endpointer and no stake in the outcome, which is exactly what
    makes it usable as the judge here. Its transcript is treated as the
    reference and Vosk's as the hypothesis; the two agreeing means the audio
    carries intelligible speech *and* Vosk tracked it.
    """
    if not whisper or whisper["n_words"] < 5 or (whisper["mean_prob"] or 0) < 0.5:
        return False
    return wer(whisper["text"], reference["text"]) <= max_wer


def wer(ref: str, hyp: str) -> float:
    """Word error rate of `hyp` against `ref`, on lightly normalized text."""
    def tokens(text):
        return [t for t in re.sub(r"[^a-z0-9' ]", " ", text.lower()).split() if t]

    r, h = tokens(ref), tokens(hyp)
    if not r:
        return 1.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return round(prev[-1] / len(r), 3)


# --------------------------------------------------------------------------
# stage 1b: the second opinion
# --------------------------------------------------------------------------

_WHISPER = None


def _whisper_init(model_name: str):
    global _WHISPER
    from faster_whisper import WhisperModel
    _WHISPER = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=1)


def whisper_decode(path: str) -> dict:
    import numpy as np

    with wave.open(path) as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    segments, _ = _WHISPER.transcribe(pcm.astype(np.float32) / 32768.0, language="en",
                                      beam_size=5, word_timestamps=True,
                                      vad_filter=False, condition_on_previous_text=False)
    words = [(w.word.strip(), round(w.start, 3), round(w.end, 3), round(w.probability, 3))
             for s in segments for w in (s.words or [])]
    return {
        "file": os.path.basename(path),
        "text": " ".join(w[0] for w in words),
        "n_words": len(words),
        # An independent read of the pause structure, for comparison with the
        # VAD's and Vosk's own.
        "word_gaps_s": [round(b[1] - a[2], 3) for a, b in zip(words, words[1:])],
        "mean_prob": round(sum(w[3] for w in words) / len(words), 3) if words else None,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_pool(fn, init, init_args, files, jobs, label):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs, initializer=init,
                             initargs=init_args) as pool:
        out = list(pool.map(fn, files, chunksize=4))
    print(f"  {label}: {len(out)} files in {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--vosk-model", required=True)
    ap.add_argument("--t-end-ms", type=int, nargs="+", default=DEFAULT_T_END_MS,
                    help="endpointer silence thresholds to sweep, in ms")
    ap.add_argument("--chunk-ms", type=int, nargs="+", default=[250],
                    help="chunk sizes to sweep; the endpointer can only fire on a "
                         "chunk boundary, so this bounds its time resolution")
    ap.add_argument("--include-stock", action="store_true",
                    help="also measure the model's shipped endpointer settings, as "
                         "the baseline any swept value has to beat. Reported as "
                         "t_end_ms=0")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="least of the VAD-detected speech the reference decode has "
                         "to account for before a file counts toward the headline "
                         "figures; see coverage()")
    ap.add_argument("--max-wer", type=float, default=0.5,
                    help="most a file's Vosk transcript may differ from Whisper's "
                         "and still count as followed")
    ap.add_argument("--whisper-model", default="small",
                    help="faster-whisper model for the second opinion")
    ap.add_argument("--no-whisper", action="store_true",
                    help="skip the second opinion. Faster, but the cut rates come "
                         "out several times too high: see the note above coverage()")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"outputs/endpoint_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    files = sorted(str(p) for p in Path(args.audio_dir).glob("*.wav"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no wav files in {args.audio_dir}")
    print(f"{len(files)} files, {args.jobs} workers -> {out_dir}", flush=True)

    print("stage 1: locating speech (Silero VAD)", flush=True)
    audio = {a["file"]: a for a in
             run_pool(analyze_audio, _vad_init, (), files, args.jobs, "vad")}

    print("stage 2: reference decode (endpointing disabled)", flush=True)
    ref_model = build_model_variant(args.vosk_model, models_dir / "reference", NO_ENDPOINT_S)
    reference = {r["file"]: r for r in
                 run_pool(decode, _decode_init, (ref_model, args.chunk_ms[0]),
                          files, args.jobs, "reference")}

    whisper = {}
    if not args.no_whisper:
        print(f"stage 2b: second opinion (faster-whisper {args.whisper_model})", flush=True)
        try:
            whisper = {w["file"]: w for w in
                       run_pool(whisper_decode, _whisper_init, (args.whisper_model,),
                                files, args.jobs, "whisper")}
        except Exception as exc:                       # model missing, no disk, ...
            print(f"  unavailable ({exc}); falling back to coverage alone, which "
                  f"reports cut rates several times too high", flush=True)

    usable = set()
    for name in audio:
        ok = (audio[name]["vad_speech_total_s"] >= 1.0
              and reference[name]["n_words"] >= 4
              and coverage(audio[name], reference[name]) >= args.min_coverage)
        if ok and whisper:
            ok = agrees_with_whisper(reference[name], whisper.get(name), args.max_wer)
        if ok:
            usable.add(name)
    print(f"  usable for the endpointer question: {len(usable)}/{len(files)}", flush=True)

    with (out_dir / "files.jsonl").open("w") as fh:
        for name in sorted(audio):
            w = whisper.get(name, {})
            fh.write(json.dumps({**audio[name],
                                 "usable": name in usable,
                                 "asr_coverage": coverage(audio[name], reference[name]),
                                 "reference_text": reference[name]["text"],
                                 "reference_n_words": reference[name]["n_words"],
                                 "reference_word_duration_s": reference[name]["word_duration_s"],
                                 "reference_first_word_start_s": reference[name]["first_word_start_s"],
                                 "reference_last_word_end_s": reference[name]["last_word_end_s"],
                                 "reference_word_gaps_s": reference[name]["word_gaps_s"],
                                 "whisper_text": w.get("text"),
                                 "whisper_n_words": w.get("n_words"),
                                 "whisper_mean_prob": w.get("mean_prob"),
                                 "whisper_word_gaps_s": w.get("word_gaps_s"),
                                 "wer_vs_whisper": (wer(w["text"], reference[name]["text"])
                                                    if w.get("n_words") else None)}) + "\n")

    print("stage 3: sweep", flush=True)
    rows = []
    detail = (out_dir / "sweep.jsonl").open("w")
    for chunk_ms in args.chunk_ms:
        swept = ([0] if args.include_stock else []) + sorted(args.t_end_ms)
        for t_end_ms in swept:
            if t_end_ms == 0:
                # The model as shipped: rules 2/3/4 at 0.5/0.75/1.0 s, each
                # gated on a different decoder confidence, so no single number
                # describes it. Measured, not swept.
                variant, label = args.vosk_model, "stock config"
            else:
                variant = build_model_variant(args.vosk_model,
                                              models_dir / f"te_{t_end_ms}",
                                              t_end_ms / 1000.0)
                label = f"t_end={t_end_ms}ms"
            runs = run_pool(decode, _decode_init, (variant, chunk_ms), files,
                            args.jobs, f"chunk={chunk_ms}ms {label}")
            scored = [score(r, audio[r["file"]]) for r in runs]
            for s, r in zip(scored, runs):
                detail.write(json.dumps({"chunk_ms": chunk_ms, "t_end_ms": t_end_ms,
                                         **s, "segments": r["segments"]}) + "\n")
            rows.append(summarize(scored, chunk_ms, t_end_ms, usable))
            print("    " + fmt_row(rows[-1]), flush=True)
    detail.close()

    fields = list(rows[0].keys())
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_dir}/summary.csv", flush=True)


def summarize(scored, chunk_ms, t_end_ms, usable) -> dict:
    subset = [s for s in scored if s["file"] in usable]
    delays = [s["endpoint_delay_ms"] for s in subset if s["endpoint_delay_ms"] is not None]
    return {
        "chunk_ms": chunk_ms,
        "t_end_ms": t_end_ms,
        "n_all": len(scored),
        "n_usable": len(subset),
        "cut_all": sum(s["acoustic_cut"] for s in scored),
        "cut_usable": sum(s["acoustic_cut"] for s in subset),
        "cut_usable_pct": round(100 * sum(s["acoustic_cut"] for s in subset) / max(len(subset), 1), 2),
        "split_usable": sum(s["asr_split"] for s in subset),
        "split_usable_pct": round(100 * sum(s["asr_split"] for s in subset) / max(len(subset), 1), 2),
        "never_fired_usable": sum(not s["endpointer_fired"] for s in subset),
        "delay_p50_ms": percentile(delays, 0.50),
        "delay_p90_ms": percentile(delays, 0.90),
        "delay_mean_ms": round(statistics.fmean(delays), 1) if delays else None,
    }


def fmt_row(r) -> str:
    return (f"cut {r['cut_usable']:>4}/{r['n_usable']} ({r['cut_usable_pct']:>5.2f}%)  "
            f"split {r['split_usable_pct']:>5.2f}%  "
            f"delay p50 {r['delay_p50_ms']}ms p90 {r['delay_p90_ms']}ms")


if __name__ == "__main__":
    main()
