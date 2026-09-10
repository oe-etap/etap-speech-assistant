#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the metadata.csv a recording set needs to be measurable.

`assistant.py` reads two columns from a `metadata.csv` sitting beside the audio:
`question`, which is what the recording was supposed to say, and `speech_end_ms`,
which is where the speech ends. The second one is what `ttfa` and
`stt_endpoint_delay` are anchored on, and without it they fall back to the
recognizer's own word timings -- a by-product of decoding rather than a
measurement of the audio, missing entirely where the recognizer failed, and
absent altogether under Whisper. Runs anchored differently are measuring from
different instants and must not be pooled, so a set without this file is a set
whose latency figures cannot be compared with anything.

The corpora ship with the text but not the timings, which is what this fills in.
Speech is located with Silero VAD through faster-whisper, at the settings
`vosk_endpoint_sweep.py` uses, so the two agree on where an utterance ends.

Get the audio right first. Every offset here is measured on the file as it sits
on disk, while the pipeline feeds the STT a mono 16 kHz conversion of it -- and
where those two disagree, every timing is wrong by the ratio between them. A
multi-channel file is downmixed here for exactly that reason, and counted in the
summary; the 24 stereo recordings in `audios/` were written by a script that
counted interleaved samples as frames instead, which put `speech_end_ms` at
twice its real value and produced negative TTFAs on a fifth of the corpus. A
sample rate other than 16 kHz is refused rather than guessed at.

The summary also counts the recordings whose trailing silence is too short for
the endpointer to fire. Those measure the end of the file, which live microphone
input never reaches, and their TTFA is optimistic; see
`min_trailing_silence_ms()` in assistant.py for where the default comes from.

Not reproduced from the older `audios/metadata.csv`: `appended_pad_ms`, which
belongs to a padding step this does not perform, and `vosk_last_word_end_ms` /
`vosk_minus_vad_ms`, which are `vosk_endpoint_sweep.py`'s cross-check of this
VAD against a recognizer.

Usage:
    python make_metadata.py --audio-dir audios/data-v3/heysquad_filtered_verified_audio_16khz_mono \\
        --source-csv audios/data-v3/heysquad_filtered_answerable_vad_asr_verified.csv
"""

import argparse
import csv
import os
import sys
import wave

SAMPLE_RATE = 16000

# Same options as vosk_endpoint_sweep.py, so an utterance ends in the same place
# for both. No padding and no silence merging: the boundaries wanted here are the
# ones the audio actually has.
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 100

# Trailing silence an endpointer needs before the file runs out, at
# --vosk-endpoint-silence-ms 600 and --audio-chunk-ms 100. Mirrors
# min_trailing_silence_ms() in assistant.py, which is where the derivation is;
# re-deriving it here keeps this script runnable without vosk or piper installed.
DEFAULT_MIN_TRAILING_MS = 1065

# What this script measures. Anything else in the source CSV is carried through
# ahead of these, so `question` keeps its place beside the recording it belongs
# to.
TIMING_COLUMNS = ("duration_ms", "speech_start_ms", "speech_end_ms",
                  "leading_silence_ms", "trailing_silence_ms", "speech_total_ms",
                  "n_segments", "max_internal_pause_ms", "trailing_zero_pad_ms",
                  "segments_ms")


def read_mono_16k(path):
    """Samples as float32 in [-1, 1), plus what had to be done to get there.

    Downmixing is not a convenience: the pipeline hands the STT an ffmpeg mono
    16 kHz conversion, so offsets measured on anything else describe a stream
    nothing ever sees.
    """
    import numpy as np

    with wave.open(path) as wf:
        channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{os.path.basename(path)}: {width * 8}-bit, expected 16-bit PCM")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{os.path.basename(path)}: {rate} Hz, expected {SAMPLE_RATE}")

    pcm = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        # Frames, not samples. Getting this wrong is what broke audios/metadata.csv.
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype("int16")
    return pcm, channels


def analyse(path, vad_opts):
    """Where the speech sits in one recording, in milliseconds."""
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps

    pcm, channels = read_mono_16k(path)
    duration_ms = round(len(pcm) / SAMPLE_RATE * 1000)

    # Digital silence at the end is padding somebody appended, not silence the
    # speaker left. Worth separating, since only the second kind says anything
    # about how the recording was made.
    nonzero = np.nonzero(pcm)[0]
    pad_ms = round((len(pcm) - 1 - nonzero[-1]) / SAMPLE_RATE * 1000) if len(nonzero) else duration_ms

    audio = pcm.astype(np.float32) / 32768.0
    segments = [(t["start"], t["end"]) for t in get_speech_timestamps(audio, vad_opts)]

    row = {"duration_ms": duration_ms, "trailing_zero_pad_ms": pad_ms,
           "n_segments": len(segments)}
    if not segments:
        # Left blank rather than zeroed: a recording the VAD heard nothing in has
        # no end of speech, and a zero would read as one at the very start.
        row.update({c: "" for c in TIMING_COLUMNS if c not in row})
        return row, channels, True

    ms = [(round(s / SAMPLE_RATE * 1000), round(e / SAMPLE_RATE * 1000))
          for s, e in segments]
    gaps = [b[0] - a[1] for a, b in zip(ms, ms[1:])]
    row.update({
        "speech_start_ms": ms[0][0],
        "speech_end_ms": ms[-1][1],
        "leading_silence_ms": ms[0][0],
        "trailing_silence_ms": duration_ms - ms[-1][1],
        "speech_total_ms": sum(e - s for s, e in ms),
        "max_internal_pause_ms": max(gaps) if gaps else 0,
        "segments_ms": " ".join(f"{s}-{e}" for s, e in ms),
    })
    return row, channels, False


def read_source(path):
    """The text columns to carry through, keyed by filename."""
    # utf-8-sig: the HeySQuAD exports lead with a BOM, which otherwise becomes
    # part of the first column's name.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} is empty")
    if "filename" not in rows[0]:
        sys.exit(f"{path} has no 'filename' column, so it cannot be matched to audio")
    return {r["filename"]: r for r in rows if r.get("filename")}, list(rows[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", required=True,
                    help="directory of .wav files; metadata.csv is written here "
                         "unless --out says otherwise")
    ap.add_argument("--source-csv",
                    help="CSV whose columns are carried through, joined on "
                         "'filename'. Without it the output holds timings only, "
                         "and the run has no 'question' to log against")
    ap.add_argument("--out", help="output path (default: <audio-dir>/metadata.csv)")
    ap.add_argument("--min-trailing-silence-ms", type=int,
                    default=DEFAULT_MIN_TRAILING_MS,
                    help=f"what the endpointer needs before the file ends "
                         f"(default {DEFAULT_MIN_TRAILING_MS}); recordings under "
                         f"it are named in the summary")
    args = ap.parse_args()

    wavs = sorted(f for f in os.listdir(args.audio_dir) if f.lower().endswith(".wav"))
    if not wavs:
        sys.exit(f"No .wav files in {args.audio_dir}")

    source, carried = ({}, [])
    if args.source_csv:
        source, carried = read_source(args.source_csv)
    carried = [c for c in carried if c not in TIMING_COLUMNS and c != "filename"]

    from faster_whisper.vad import VadOptions
    vad_opts = VadOptions(threshold=VAD_THRESHOLD,
                          min_speech_duration_ms=VAD_MIN_SPEECH_MS,
                          min_silence_duration_ms=0, speech_pad_ms=0)

    out_path = args.out or os.path.join(args.audio_dir, "metadata.csv")
    rows, downmixed, silent, short = [], [], [], []

    for i, name in enumerate(wavs, 1):
        print(f"\r[{i}/{len(wavs)}] {name}", end="", flush=True)
        try:
            timings, channels, no_speech = analyse(os.path.join(args.audio_dir, name),
                                                   vad_opts)
        except ValueError as e:
            sys.exit(f"\n{e}\nConvert the set first, for example:\n"
                     f"  ffmpeg -i in.wav -ac 1 -ar 16000 -sample_fmt s16 out.wav")
        if channels > 1:
            downmixed.append(name)
        if no_speech:
            silent.append(name)
        elif timings["trailing_silence_ms"] < args.min_trailing_silence_ms:
            short.append((name, timings["trailing_silence_ms"]))

        row = {"filename": name}
        row.update({c: source.get(name, {}).get(c, "") for c in carried})
        row.update(timings)
        rows.append(row)
    print()

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, ["filename"] + carried + list(TIMING_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    total_s = sum(r["duration_ms"] for r in rows) / 1000
    print(f"\nWrote {out_path}: {len(rows)} recordings, {total_s / 60:.1f} minutes")
    if carried:
        print(f"  carried from {os.path.basename(args.source_csv)}: {', '.join(carried)}")
        missing = [r["filename"] for r in rows if r["filename"] not in source]
        unused = sorted(set(source) - {r["filename"] for r in rows})
        if missing:
            print(f"  [WARN] {len(missing)} recordings had no row there, so their "
                  f"'question' is blank: {', '.join(missing[:5])}"
                  f"{' ...' if len(missing) > 5 else ''}")
        if unused:
            print(f"  {len(unused)} rows named no file in this directory")
    elif args.source_csv is None:
        print("  [WARN] no --source-csv, so there is no 'question' column: the run "
              "will log what was heard but not what was said")

    if downmixed:
        print(f"  {len(downmixed)} multi-channel recordings were downmixed for the "
              f"VAD, which is what the pipeline feeds the STT")
    if silent:
        print(f"  [WARN] the VAD heard nothing in {len(silent)}: "
              f"{', '.join(silent[:5])}{' ...' if len(silent) > 5 else ''}. Their "
              f"timings are blank and their ttfa will be too")
    if short:
        print(f"  [WARN] {len(short)} recordings have less than "
              f"{args.min_trailing_silence_ms} ms of trailing silence, so the "
              f"endpointer fires on the end of the file rather than on the silence, "
              f"and their TTFA is optimistic:")
        for name, ms in short[:10]:
            print(f"      {name}: {ms} ms")
        if len(short) > 10:
            print(f"      ... and {len(short) - 10} more")
    else:
        print(f"  every recording has at least {args.min_trailing_silence_ms} ms of "
              f"trailing silence, so the endpointer decides when the utterance ends")


if __name__ == "__main__":
    main()
