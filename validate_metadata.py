#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refuse a recording set before its metadata.csv wastes a campaign's time.

`assistant.py:245` `reference_speech_end_s()` reads `speech_end_ms` from the
corpus `metadata.csv` and trusts it unconditionally -- it is the anchor
`stt_endpoint_delay` and `ttfa` are measured from. A bad value does not crash
anything; it produces plausible-looking numbers that are wrong, and the
problem surfaces only when results are analysed, after the runs that would
have caught it in seconds have already been paid for.

This has happened before, and `make_metadata.py`'s docstring records it: a
metadata generator counted interleaved samples as frames on stereo files,
putting `speech_end_ms` (and every other timing column) at twice its real
value. This module's own checks, run against the `metadata.csv` shipped in
`audios/`, still find that bug live today: all 24 stereo recordings there
disagree with their own file's duration by a factor of 2.00, and none of the
96 mono ones disagree at all. That corpus has not been regenerated since the
bug was fixed in `make_metadata.py` -- which is exactly the situation this
script exists to catch before a campaign runs on it.

Do NOT gate on "16 kHz mono". The pipeline (`ensure_wav_mono_16k`,
`assistant.py:271`) and `make_metadata.py` (`read_mono_16k`) both downmix
multi-channel audio before measuring it, so a stereo file with *correct*
metadata runs fine -- channel count is reported here, never a failure
reason. What actually breaks is the frame count a wrong generator measured
against, which is exactly what the duration cross-check below catches,
independently of channel count.

Standard-library only, by the same rule `run_statistics.py` states for
analysis tooling: this has to run on whatever interpreter is at hand, not
the pipeline's environment. `make_metadata.py` is imported for
DEFAULT_MIN_TRAILING_MS rather than re-deriving or copying it a third time;
that module only imports numpy/faster_whisper lazily inside its functions,
so importing it here for one constant costs nothing.

Exit code contract (a driver gates on this):
    0  every row passed every check.
    1  at least one row failed, or the CSV/audio directory could not be read
       well enough to check (missing required column, unreadable CSV, no
       such directory). The offending rows are named on stdout first.

Usage:
    python3 validate_metadata.py --audio-dir audios
    python3 validate_metadata.py --audio-dir audios/data-v3/heysquad_filtered_verified_audio_16khz_mono \\
        --min-trailing-silence-ms 1065
"""

import argparse
import csv
import os
import sys
import wave

from make_metadata import DEFAULT_MIN_TRAILING_MS

# Rounding slop between two independent duration computations -- this
# script's own wave-module frame count vs. whatever produced the CSV's
# duration_ms -- and not the multi-second class of error the stereo bug
# produces. Measured on audios/metadata.csv: 2 of 120 rows are off by
# exactly 1 ms (harmless), 24 are off by a factor of 2.00 (the bug). One ms
# of slack absorbs the first without coming close to masking the second.
DURATION_TOLERANCE_MS = 1

# Without these, none of the checks below can even run.
REQUIRED_COLUMNS = ("filename", "question", "speech_end_ms", "duration_ms",
                    "trailing_silence_ms")


def parse_ms(value):
    """A CSV field as a float, or None if it is missing, blank, or not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def wav_properties(path):
    """(channels, rate, duration_ms) read straight off the file's own header.

    duration_ms is nframes / rate * 1000, and wave.getnframes() already
    counts frames -- one tick per channel, not per sample -- so this number
    is correct on a multi-channel file without any downmixing step. That is
    exactly what the buggy generator got wrong: it counted interleaved
    samples as frames, wrong by exactly the channel count, which is why it
    disagrees with this figure by ~2x on a stereo file and not at all on a
    mono one.
    """
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        nframes = wf.getnframes()
    duration_ms = round(nframes / rate * 1000) if rate else 0
    return channels, rate, duration_ms


def check_row(row, audio_dir, min_trailing_ms):
    """Problems with one metadata row, as a list of human-readable strings.

    Returns (problems, channels_or_None, rate_or_None, pause_ms_or_None) so
    the caller can fold channel/rate/pause reporting in without re-reading
    the row or the file.
    """
    name = (row.get("filename") or "").strip()
    if not name:
        return ["a row has no filename"], None, None, None

    problems = []

    if not (row.get("question") or "").strip():
        problems.append(f"{name}: 'question' is empty -- no reference text "
                        f"for WER or intent coverage")

    speech_end = parse_ms(row.get("speech_end_ms"))
    duration = parse_ms(row.get("duration_ms"))
    if speech_end is None:
        problems.append(f"{name}: speech_end_ms is missing or not a number "
                        f"({row.get('speech_end_ms')!r}) -- it is the anchor "
                        f"for every timing metric")
    if duration is None:
        problems.append(f"{name}: duration_ms is missing or not a number "
                        f"({row.get('duration_ms')!r})")
    if speech_end is not None and duration is not None:
        if not (0 < speech_end < duration):
            problems.append(f"{name}: speech_end_ms={speech_end:g} is not "
                            f"between 0 and duration_ms={duration:g} -- the "
                            f"stereo frame-count bug produces exactly this")

    trailing = parse_ms(row.get("trailing_silence_ms"))
    if trailing is None:
        problems.append(f"{name}: trailing_silence_ms is missing or not a "
                        f"number ({row.get('trailing_silence_ms')!r})")
    elif trailing < min_trailing_ms:
        problems.append(f"{name}: only {trailing:g} ms of trailing silence, "
                        f"need {min_trailing_ms} -- the endpointer would fire "
                        f"on end-of-file rather than on silence, so ttfa "
                        f"would measure a path live input never takes")

    channels = rate = None
    audio_path = os.path.join(audio_dir, name)
    try:
        channels, rate, file_duration = wav_properties(audio_path)
    except FileNotFoundError:
        problems.append(f"{name}: no such file in {audio_dir}")
    except (wave.Error, EOFError, OSError) as e:
        problems.append(f"{name}: not readable as PCM WAV ({e})")
    else:
        if duration is not None and abs(duration - file_duration) > DURATION_TOLERANCE_MS:
            ratio = f", {duration / file_duration:.2f}x" if file_duration else ""
            problems.append(
                f"{name}: metadata duration_ms={duration:g} but the file "
                f"itself is {file_duration} ms (channels={channels}, "
                f"rate={rate}{ratio}) -- timings were computed against the "
                f"wrong frame count")

    pause = parse_ms(row.get("max_internal_pause_ms"))
    return problems, channels, rate, pause


def percentile(sorted_values, pct):
    """Nearest-rank percentile of an already-sorted sequence.

    Hand-rolled rather than `statistics.quantiles` because that raises below
    two data points, and a sample folder of one file is exactly the case
    `00-INDEX.md` says every check here must tolerate.
    """
    if not sorted_values:
        return None
    k = round(pct / 100 * (len(sorted_values) - 1))
    return sorted_values[max(0, min(len(sorted_values) - 1, k))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", required=True,
                    help="directory of .wav files; metadata.csv is read from "
                         "here unless --metadata says otherwise")
    ap.add_argument("--metadata",
                    help="path to metadata.csv (default: <audio-dir>/metadata.csv)")
    ap.add_argument("--min-trailing-silence-ms", type=int,
                    default=DEFAULT_MIN_TRAILING_MS,
                    help=f"what the endpointer needs before the file ends "
                         f"(default {DEFAULT_MIN_TRAILING_MS}, "
                         f"make_metadata.DEFAULT_MIN_TRAILING_MS); pass the "
                         f"figure for the endpoint schedule actually used if "
                         f"it differs")
    args = ap.parse_args()

    if not os.path.isdir(args.audio_dir):
        sys.exit(f"{args.audio_dir}: no such directory")
    metadata_path = args.metadata or os.path.join(args.audio_dir, "metadata.csv")
    try:
        with open(metadata_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as e:
        sys.exit(f"{metadata_path}: {e}")

    if not rows:
        sys.exit(f"{metadata_path} is empty")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        sys.exit(f"{metadata_path} has no {', '.join(missing)} column(s); "
                 f"cannot validate")

    all_problems = []
    channel_counts, rate_counts, pauses = {}, {}, []
    for row in rows:
        problems, channels, rate, pause = check_row(row, args.audio_dir,
                                                     args.min_trailing_silence_ms)
        all_problems.extend(problems)
        if channels is not None:
            channel_counts[channels] = channel_counts.get(channels, 0) + 1
            rate_counts[rate] = rate_counts.get(rate, 0) + 1
        if pause is not None:
            pauses.append(pause)

    print(f"{metadata_path}: {len(rows)} rows checked against {args.audio_dir}")

    if channel_counts:
        chans = ", ".join(f"{n} at {c}ch" for c, n in sorted(channel_counts.items()))
        rates = ", ".join(f"{n} at {r} Hz" for r, n in sorted(rate_counts.items()))
        print(f"  channels: {chans}")
        print(f"  sample rates: {rates}")

    if pauses:
        pauses.sort()
        print(f"  max_internal_pause_ms over {len(pauses)} recordings: "
             f"median={percentile(pauses, 50):.0f} p90={percentile(pauses, 90):.0f} "
             f"p95={percentile(pauses, 95):.0f} max={pauses[-1]:.0f} "
             f"(reported, not gated -- how endpoint-hard the corpus is)")

    if all_problems:
        print(f"\n[FAIL] {len(all_problems)} problem(s) in {metadata_path}:")
        for p in all_problems:
            print(f"  {p}")
        sys.exit(1)

    print(f"\n[OK] every row passed")


if __name__ == "__main__":
    main()
