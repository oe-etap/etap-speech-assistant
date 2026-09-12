#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the sub-experiment that turns the reconstruction's accuracy into a number.

`ttfa_reconstruct.py` assumes `ttfa` can be rebuilt from an ASR pass and a
text-mode launch. That assumption has one real gap: the recognizer is still
decoding roughly a second of trailing silence while the LLM is already
producing its first tokens, so in a full run those two compete for the CPU
and in the split arms they do not. Nothing in the archive can size that --
it only ever contained full runs -- so it has to be measured by running the
same items both ways.

Five phases, each skipped if its output is already there, so a killed run
resumes by being re-issued:

    subset    stratified pick of the corpus, written as its own folder
    asr       one --asr-only launch over the subset: the canonical pass
    text      --input-mode text, cells x replicates, off those transcripts
    measured  the full pipeline, same cells x replicates, same items
    compare   ttfa_reconstruct.py over all three -> the comparison tables

The subset is stratified on how endpoint-hard each item is, because that is
the axis the gap lives on: an item the recognizer settles instantly leaves
nothing to overlap, and an item it worries over leaves the most. The default
key is `max_internal_pause_ms` from the corpus metadata, which
`validate_metadata.py` already reports under exactly that description; pass
`--key-from-asr` instead once a real ASR pass exists, since a measured
`stt_endpoint_delay` beats a proxy for it.

Cells are named, never discovered: "2-3 LLM configurations spanning the size
range" is a judgement about the model lineup, not something a script should
guess. Pick them the way the sweep does, smallest to largest.

Standard library only -- it builds command lines and reads CSVs; the
pipeline's environment is needed by the launches it starts, not by this.

Usage:
    # See the whole plan without running anything.
    python3 ttfa_validation.py --out-dir /tmp/val --audio-dir audios \\
        --cells 17-gemma3_1b_it_q4_K_M-t0_0-vosk_cpu \\
                19-gemma3_4b_it_q4_K_M-t0_0-vosk_cpu \\
                21-gemma3_27b_it_q4_K_M-t0_0-vosk_cpu \\
        --items 12 --replicates 3 --seed 20260912 --dry-run

    # Run it. Everything after `--` is forwarded to every launch.
    python3 ttfa_validation.py --out-dir /tmp/val --audio-dir audios \\
        --cells ... --items 12 --replicates 3 --seed 20260912 \\
        -- --llm-num-gpu 0
"""

import argparse
import csv
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_statistics as rstat            # noqa: E402
import ttfa_reconstruct as recon          # noqa: E402

# `assistant.py`'s own run-directory name (`%Y%m%d_%H%M%S`); a launch writes
# one of these *below* whatever --out-dir it was given, so every path this
# script hands on has to glob one level down rather than trust --out-dir.
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

PHASES = ("subset", "asr", "text", "measured", "compare")

# `validate_metadata.py` reports this column as "how endpoint-hard the corpus
# is": a long pause inside an utterance is what makes a decoder-confidence
# endpointer hesitate. Only a proxy -- the real quantity is the measured
# `stt_endpoint_delay`, which --key-from-asr uses when a pass exists.
DEFAULT_KEY_COLUMN = "max_internal_pause_ms"
ASR_KEY_STAGE = "stt_endpoint_delay"


def die(message):
    sys.exit(f"error: {message}")


def run(cmd, dry_run, cwd=None):
    print(f"[RUN] {shlex.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=cwd).returncode


def find_run_dir(out_dir):
    """The timestamped directory a launch wrote under `out_dir`, newest last."""
    if not os.path.isdir(out_dir):
        return None
    candidates = sorted(name for name in os.listdir(out_dir)
                        if TIMESTAMP_DIR_RE.match(name)
                        and os.path.isdir(os.path.join(out_dir, name)))
    return os.path.join(out_dir, candidates[-1]) if candidates else None


# ---------- Phase 1: the stratified subset ----------
def read_metadata(audio_dir):
    path = os.path.join(audio_dir, "metadata.csv")
    if not os.path.isfile(path):
        die(f"{path}: no such file")
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        die(f"{path}: no data rows")
    return rows


def keys_from_metadata(rows, column):
    if column not in rows[0]:
        die(f"metadata.csv has no {column!r} column; got {sorted(rows[0])}")
    keys = {}
    for row in rows:
        filename = (row.get("filename") or "").strip()
        raw = (row.get(column) or "").strip()
        if not filename or not raw:
            continue
        try:
            keys[filename] = float(raw)
        except ValueError:
            continue
    return keys


def keys_from_asr(tree):
    """Measured `stt_endpoint_delay` per item, over whatever runs are there.

    An item is keyed by its median across the passes, so a single unlucky
    recognition does not decide which stratum a recording belongs in.
    """
    arm = recon.load_arm("asr", recon.Path(tree))
    if not arm.runs:
        die(f"{tree}: no latency CSV to take endpoint delays from")
    gathered = {}
    for run_record in arm.runs:
        for item, value in recon.item_series(run_record, ASR_KEY_STAGE).items():
            gathered.setdefault(item, []).append(value)
    if not gathered:
        die(f"{tree}: no {ASR_KEY_STAGE} rows")
    return {item: rstat.percentile(values, 0.5) for item, values in gathered.items()}


def stratified_pick(keys, wanted, strata, seed):
    """`wanted` items spread evenly over `strata` bands of the hardness key.

    Equal-count bands rather than equal-width ones: the key is heavy-tailed,
    so equal-width bands would put almost everything in the first one and
    leave the hard band a lottery. The remainder goes to the harder bands,
    and the single hardest recording is forced in if the draw missed it --
    with a small corpus the top band is thin enough that chance can drop the
    one case the experiment exists to cover.
    """
    if not keys:
        die("no items carry the hardness key")
    ordered = sorted(keys, key=lambda name: (keys[name], name))
    wanted = min(wanted, len(ordered))
    strata = max(1, min(strata, wanted))
    rng = random.Random(seed)

    bands = []
    for index in range(strata):
        lo = (index * len(ordered)) // strata
        hi = ((index + 1) * len(ordered)) // strata
        bands.append(ordered[lo:hi])

    quotas = [wanted // strata] * strata
    for index in range(wanted % strata):
        quotas[strata - 1 - index] += 1

    chosen = []
    for band, quota in zip(bands, quotas):
        pool = list(band)
        rng.shuffle(pool)
        chosen.extend(pool[:quota])

    hardest = ordered[-1]
    if hardest not in chosen and chosen:
        chosen.sort(key=lambda name: (keys[name], name))
        chosen[0] = hardest

    for name in reversed(ordered):                 # a band thinner than its quota
        if len(chosen) >= wanted:
            break
        if name not in chosen:
            chosen.append(name)

    return sorted(chosen, key=lambda name: (keys[name], name)), bands


def build_subset(audio_dir, rows, chosen, dest):
    """The picked recordings as a corpus of their own, metadata and all.

    A folder rather than a file list because `--audio` takes a directory and
    every downstream lookup (`speech_end_ms`, the evaluation join) goes
    through `metadata.csv`; carrying only the chosen rows keeps the subset
    passing `validate_metadata.py` on its own.
    """
    os.makedirs(dest, exist_ok=True)
    by_name = {(row.get("filename") or "").strip(): row for row in rows}
    kept = []
    for name in chosen:
        source = os.path.join(audio_dir, name)
        if not os.path.isfile(source):
            die(f"{source}: chosen for the subset but not on disk")
        shutil.copy2(source, os.path.join(dest, name))
        kept.append(by_name[name])
    with open(os.path.join(dest, "metadata.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    return kept


# ---------- Phases 2-4: the three arms ----------
def asr_pass(args, extra_args, subset_dir, asr_root):
    """The canonical pass. One launch, never replicated.

    Replicating it would be replicating the wrong thing: the text arm runs on
    one frozen transcript set, so a second pass produces delays belonging to
    a different recognition. Repeat passes go in a separate tree and reach
    the report as `ttfa_reconstruct.py`'s spread figure.
    """
    config = args.asr_config or os.path.join(args.configs_dir, f"{args.cells[0]}.yaml")
    if not os.path.isfile(config):
        die(f"{config}: no such config for the ASR pass")
    cmd = [args.python_bin, "assistant.py", "--config", config, "--asr-only",
           "--audio", subset_dir, "--out-dir", asr_root,
           "--cell-id", "asr-pass", "--launch-id", "canonical"] + list(extra_args)
    return run(cmd, args.dry_run)


def campaign(args, extra_args, root, audio_dir, forwarded, skip_metadata):
    cmd = [args.python_bin, "run_campaign.py",
           "--configs-dir", args.configs_dir,
           "--cells"] + list(args.cells) + [
           "--out-dir", root,
           "--rounds", str(args.replicates),
           "--seed", str(args.seed),
           "--audio-dir", audio_dir]
    if skip_metadata:
        cmd.append("--skip-metadata-check")
    if args.dry_run:
        cmd.append("--dry-run")
    cmd += ["--"] + list(forwarded) + list(extra_args)
    return run(cmd, dry_run=False)          # run_campaign has its own --dry-run


# ---------- Phase 5: the tables ----------
def compare(args, asr_root, text_root, measured_root):
    cmd = [args.python_bin, "ttfa_reconstruct.py",
           "--asr-arm", asr_root,
           "--text-arm", text_root,
           "--measured", measured_root,
           "--per-item-csv", os.path.join(args.out_dir, "reconstructed_per_item.csv"),
           "--comparison-csv", os.path.join(args.out_dir, "measured_vs_reconstructed.csv"),
           "--json", os.path.join(args.out_dir, "reconstruction.json")]
    if args.offset_ms is not None:
        cmd += ["--offset-ms", str(args.offset_ms)]
    return run(cmd, args.dry_run)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra_args = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra_args = argv[:cut], argv[cut + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", required=True,
                        help="Where the subset and all three arms are written.")
    parser.add_argument("--audio-dir", default="audios",
                        help="Corpus to draw the subset from (default: audios).")
    parser.add_argument("--cells", nargs="+", required=True,
                        help="Config stems under --configs-dir, 2-3 of them, chosen to "
                             "span the model size range.")
    parser.add_argument("--configs-dir", default="configs",
                        help="Where those cells' YAML files live (default: configs).")
    parser.add_argument("--asr-config", default=None,
                        help="Config for the canonical ASR pass (default: the first "
                             "cell's). Only its recognizer settings matter -- --asr-only "
                             "builds no LLM and no synthesizer -- so this is worth setting "
                             "only when the cells disagree on the recognizer.")
    parser.add_argument("--items", type=int, default=12,
                        help="How many recordings the subset holds (default: 12).")
    parser.add_argument("--strata", type=int, default=3,
                        help="Equal-count bands of the hardness key to spread them over "
                             "(default: 3).")
    parser.add_argument("--key-column", default=DEFAULT_KEY_COLUMN,
                        help=f"metadata.csv column to stratify on (default: "
                             f"{DEFAULT_KEY_COLUMN}).")
    parser.add_argument("--key-from-asr", default=None,
                        help="Stratify on measured stt_endpoint_delay from this ASR tree "
                             "instead of on a metadata column.")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Launches per cell in each of the two LLM arms (default: 3).")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seeds both the subset draw and the campaign's launch order.")
    parser.add_argument("--offset-ms", type=float, default=None,
                        help="Passed through to ttfa_reconstruct.py's queue-handoff offset.")
    parser.add_argument("--python-bin", default=sys.executable,
                        help="Interpreter the launches run under (default: this one).")
    parser.add_argument("--stop-after", choices=PHASES, default=None,
                        help="Run up to and including this phase, then stop.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command and the subset it would draw, run nothing.")
    args = parser.parse_args(argv)

    if args.items < 1 or args.replicates < 1:
        die("--items and --replicates must both be at least 1")

    subset_dir = os.path.join(args.out_dir, "subset")
    asr_root = os.path.join(args.out_dir, "asr")
    text_root = os.path.join(args.out_dir, "text")
    measured_root = os.path.join(args.out_dir, "measured")
    os.makedirs(args.out_dir, exist_ok=True)

    def stop_here(phase):
        return args.stop_after is not None and PHASES.index(phase) >= PHASES.index(args.stop_after)

    # --- subset
    rows = read_metadata(args.audio_dir)
    if args.key_from_asr:
        keys = keys_from_asr(args.key_from_asr)
        key_name = f"{ASR_KEY_STAGE} (measured, {args.key_from_asr})"
        # The ASR log keys items by stem; the corpus keys them by filename.
        by_stem = {os.path.splitext(row["filename"])[0]: row["filename"] for row in rows}
        keys = {by_stem[stem]: value for stem, value in keys.items() if stem in by_stem}
    else:
        keys = keys_from_metadata(rows, args.key_column)
        key_name = f"{args.key_column} (metadata proxy)"

    chosen, bands = stratified_pick(keys, args.items, args.strata, args.seed)
    print(f"[INFO] stratified on {key_name} over {len(keys)} candidate recordings")
    for index, band in enumerate(bands, start=1):
        taken = [name for name in chosen if name in band]
        print(f"  band {index}: key {keys[band[0]]:.0f}..{keys[band[-1]]:.0f} ms, "
              f"{len(band)} candidates, {len(taken)} taken: {', '.join(taken) or '-'}")
    print(f"[INFO] subset ({len(chosen)}): {', '.join(chosen)}")

    if not args.dry_run:
        build_subset(args.audio_dir, rows, chosen, subset_dir)
        with open(os.path.join(args.out_dir, "subset.json"), "w", encoding="utf-8") as handle:
            json.dump({"key": key_name, "seed": args.seed, "strata": args.strata,
                       "items": {name: keys[name] for name in chosen}},
                      handle, indent=2, sort_keys=True)
        print(f"[INFO] wrote {subset_dir} and {os.path.join(args.out_dir, 'subset.json')}")
    if stop_here("subset"):
        return 0

    # --- asr
    if find_run_dir(asr_root) and not args.dry_run:
        print(f"[SKIP] canonical ASR pass already under {asr_root}")
    elif asr_pass(args, extra_args, subset_dir, asr_root) != 0:
        die("the ASR pass failed; its log is under the run directory")
    if stop_here("asr"):
        return 0

    run_dir = find_run_dir(asr_root)
    transcripts = os.path.join(run_dir, "transcripts.jsonl") if run_dir else \
        os.path.join(asr_root, "<timestamp>", "transcripts.jsonl")
    if not args.dry_run and not os.path.isfile(transcripts):
        die(f"{transcripts}: the ASR pass wrote no transcripts")

    # --- text arm
    if campaign(args, extra_args, text_root, subset_dir,
                ["--input-mode", "text", "--transcripts", transcripts],
                skip_metadata=True) != 0:
        die("the text arm did not finish")
    if stop_here("text"):
        return 0

    # --- measured arm
    if campaign(args, extra_args, measured_root, subset_dir,
                ["--audio", subset_dir], skip_metadata=False) != 0:
        die("the measured arm did not finish")
    if stop_here("measured"):
        return 0

    # --- compare
    return compare(args, asr_root, text_root, measured_root)


if __name__ == "__main__":
    sys.exit(main())
