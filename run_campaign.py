#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run configuration cells x replicates as independent assistant.py launches.

There is no orchestration for a multi-cell sweep today: every cell under
configs/ has been launched by hand, one process, one --config, one --out-dir.
This runs the whole matrix -- cells x replicates -- as one driver invocation
that is safe to leave running for days and safe to kill and restart.

Interleaved, not batched. Running a cell's replicates back to back would keep
the model resident and cost fewer reloads, but it would also mean all of a
cell's replicates share one thermal and cache neighbourhood -- confounding the
between-launch drift replication exists to expose -- and it would make
replicate 1 systematically the cold one for every cell. So each round is a
fresh seeded permutation of *all* cells, and the campaign is round 1 in full,
then round 2 in full, and so on; a given cell's replicates end up tens of
launches apart. The reload this costs is paid during warmup, before the first
timed item.

The run directory a launch actually writes is one level below what this
passes as --out-dir: assistant.py:1591 is
`run_dir = os.path.join(args.out_dir, timestamp)`, so a launch given
`--out-dir <root>/<cell>/r2` writes `<root>/<cell>/r2/<timestamp>/`, and a
second attempt at the same launch makes a *second* timestamped directory
beside the first rather than overwriting it. Resuming therefore means
globbing one level down and judging each candidate, not checking --out-dir
itself for existence. A crashed attempt's half-written directory is left
exactly where it is rather than deleted or renamed -- it costs nothing to
leave (the next attempt gets its own fresh timestamp) and it is the forensic
trail for whatever went wrong.

A launch counts as complete when it has both of: config_used.yaml (written at
assistant.py:1620, after warmup and before the first item -- proves the
process got that far) and a non-empty latency_log_*.csv whose *last* row still
names the cell_id/launch_id this launch was given (proves the CSV was not left
over from a different launch that once used this same directory). Neither
proves every input file was processed: corpus size is a runtime input, not a
constant this script knows, so item-count completeness is never checked --
only that the launch was still this launch when it stopped writing.

Every launch's combined stdout/stderr is kept (`<out-dir>/driver_launch_*.log`)
because that is the only place a warm-up failure appears: the retry warning
prints before config_used.yaml exists (assistant.py:1568) and is never written
to a file of its own. This driver greps its own log for that string instead
of asking assistant.py to log it twice.

Needs nothing beyond the standard library: cell identity comes from a
filename, never from a YAML value this would otherwise have to parse.

Usage:
    # Preview the schedule -- no corpus check, no subprocess, no manifest.
    python3 run_campaign.py --out-dir /tmp/sweep1 --rounds 3 --seed 20260912 --dry-run

    # Run it for real, forwarding one flag to every launch.
    python3 run_campaign.py --out-dir /tmp/sweep1 --rounds 3 --seed 20260912 \\
        -- --llm-num-gpu 0

    # A handful of cells only, against a fixture corpus, skipping the full
    # 120-file corpus's known-bad metadata:
    python3 run_campaign.py --out-dir /tmp/sweep1 --rounds 1 --seed 1 \\
        --cells 17-gemma3_1b_it_q4_K_M-t0_0-vosk_cpu --audio-dir /tmp/fixture \\
        -- --audio /tmp/fixture --llm-num-gpu 0

Re-running the identical command resumes: already-complete launches are
skipped, everything else (never attempted, or left behind by a crash) runs
again. Changing --seed, --rounds or the cell set part-way through a campaign
is refused rather than silently reinterpreted -- point --out-dir at a new,
empty directory instead.
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "campaign_manifest.json"

# assistant.py's own timestamp format (datetime.now().strftime("%Y%m%d_%H%M%S")
# at assistant.py:1360/1591) -- a run directory is named exactly this.
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

# assistant.py:1568. Printed before config_used.yaml exists, so it lives only
# in the log this driver captures -- see the module docstring.
WARMUP_RETRY_NEEDLE = "The LLM is not resident after warmup"

TERMINAL_STATUSES = ("completed", "skipped_already_complete")


@dataclass(frozen=True)
class Cell:
    cell_id: str
    config_path: str


@dataclass(frozen=True)
class LaunchPlan:
    round: int
    launch_id: str
    cell_id: str
    config_path: str


# ---------- Cell list ----------

def load_cells_from_csv(path):
    """Cells from configs/configs.csv: semicolon-separated, `file-name` column.

    `file-name`'s basename without `.yaml` is used as cell_id so it matches
    what assistant.py's own --cell-id defaults to from --config (see
    assistant.py's cell_id resolution just after argparse) -- a launch this
    driver starts and one started by hand from the same config file land on
    the same identity.
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
    except OSError as e:
        sys.exit(f"--configs-csv {path}: {e}")
    if not rows:
        sys.exit(f"--configs-csv {path}: no data rows")
    if "file-name" not in rows[0]:
        sys.exit(f"--configs-csv {path}: no 'file-name' column; got {list(rows[0].keys())}")

    base_dir = os.path.dirname(path)
    cells = []
    for row in rows:
        file_name = row["file-name"].strip()
        if not file_name:
            continue
        cell_id = os.path.splitext(os.path.basename(file_name))[0]
        cells.append(Cell(cell_id=cell_id, config_path=os.path.join(base_dir, file_name)))
    return cells


def load_cells_from_dir(path):
    """Cells from a directory of *.yaml config files, one cell per file."""
    if not os.path.isdir(path):
        sys.exit(f"--configs-dir {path}: no such directory")
    cells = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".yaml"):
            cells.append(Cell(cell_id=name[:-len(".yaml")], config_path=os.path.join(path, name)))
    if not cells:
        sys.exit(f"--configs-dir {path}: no *.yaml files found")
    return cells


def dedupe_or_die(cells, source_desc):
    seen = {}
    for c in cells:
        if c.cell_id in seen and seen[c.cell_id] != c.config_path:
            sys.exit(f"{source_desc}: cell_id {c.cell_id!r} is ambiguous between "
                      f"{seen[c.cell_id]!r} and {c.config_path!r}")
        seen[c.cell_id] = c.config_path
    # Preserve first-seen order while dropping exact duplicates.
    out, added = [], set()
    for c in cells:
        if c.cell_id not in added:
            out.append(c)
            added.add(c.cell_id)
    return out


def filter_cells(cells, wanted):
    if not wanted:
        return cells
    by_id = {c.cell_id: c for c in cells}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        sys.exit(f"--cells names unknown cell(s): {missing}; available: "
                  f"{sorted(by_id)}")
    return [by_id[w] for w in wanted]


# ---------- Scheduling ----------

def build_schedule(cells, rounds, seed):
    """R fresh seeded permutations of `cells`, concatenated round by round.

    Round r is permuted from `random.Random(seed + r)`, so the same seed
    always reproduces the same campaign and different rounds get different
    orders. A permutation cannot repeat a cell against itself, so no two
    launches are ever the same cell *within* a round; the one seam a
    round-by-round concatenation can still produce a repeat across is the
    boundary between round r's last launch and round r+1's first, which is
    fixed by swapping round r+1's first two entries when they'd collide.
    """
    if rounds < 1:
        sys.exit(f"--rounds must be >= 1, got {rounds}")
    if not cells:
        sys.exit("no cells to schedule")

    schedule = []
    prev_last_cell_id = None
    for r in range(1, rounds + 1):
        perm = list(cells)
        random.Random(seed + r).shuffle(perm)
        if prev_last_cell_id is not None and len(perm) > 1 and perm[0].cell_id == prev_last_cell_id:
            perm[0], perm[1] = perm[1], perm[0]
        launch_id = f"r{r}"
        for cell in perm:
            schedule.append(LaunchPlan(round=r, launch_id=launch_id,
                                        cell_id=cell.cell_id, config_path=cell.config_path))
        prev_last_cell_id = perm[-1].cell_id

    _verify_no_consecutive_repeats(schedule)
    return schedule


def _verify_no_consecutive_repeats(schedule):
    """A single cell in the pool makes a repeat unavoidable -- nothing to
    interleave against -- so that case is exempted rather than failing."""
    if len({lp.cell_id for lp in schedule}) <= 1:
        return
    for a, b in zip(schedule, schedule[1:]):
        if a.cell_id == b.cell_id:
            raise RuntimeError(
                f"scheduling bug: {a.cell_id!r} scheduled back to back "
                f"(round {a.round} launch {a.launch_id} -> round {b.round} launch {b.launch_id})")


# ---------- Resume / completion marker ----------

def find_run_dirs(out_dir):
    if not os.path.isdir(out_dir):
        return []
    return sorted(
        os.path.join(out_dir, name)
        for name in os.listdir(out_dir)
        if TIMESTAMP_DIR_RE.match(name) and os.path.isdir(os.path.join(out_dir, name))
    )


def _last_csv_row(csv_path):
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            last = None
            for row in csv.DictReader(f):
                last = row
            return last
    except OSError:
        return None


def completed_run_dir(out_dir, cell_id, launch_id):
    """The run_dir under `out_dir` that is a finished launch of (cell_id,
    launch_id), newest first, or None. See the module docstring for what
    "finished" means and why item count is not part of it."""
    for run_dir in reversed(find_run_dirs(out_dir)):
        if not os.path.isfile(os.path.join(run_dir, "config_used.yaml")):
            continue
        csv_candidates = sorted(glob.glob(os.path.join(run_dir, "latency_log_*.csv")))
        if not csv_candidates:
            continue
        last_row = _last_csv_row(csv_candidates[-1])
        if not last_row:
            continue
        if last_row.get("cell_id", "") == cell_id and last_row.get("launch_id", "") == launch_id:
            return run_dir
    return None


# ---------- Manifest ----------

def manifest_path(campaign_root):
    return os.path.join(campaign_root, MANIFEST_FILENAME)


def load_manifest(campaign_root):
    path = manifest_path(campaign_root)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest_atomic(campaign_root, manifest):
    """Whole-file rewrite + os.replace so a driver killed mid-write leaves the
    previous, complete manifest in place rather than a truncated one."""
    path = manifest_path(campaign_root)
    fd, tmp_path = tempfile.mkstemp(prefix=".campaign_manifest.", suffix=".tmp", dir=campaign_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def plan_signature(cells, rounds, seed):
    return {
        "seed": seed,
        "rounds": rounds,
        "cells": [{"cell_id": c.cell_id, "config_path": c.config_path} for c in cells],
    }


def rounds_summary(schedule, rounds):
    out = []
    for r in range(1, rounds + 1):
        cell_ids = [lp.cell_id for lp in schedule if lp.round == r]
        out.append({"round": r, "launch_id": f"r{r}", "cell_ids": cell_ids})
    return out


def upsert_launch_record(manifest, record):
    """Replace the existing record for (cell_id, launch_id) if this is a
    retry, so the manifest always shows each launch slot's latest attempt --
    the earlier attempt's own log file is untouched on disk either way."""
    key = (record["cell_id"], record["launch_id"])
    for i, existing in enumerate(manifest["launches"]):
        if (existing["cell_id"], existing["launch_id"]) == key:
            manifest["launches"][i] = record
            return
    manifest["launches"].append(record)


# ---------- Running a launch ----------

def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _log_mentions(log_path, needle):
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return any(needle in line for line in f)
    except OSError:
        return False


def _out_dir_ever_warmup_retried(out_dir):
    """Whether any driver log under out_dir ever recorded the warm-up retry.

    Skipping a completed launch must not re-derive its record from scratch:
    an earlier invocation may have seen the retry fire and this one must not
    quietly drop that from the manifest just because nothing ran this time.
    The log files, unlike the manifest, are never overwritten, so they are
    the source of truth here too."""
    return any(_log_mentions(p, WARMUP_RETRY_NEEDLE)
               for p in glob.glob(os.path.join(out_dir, "driver_launch_*.log")))


def build_command(python_bin, assistant_script, config_path, out_dir, cell_id, launch_id, extra_args):
    return [python_bin, assistant_script,
            "--config", config_path,
            "--out-dir", out_dir,
            "--cell-id", cell_id,
            "--launch-id", launch_id] + list(extra_args)


def run_one_launch(args, extra_args, order_index, total, lp, campaign_root):
    out_dir = os.path.join(campaign_root, lp.cell_id, lp.launch_id)

    prior = completed_run_dir(out_dir, lp.cell_id, lp.launch_id)
    if prior is not None:
        print(f"[SKIP] {order_index}/{total} round={lp.round} {lp.cell_id}/{lp.launch_id}: "
              f"already complete at {prior}")
        prior_logs = sorted(glob.glob(os.path.join(out_dir, "driver_launch_*.log")))
        return {
            "order_index": order_index, "round": lp.round, "cell_id": lp.cell_id,
            "launch_id": lp.launch_id, "config_path": lp.config_path, "out_dir": out_dir,
            "run_dir": prior, "command": None, "pid": None,
            "start_ts": None, "end_ts": None, "duration_s": None,
            "exit_code": None, "status": "skipped_already_complete",
            "warmup_retry": _out_dir_ever_warmup_retried(out_dir),
            "log_path": prior_logs[-1] if prior_logs else None,
        }

    stale = find_run_dirs(out_dir)
    if stale:
        print(f"[WARN] {len(stale)} incomplete attempt(s) already under {out_dir}; "
              f"leaving them and starting a fresh one")

    os.makedirs(out_dir, exist_ok=True)
    cmd = build_command(args.python_bin, args.assistant_script, lp.config_path,
                         out_dir, lp.cell_id, lp.launch_id, extra_args)
    attempt_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"driver_launch_{attempt_ts}.log")

    print(f"[RUN]  {order_index}/{total} round={lp.round} {lp.cell_id}/{lp.launch_id}: "
          f"{shlex.join(cmd)}")

    start_ts = _now_iso()
    t0 = time.monotonic()
    pid = None
    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            pid = proc.pid
            exit_code = proc.wait()
    except OSError as e:
        print(f"[FAIL] {lp.cell_id}/{lp.launch_id}: could not start ({e})")
        exit_code = None
    duration_s = time.monotonic() - t0
    end_ts = _now_iso()

    warmup_retry = _log_mentions(log_path, WARMUP_RETRY_NEEDLE)
    run_dir = completed_run_dir(out_dir, lp.cell_id, lp.launch_id)
    if run_dir is not None:
        status = "completed"
    else:
        status = "failed"
        # Still point the record at whatever directory the attempt did leave
        # behind, if any, so a human (or a re-run's [WARN]) has somewhere to
        # look -- see the module docstring on why it is left rather than removed.
        leftovers = find_run_dirs(out_dir)
        run_dir = leftovers[-1] if leftovers else None

    if warmup_retry:
        print(f"[WARN] {lp.cell_id}/{lp.launch_id}: warm-up retry fired; its first item is unusable")
    if status != "completed":
        print(f"[FAIL] {order_index}/{total} {lp.cell_id}/{lp.launch_id}: "
              f"exit={exit_code} log={log_path}")

    return {
        "order_index": order_index, "round": lp.round, "cell_id": lp.cell_id,
        "launch_id": lp.launch_id, "config_path": lp.config_path, "out_dir": out_dir,
        "run_dir": run_dir, "command": cmd, "pid": pid,
        "start_ts": start_ts, "end_ts": end_ts, "duration_s": round(duration_s, 3),
        "exit_code": exit_code, "status": status,
        "warmup_retry": warmup_retry, "log_path": log_path,
    }


# ---------- Metadata preflight ----------

def run_metadata_gate(args):
    if args.skip_metadata_check:
        print(f"[INFO] --skip-metadata-check set; not validating {args.audio_dir!r} before launching")
        return
    validator = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_metadata.py")
    cmd = [args.python_bin, validator, "--audio-dir", args.audio_dir]
    print(f"[INFO] validating corpus: {shlex.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\n{args.audio_dir!r} failed validate_metadata.py (exit {result.returncode}); "
                  f"fix the corpus/metadata, point --audio-dir elsewhere, or pass "
                  f"--skip-metadata-check to run anyway")


# ---------- Dry run ----------

def print_schedule(schedule, args, extra_args, campaign_root):
    for order_index, lp in enumerate(schedule, start=1):
        out_dir = os.path.join(campaign_root, lp.cell_id, lp.launch_id)
        cmd = build_command(args.python_bin, args.assistant_script, lp.config_path,
                             out_dir, lp.cell_id, lp.launch_id, extra_args)
        tag = "skip" if completed_run_dir(out_dir, lp.cell_id, lp.launch_id) else "run "
        print(f"{order_index:4d}  [{tag}] round={lp.round} launch_id={lp.launch_id} "
              f"cell={lp.cell_id}  {shlex.join(cmd)}")


# ---------- Summary ----------

def print_summary(manifest):
    launches = manifest["launches"]
    completed = [l for l in launches if l["status"] == "completed"]
    skipped = [l for l in launches if l["status"] == "skipped_already_complete"]
    failed = [l for l in launches if l["status"] not in TERMINAL_STATUSES]
    warmup = [l for l in launches if l.get("warmup_retry")]

    print("\n" + "=" * 72)
    print(f"Campaign summary: {len(launches)} launch(es) -- "
          f"{len(completed)} completed, {len(skipped)} already complete, "
          f"{len(failed)} failed")

    if warmup:
        print(f"\n[WARN] {len(warmup)} launch(es) hit the warm-up retry -- their first "
              f"item is unusable; budget one discarded item per launch, not per cell:")
        for l in warmup:
            print(f"    {l['cell_id']}/{l['launch_id']}  {l['run_dir'] or l['out_dir']}")

    if failed:
        print(f"\n[FAIL] {len(failed)} launch(es) did NOT complete -- re-running this same "
              f"command will retry only these:")
        for l in failed:
            print(f"    {l['cell_id']}/{l['launch_id']}  exit={l['exit_code']}  log={l['log_path']}")
    else:
        print("\nAll launches completed or were already complete.")
    print("=" * 72)
    return len(failed) == 0


# ---------- CLI ----------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("cell source")
    src.add_argument("--configs-csv", default=None,
                      help="semicolon-separated cell list (default: configs/configs.csv, "
                           "unless --configs-dir is given)")
    src.add_argument("--configs-dir", default=None,
                      help="alternative to --configs-csv: every *.yaml file in this "
                           "directory is a cell")
    p.add_argument("--cells", nargs="+", default=None,
                    help="restrict to these cell_ids (default: every cell in the source)")

    p.add_argument("--out-dir", required=True,
                    help="campaign root; per-launch output goes to "
                         "<out-dir>/<cell_id>/<launch_id>/<timestamp>/, and the manifest "
                         "lives at <out-dir>/" + MANIFEST_FILENAME)
    p.add_argument("--rounds", type=int, default=3,
                    help="replicates per cell (default 3 -- run_statistics.py's "
                         "MIN_PAIRS_FOR_SIGNED_RANK=6 is the point past which "
                         "launch-level testing, not just description, becomes possible)")
    p.add_argument("--seed", type=int, required=True,
                    help="round r is permuted from Random(seed + r); the same seed "
                         "always reproduces the same campaign")

    p.add_argument("--python-bin", default=sys.executable,
                    help="interpreter used for every launch (default: this process's own)")
    p.add_argument("--assistant-script", default="assistant.py",
                    help="script each launch runs (default: assistant.py next to this "
                         "driver); overridable so the driver itself can be tested against "
                         "a cheap stand-in instead of the real pipeline")

    p.add_argument("--audio-dir", default="audios",
                    help="corpus checked by validate_metadata.py before the first launch "
                         "(default: audios). This does not change what any launch reads "
                         "-- each cell's own --audio setting does that -- it only says "
                         "which corpus this preflight check looks at, so point it at "
                         "whatever a --audio override in the extra args after `--` uses")
    p.add_argument("--skip-metadata-check", action="store_true",
                    help="run without validate_metadata.py's gate. For a corpus with a "
                         "known, accepted problem only -- not for making a failure go away")
    p.add_argument("--dry-run", action="store_true",
                    help="print the schedule (with [skip]/[run] against the current "
                         "output directory) and exit; touches no files, runs nothing")
    return p


def split_extra_args(argv):
    """Whatever follows a literal `--` is forwarded verbatim to every launch,
    letting the driver stay agnostic to which run mode is being exercised."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def load_cells(args):
    if args.configs_dir and args.configs_csv:
        sys.exit("--configs-csv and --configs-dir are mutually exclusive")
    if args.configs_dir:
        cells = load_cells_from_dir(args.configs_dir)
        source_desc = args.configs_dir
    else:
        cells = load_cells_from_csv(args.configs_csv or os.path.join("configs", "configs.csv"))
        source_desc = args.configs_csv or os.path.join("configs", "configs.csv")
    cells = dedupe_or_die(cells, source_desc)
    return filter_cells(cells, args.cells)


def main():
    # A campaign runs for hours to days and is normally started under nohup
    # with its stdout redirected to a file for `tail -f`. Redirected-to-a-file
    # stdout is fully (block) buffered by default, so without this an operator
    # watching that file sees nothing until the whole campaign -- or a buffer
    # flush many launches later -- finishes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # stdout without reconfigure() (e.g. already replaced); not fatal.

    argv, extra_args = split_extra_args(sys.argv[1:])
    args = build_arg_parser().parse_args(argv)

    cells = load_cells(args)
    campaign_root = os.path.abspath(args.out_dir)
    schedule = build_schedule(cells, args.rounds, args.seed)

    if args.dry_run:
        print(f"# {len(cells)} cell(s) x {args.rounds} round(s) = {len(schedule)} launch(es), "
              f"seed={args.seed}")
        print_schedule(schedule, args, extra_args, campaign_root)
        return 0

    run_metadata_gate(args)

    os.makedirs(campaign_root, exist_ok=True)
    plan = plan_signature(cells, args.rounds, args.seed)
    existing = load_manifest(campaign_root)
    if existing is not None:
        if existing.get("plan") != plan:
            sys.exit(f"{manifest_path(campaign_root)} already records a different campaign "
                     f"(seed, --rounds or the cell set changed); point --out-dir at a new, "
                     f"empty directory instead of mixing two campaigns in one")
        manifest = existing
        print(f"[INFO] resuming campaign at {campaign_root} "
              f"({len(manifest['launches'])} launch(es) already recorded)")
    else:
        manifest = {
            "version": MANIFEST_VERSION,
            "created_at": _now_iso(),
            "plan": plan,
            "rounds": rounds_summary(schedule, args.rounds),
            "python_bin": args.python_bin,
            "assistant_script": args.assistant_script,
            "extra_args": extra_args,
            "launches": [],
        }
        write_manifest_atomic(campaign_root, manifest)
        print(f"[INFO] starting new campaign at {campaign_root}: "
              f"{len(cells)} cell(s) x {args.rounds} round(s) = {len(schedule)} launch(es)")

    for order_index, lp in enumerate(schedule, start=1):
        record = run_one_launch(args, extra_args, order_index, len(schedule), lp, campaign_root)
        upsert_launch_record(manifest, record)
        write_manifest_atomic(campaign_root, manifest)

    all_ok = print_summary(manifest)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
