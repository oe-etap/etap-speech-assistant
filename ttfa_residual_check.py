#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that ttfa equals stt_endpoint_delay + llm_ttfc + tts_first_chunk.

    residual = measured ttfa - (stt_endpoint_delay + llm_ttfc + tts_first_chunk)

`README.md` ("Relationships Between Metrics") claims the sum holds "to within
a few ms"; this is the reproducible check of that claim, over however many
latency CSVs a campaign has produced.

Reports three strata separately, never pooled into one distribution, because
pooling would hide exactly what this check exists to find:

  - each run's first item pays for whatever the run loads lazily. A handful
    of these land seconds to tens of seconds above the rest: the warm-up
    failure `assistant.py` ~1545 documents lands the model load inside ttfa
    and inside none of the three terms that are supposed to sum to it, so a
    huge residual is the symptom.
  - items whose `e2e_response_ready` row logged more than one response WAV.
    In file mode that is a double endpoint fire; the same field's other,
    unremarkable use is a mic session legitimately answering more than once
    (README.md's extra_json entry for `output_wav_count`), which is why this
    stratum is restricted to file-mode runs rather than read off every item.

Collapses repeated (item, stage) rows to their median before computing
anything, matching `aggregate_logs.py`'s `read_run`: more than one row under
the same item and stage means that item held several timed utterances, and
the median stands in so the item counts once regardless of how many it held.

Standard library only, deliberately, for the same reason `run_statistics.py`
gives for itself: this runs over archived logs on whatever interpreter is at
hand and must not need the pipeline's environment installed. Does not import
`aggregate_logs.py` even though nothing there is currently non-stdlib - that
module has no reason to stay stdlib-only the way `run_statistics.py` is
pinned to by its own docstring, and this script's one job is to keep running
regardless of what that file's import chain grows.
"""

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import run_statistics as rstat

# The four stages a residual needs. An item is usually missing one of these
# because it ran under "fast" pacing, where ttfa is left blank (README.md:
# no offset inside sped-up audio corresponds to a wall-clock instant).
REQUIRED_STAGES = ("stt_endpoint_delay", "llm_ttfc", "tts_first_chunk", "ttfa")
RECONSTRUCTION_STAGES = ("stt_endpoint_delay", "llm_ttfc", "tts_first_chunk")

# Above this the residual is no longer the small, systematic queue-handoff
# offset (measured over the archived corpus at p05 +3ms / p95 +6ms on the
# post-first-item population) - it is a warm-up, a double fire, or a genuine
# break in additivity worth reading by name rather than folding into a
# percentile.
OUTLIER_THRESHOLD_MS = 20.0


@dataclass
class RunRecord:
    """One CSV: its items in row order, and what each one measured."""
    path: Path
    ordered_items: List[str] = field(default_factory=list)
    durations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    wav_count: Dict[str, int] = field(default_factory=dict)
    file_mode: bool = False


def read_run(path: Path) -> Optional[RunRecord]:
    """Parse one latency CSV into per-item stage durations.

    "First item" is whichever item's rows appear first in the file -
    `ordered_items` is built the same way `aggregate_logs.py`'s `read_run`
    builds it, so the two scripts cannot identify a run's warm-up item
    differently.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return None
    if not rows:
        return None

    ordered_items: List[str] = []
    for row in rows:
        item = (row.get("item") or "").strip()
        if item and item not in ordered_items:
            ordered_items.append(item)
    if not ordered_items:
        return None

    run = RunRecord(path=path, ordered_items=ordered_items)

    # input_mode is one of aggregate_logs.py's RUN_CONTEXT_COLUMNS: a property
    # of the whole run, not of one row, and every archived CSV carries exactly
    # one value in it. Requiring the set of non-blank values to be exactly
    # {"file"} is what excludes mic sessions from the double-fire stratum
    # below (see the module docstring) rather than counting their ordinary
    # multi-response behaviour as a fault.
    modes_seen = {(row.get("input_mode") or "").strip() for row in rows} - {""}
    run.file_mode = modes_seen == {"file"}

    # More than one row for the same (stage, item) means the item held
    # several timed utterances; collected here and collapsed to the median
    # below, before any residual is computed, so a repeated item still
    # counts once.
    repeated: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        item = (row.get("item") or "").strip()
        stage = (row.get("stage") or "").strip()
        if not item or not stage:
            continue

        duration_str = (row.get("duration_ms") or "").strip()
        if duration_str:
            try:
                repeated.setdefault((stage, item), []).append(float(duration_str))
            except ValueError:
                pass

        if stage == "e2e_response_ready":
            extra_raw = (row.get("extra_json") or "").strip()
            if not extra_raw:
                continue
            try:
                extra = json.loads(extra_raw)
            except json.JSONDecodeError:
                continue
            count = extra.get("output_wav_count") if isinstance(extra, dict) else None
            if isinstance(count, (int, float)):
                run.wav_count[item] = int(count)

    for (stage, item), values in repeated.items():
        run.durations.setdefault(stage, {})[item] = rstat.percentile(values, 0.5)

    return run


def find_csvs(corpus: Path) -> Tuple[List[Path], List[Path]]:
    """Every distinct latency CSV under a tree, and the copies that were dropped.

    Same glob as `aggregate_logs.py`'s `find_latest_csv_logs`, so a corpus
    that counts as N files there counts the same way here -- except that a run
    archived at two paths is counted once. 28 of the 175 CSVs under `outputs/`
    are byte-identical pairs, `outputs/text-only/<cell>/<ts>/` being a copy of
    `outputs/sub1/run1/<cell>/<ts>/`; counting both inflated the item total by
    a quarter (16594 against 13234) and the outlier counts with it, while the
    medians barely moved. Keyed on content rather than on the (cell, timestamp)
    identity `campaign_report.py` dedupes by, because this script never derives
    that identity and byte-equality is the stronger claim anyway.
    """
    if corpus.is_file():
        return [corpus], []
    kept: List[Path] = []
    dropped: List[Path] = []
    seen: Set[str] = set()
    for path in sorted(corpus.glob("**/latency_log_*.csv")):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest in seen:
            dropped.append(path)
        else:
            seen.add(digest)
            kept.append(path)
    return kept, dropped


@dataclass
class Outlier:
    residual_ms: float
    item: str
    path: Path


@dataclass
class Residuals:
    """Every stratum this check reports, plus what the corpus was made of."""
    after_first: List[float] = field(default_factory=list)
    first_item: List[float] = field(default_factory=list)
    double_fire: List[float] = field(default_factory=list)
    outliers: List[Outlier] = field(default_factory=list)
    csvs_seen: int = 0
    duplicate_csvs: List[Path] = field(default_factory=list)
    csvs_with_all_stages: int = 0
    file_mode_items: int = 0
    double_fire_incidence: int = 0


def collect(corpus: Path, threshold: float) -> Residuals:
    """Walk the corpus once, sorting every qualifying item into its stratum."""
    result = Residuals()
    paths, duplicates = find_csvs(corpus)
    result.csvs_seen = len(paths)
    result.duplicate_csvs = duplicates

    for path in paths:
        run = read_run(path)
        if run is None:
            continue

        if run.file_mode:
            result.file_mode_items += len(run.ordered_items)

        first_item = run.ordered_items[0]
        contributed = False

        for item in run.ordered_items:
            wav_count = run.wav_count.get(item, 0)
            is_double_fire = run.file_mode and wav_count > 1
            if is_double_fire:
                result.double_fire_incidence += 1

            if not all(item in run.durations.get(stage, {}) for stage in REQUIRED_STAGES):
                continue

            reconstructed = sum(run.durations[stage][item] for stage in RECONSTRUCTION_STAGES)
            residual = run.durations["ttfa"][item] - reconstructed
            contributed = True

            bucket = result.first_item if item == first_item else result.after_first
            bucket.append(residual)
            if abs(residual) > threshold:
                result.outliers.append(Outlier(residual, item, path))
            if is_double_fire:
                result.double_fire.append(residual)

        if contributed:
            result.csvs_with_all_stages += 1

    return result


# ---------- Reporting ----------
def stratum_line(label: str, values: Sequence[float], threshold: float) -> str:
    if not values:
        return f"{label:<26} n=0 (no qualifying items)"
    over = sum(1 for v in values if abs(v) > threshold)
    return (f"{label:<26} n={len(values):<7} median={rstat.percentile(values, 0.5):+.0f} ms"
            f"  p95={rstat.percentile(values, 0.95):+.0f} ms  max={max(values):.0f} ms"
            f"  |resid|>{threshold:.0f}ms: {over}")


def format_report(result: Residuals, threshold: float) -> str:
    lines = [
        f"CSVs under corpus: {result.csvs_seen} distinct "
        f"({result.csvs_with_all_stages} hold an item with all four stages)"
        + (f", {len(result.duplicate_csvs)} byte-identical copies dropped"
           if result.duplicate_csvs else ""),
        "",
        stratum_line("items after the first", result.after_first, threshold),
        stratum_line("first item of each run", result.first_item, threshold),
    ]

    all_values = result.after_first + result.first_item
    if all_values:
        n = len(all_values)
        le5 = sum(1 for v in all_values if abs(v) <= 5) / n
        le20 = sum(1 for v in all_values if abs(v) <= 20) / n
        lines.append(f"|residual| <= 5 ms: {le5:.1%} of all {n} items;  <= 20 ms: {le20:.1%}")

    lines.append("")
    if result.file_mode_items:
        lines.append(f"items with output_wav_count > 1 (e2e_response_ready's extra_json): "
                     f"{result.double_fire_incidence} of {result.file_mode_items} file-mode items")
        if result.double_fire:
            lines.append(stratum_line("  - residual where computable", result.double_fire, threshold))
    else:
        lines.append("no file-mode items in this corpus; double-fire stratum is empty")

    if result.outliers:
        lines.append("")
        lines.append(f"Outliers (|residual| > {threshold:.0f} ms), largest first:")
        for outlier in sorted(result.outliers, key=lambda o: -abs(o.residual_ms)):
            lines.append(f"  {outlier.residual_ms:+9.0f} ms  {outlier.item:<30} {outlier.path}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("corpus", nargs="?", default="outputs",
                        help="Directory to search recursively for latency_log_*.csv files, "
                             "or a single CSV file (default: outputs).")
    parser.add_argument("--outlier-threshold-ms", type=float, default=OUTLIER_THRESHOLD_MS,
                        help="Residual magnitude above which an item is listed by name "
                             f"(default: {OUTLIER_THRESHOLD_MS:.0f}).")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"error: '{corpus}' does not exist", file=sys.stderr)
        return 1

    result = collect(corpus, args.outlier_threshold_ms)
    if result.csvs_seen == 0:
        print(f"no latency_log_*.csv files found under '{corpus}'", file=sys.stderr)
        return 1

    print(format_report(result, args.outlier_threshold_ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
