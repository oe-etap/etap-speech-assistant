#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize a campaign of launches, one cell and one launch at a time.

`aggregate_logs.py` pools every CSV handed to it into one sample per stage
(the `repeated` dict, ~lines 207-266, keyed by `(stage, item)`), which is
right for one launch but wrong for several: pooling R launches of the same
cell back into one sample is exactly the replication the campaign paid for,
thrown away before anyone can see it. `--compare`/`--compare-count` have the
same property on the two-sample side. Neither flag makes the launch visible
as a grouping level, and nothing downstream can recover between-launch
spread from a number that already averaged over it.

This module adds that level on top, without touching how one launch is read.
Per `(cell, launch, stage)` it takes the median over items - the same
operation `aggregate_logs.py` performs on one CSV - then per `(cell, stage)`
the median of the R launch medians, reported with its range rather than a
confidence interval: `run_statistics.MIN_N_FOR_BOOTSTRAP = 20` exists because
a nominal 95% interval over a handful of values delivers far less (measured
at 77% for n=4), and a campaign's replicate count lives well below that. The
same reasoning applies to `run_statistics.MIN_PAIRS_FOR_SIGNED_RANK = 6`: a
per-launch contrast is reported as R independent shift estimates, not
smoothed into one. Nothing in `run_statistics.py` is re-derived or replaced;
`percentile()` and `compare_paired()` are called exactly as `aggregate_logs.py`
and `ttfa_residual_check.py` call them, so a figure computed here and one
computed there over the same launch cannot silently drift apart by having
picked different order-statistic conventions - a truncated-index percentile
(`sorted[int(n*q)]`) and a linearly interpolated one give visibly different
answers on a short, heavy-tailed sample.

Every launch's first item is dropped before anything else: 00-INDEX.md's
"Measured facts" section documents it as bimodal, not merely noisy - 162 of
172 archived first items cost single-digit milliseconds and 10 cost 3.5-50 s
of model load, nothing in between - so no percentile over it means anything,
and R launches means R such items rather than one. The exclusion is the same
computation `aggregate_logs.py`'s `read_run` does for `run.skipped_warmup`
(the first item in CSV row order), reimplemented here rather than imported so
that this file's dependency on that one is nil - matching `ttfa_residual_
check.py`'s stated reason for the same choice: this walks archived logs on
whatever interpreter is at hand, and must not need `aggregate_logs.py`'s
import chain to stay stdlib-only along with it.

Identity for a launch's rows comes from the `cell_id`/`launch_id` columns
`assistant.py` now writes (task 02) when they carry a value; a blank or
absent pair - every run that predates them - falls back to the path.
`assistant.py:1570` names a run's directory `<out-dir>/<timestamp>/` via
`strftime("%Y%m%d_%H%M%S")`, so the timestamped directory is already a
unique per-launch name and its parent is the cell, exactly as 00-INDEX.md's
"shape of a run directory" states. `evaluation_items.csv` carries no such
columns at all (its schema is `evaluation/aggregation.py:ItemResult.
flat_row`, which has neither), so its identity is always read off the path;
by default it sits one directory deeper than the latency CSV, at
`<run_dir>/evaluation/evaluation_items.csv` (`evaluation/pipeline.py:
default_out_dir`), which is why identity is resolved by searching upward for
a timestamp-shaped directory name rather than by a fixed number of `.parent`
hops - one search handles both depths.

Standard library only, for the reason `run_statistics.py` gives for itself:
this is meant to run over a campaign's logs on whatever interpreter comes to
hand, without needing the pipeline's own environment installed.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import run_statistics as rstat

# assistant.py:1360 - datetime.now().strftime("%Y%m%d_%H%M%S"). Used to find
# the directory a launch wrote, walking up from whatever CSV was handed in,
# rather than assuming a fixed depth (see the module docstring).
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")

# 00-INDEX.md "Measured facts": the first item of every launch is bimodal
# (single-digit ms or 3.5-50 s of model load, nothing between), never a
# representative measurement. One per launch, always - this does not read
# log_averages.json's `warmup_dropped_per_run` (which defaults to 0 unless a
# human re-ran aggregate_logs.py with --warmup) because that would make the
# exclusion conditional on a step nobody is required to take; the plan calls
# for the same exclusion unconditionally, the way ttfa_residual_check.py does.
WARMUP_ITEMS = 1

# aggregate_logs.py:47 RUN_CONTEXT_COLUMNS, copied rather than imported for
# the reason given in the module docstring. Used only to warn when a cell's
# own launches disagree on one of them, which is the same "not the same
# experiment" check aggregate_logs.py runs across the logs it pools - applied
# one grouping level up, across a cell's launches instead of across CSVs.
RUN_CONTEXT_COLUMNS = ["input_mode", "audio_pacing", "utterance_trigger",
                       "stt_engine", "tts_engine", "mode"]

# aggregate_logs.py:40 KNOWN_STAGES, copied for display order only: a stage
# absent from this list still gets processed and reported, just sorted after
# the named ones, so an unrecognized name cannot hide a row the way a missing
# one silently would.
KNOWN_STAGES = ["stt", "stt_endpoint_delay", "llm_prompt_eval", "llm_ttft",
                "llm_first_chunk_fill", "llm_ttfc", "tts_first_chunk", "ttfa",
                "llm_eval", "tts_total", "e2e_response_ready"]

# evaluation/README.md "Tier 0 - decidable prompt adherence (always)" and
# "Tier 1 - relevance"; names match evaluation/constraints.py and
# evaluation/relevance.py exactly. Tier 2 (judge/factuality) and acceptance
# are deliberately not here: the plan asks for "the item-level format-pass
# rate and the other Tier 0/1 aggregates", nothing past that tier.
TIER0_METRICS = ["constraint_item_pass_strict", "constraint_item_pass_loose",
                 "constraint_check_rate_strict", "constraint_check_rate_loose"]
TIER1_METRICS = ["request_coverage", "intent_coverage", "coverage_intent_gap",
                 "echo_ratio", "answer_presence", "reference_exact_match",
                 "reference_token_f1", "reference_rouge_1", "reference_rouge_l"]
RESPONSE_METRICS = TIER0_METRICS + TIER1_METRICS

ROW_FIELDS = ["kind", "cell_id", "contrast_cell_id", "launch_id", "name",
             "n_items", "n_launches", "value", "min_value", "max_value",
             "p_value", "established", "note"]


# ---------- Identity ----------
def resolve_launch_dir(csv_path: Path) -> Optional[Path]:
    """The timestamped directory a launch wrote, searching upward from a CSV.

    A latency CSV sits directly in it; evaluation_items.csv sits one level
    deeper by default (`evaluation/pipeline.py:default_out_dir`). Searching
    rather than hard-coding a hop count handles both without caring which
    kind of file was passed in, and stays correct if evaluation output ever
    moves another level.
    """
    for candidate in (csv_path.parent, *csv_path.parents):
        if TIMESTAMP_RE.match(candidate.name):
            return candidate
    return None


def path_identity(csv_path: Path) -> Tuple[str, str]:
    """(cell, launch) guessed from the path, for files with no identity columns.

    00-INDEX.md: "the cell name is the PARENT of the timestamped directory."
    The timestamp is already a unique per-launch name, so it doubles as the
    launch fallback. When no ancestor looks like a timestamp - a hand-built
    sample tree, say - the immediate parent/grandparent are used the same
    way, on the assumption that whatever built the tree followed the same
    one-directory-per-launch shape even without real timestamps.
    """
    launch_dir = resolve_launch_dir(csv_path)
    if launch_dir is not None:
        return launch_dir.parent.name, launch_dir.name
    return csv_path.parent.parent.name, csv_path.parent.name


def _item_key(name: Optional[str]) -> str:
    """Normalize a recording name to the key the latency log uses.

    Mirrors `evaluation/latency.py:item_key` exactly (the log writes the
    stem, "00004"; a transcript or evaluation row may carry the filename,
    "00004.wav") so the two files cannot disagree on what an item is called.
    """
    if not name:
        return ""
    return Path(str(name).strip()).stem


def _parse_metric_value(raw: Optional[str]) -> Optional[float]:
    """A CSV cell to a float, treating Python's str(bool) as 1.0/0.0.

    `evaluation/aggregation.py:flat_row` writes `constraint_item_pass_strict`
    etc. as a raw Python bool, which csv.DictWriter renders as the literal
    text "True"/"False" rather than a number. Reading it as 1.0/0.0 is what
    makes "mean over items" equal the pass rate the plan asks for.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text == "True":
        return 1.0
    if text == "False":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def _ordered(names: Iterable[str], preferred: Sequence[str]) -> List[str]:
    """`preferred` first, in its own order, then anything else alphabetically."""
    names = set(names)
    ordered = [name for name in preferred if name in names]
    ordered += sorted(names - set(preferred))
    return ordered


# ---------- Reading: latency CSVs ----------
@dataclass
class LatencyRun:
    """One launch's latency CSV: its items in row order, and its durations."""
    path: Path
    cell: str
    launch: str
    ordered_items: List[str] = field(default_factory=list)
    durations: Dict[str, Dict[str, float]] = field(default_factory=dict)  # stage -> item -> ms
    context: Dict[str, Set[str]] = field(default_factory=dict)
    collapsed: int = 0

    @property
    def warmup_items(self) -> List[str]:
        return self.ordered_items[:WARMUP_ITEMS]


def read_latency_csv(path: Path) -> Optional[LatencyRun]:
    """Parse one latency CSV: median-collapse repeats, resolve cell/launch identity.

    Matches `aggregate_logs.py`'s `read_run` and `ttfa_residual_check.py`'s
    `read_run` on how repeated `(stage, item)` rows become one value: an item
    that held several timed utterances is represented by their median so it
    weighs the same as a single-utterance item, via `rstat.percentile` and
    nothing else (see the module docstring on why that call is not
    interchangeable with `statistics.median`, even though the two agree at
    q=0.5 - a second convention creeping in here would only take one q value
    that stops matching before anyone reads the code differently).
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
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

    # Blank counts as absent, not as a real identity: a bare CLI run without
    # --cell-id/--launch-id writes the column as "" (assistant.py:1472-1475),
    # and grouping every such run under one empty-string cell would silently
    # pool unrelated launches - precisely what this module exists to stop.
    cell_values = {(row.get("cell_id") or "").strip() for row in rows} - {""}
    launch_values = {(row.get("launch_id") or "").strip() for row in rows} - {""}
    fallback_cell, fallback_launch = path_identity(path)
    cell = sorted(cell_values)[0] if cell_values else fallback_cell
    launch = sorted(launch_values)[0] if launch_values else fallback_launch
    if len(cell_values) > 1 or len(launch_values) > 1:
        print(f"warning: {path} carries more than one cell_id/launch_id value "
             f"({sorted(cell_values)} / {sorted(launch_values)}); using "
             f"{cell!r}/{launch!r}", file=sys.stderr)

    run = LatencyRun(path=path, cell=cell, launch=launch, ordered_items=ordered_items)
    run.context = {col: set() for col in RUN_CONTEXT_COLUMNS}

    repeated: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        item = (row.get("item") or "").strip()
        stage = (row.get("stage") or "").strip()

        for col in RUN_CONTEXT_COLUMNS:
            value = (row.get(col) or "").strip()
            run.context[col].add(value or "(unset)")

        if not item or not stage:
            continue
        duration_str = (row.get("duration_ms") or "").strip()
        if duration_str:
            try:
                repeated.setdefault((stage, item), []).append(float(duration_str))
            except ValueError:
                pass

    for (stage, item), values in repeated.items():
        if len(values) > 1:
            run.collapsed += 1
        run.durations.setdefault(stage, {})[item] = rstat.percentile(values, 0.5)

    return run


def find_latency_csvs(root: Path) -> List[Path]:
    """Every latency CSV under a campaign root, or the single file named directly.

    Same glob as `aggregate_logs.py`'s `find_latest_csv_logs` and
    `ttfa_residual_check.py`'s `find_csvs`, so a tree that counts as N files
    there counts the same way here.
    """
    if root.is_file():
        return [root]
    return sorted(root.glob("**/latency_log_*.csv"))


# ---------- Reading: evaluation_items.csv ----------
@dataclass
class EvaluationRun:
    """One launch's evaluation_items.csv, melted to metric -> item -> value."""
    path: Path
    cell: str
    launch: str
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_items: int = 0
    n_warmup_dropped: int = 0


def read_evaluation_csv(path: Path, cell: str, launch: str,
                        warmup_key: Optional[str]) -> Optional[EvaluationRun]:
    """Melt one evaluation_items.csv into long form, excluding the warm-up item.

    `warmup_key` is the sibling latency CSV's first item (already normalized
    to a stem); passing it in, rather than recomputing "first row" from this
    file, is what keeps the two exclusions from drifting apart per the plan.
    Absent a sibling latency log - an evaluation-only tree - this file's own
    first row stands in, since there is nothing else to take the convention
    from.

    Reads utf-8-sig: `evaluation/reporting.py:write_item_csv` opens with that
    encoding, so the header's first column carries a BOM under plain utf-8
    and "item_id" would otherwise never match.
    """
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return None
    if not rows:
        return None

    if warmup_key is None:
        # evaluation/README.md: "filename ... the join key to the latency
        # log"; item_id may instead come from a scenario spec's own serial,
        # so filename is preferred and item_id only a fallback when it is
        # blank.
        first_raw = (rows[0].get("filename") or rows[0].get("item_id") or "").strip()
        warmup_key = _item_key(first_raw) if first_raw else None

    run = EvaluationRun(path=path, cell=cell, launch=launch)
    for row in rows:
        raw_name = (row.get("filename") or row.get("item_id") or "").strip()
        if not raw_name:
            continue
        item = _item_key(raw_name)
        run.n_items += 1
        if item == warmup_key:
            run.n_warmup_dropped += 1
            continue
        for metric in RESPONSE_METRICS:
            value = _parse_metric_value(row.get(metric))
            if value is None:
                continue
            run.metrics.setdefault(metric, {})[item] = value

    return run


def find_evaluation_csvs(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.name == "evaluation_items.csv" else []
    return sorted(root.glob("**/evaluation_items.csv"))


# ---------- Loading a campaign ----------
@dataclass
class Campaign:
    latency_runs: List[LatencyRun]
    evaluation_runs: List[EvaluationRun]
    warnings: List[str]


def load_campaign(root: Path) -> Campaign:
    """Walk the campaign root once, resolving identity and warm-up together.

    Evaluation's warm-up item is looked up from the latency run sharing its
    (cell, launch), so latency CSVs are read first.
    """
    warnings: List[str] = []

    by_identity: Dict[Tuple[str, str], LatencyRun] = {}
    for path in find_latency_csvs(root):
        run = read_latency_csv(path)
        if run is None:
            warnings.append(f"{path}: no readable item rows, skipped")
            continue
        key = (run.cell, run.launch)
        if key in by_identity:
            # Sorted iteration order makes this deterministic: the
            # lexicographically later latency_log_<timestamp>.csv name wins,
            # the same "newest" tie-break `evaluation/latency.py:
            # discover_latency_log` uses for one run directory holding more
            # than one log.
            warnings.append(f"{path}: another latency CSV already claims "
                            f"cell={run.cell!r} launch={run.launch!r} "
                            f"({by_identity[key].path}); keeping the latter")
        by_identity[key] = run
    latency_runs = list(by_identity.values())

    evaluation_runs: List[EvaluationRun] = []
    for path in find_evaluation_csvs(root):
        cell, launch = path_identity(path)
        sibling = by_identity.get((cell, launch))
        warmup_key = (sibling.ordered_items[0]
                     if sibling and sibling.ordered_items else None)
        run = read_evaluation_csv(path, cell, launch, warmup_key)
        if run is None:
            warnings.append(f"{path}: no readable item rows, skipped")
            continue
        evaluation_runs.append(run)

    return Campaign(latency_runs=latency_runs, evaluation_runs=evaluation_runs,
                    warnings=warnings)


def context_warnings(latency_runs: List[LatencyRun]) -> List[str]:
    """Flag a cell whose own launches disagree on a RUN_CONTEXT_COLUMNS value.

    Two launches of one cell are supposed to be repeats of the same
    experiment (assistant.py:1630-1636's comment on why cell_id/launch_id
    stay out of RUN_CONTEXT_COLUMNS in the first place). If they disagree on
    one of those columns anyway, the median-of-medians in step 2 is quietly
    averaging over a difference in what was run, not over launch noise -
    the same failure aggregate_logs.py's own collect_warnings checks for
    across the CSVs it pools, applied one level up.
    """
    warnings: List[str] = []
    by_cell: Dict[str, List[LatencyRun]] = {}
    for run in latency_runs:
        by_cell.setdefault(run.cell, []).append(run)

    for cell, runs in sorted(by_cell.items()):
        for col in RUN_CONTEXT_COLUMNS:
            seen: Set[str] = set()
            for run in runs:
                seen |= run.context.get(col, set())
            if len(seen) > 1:
                warnings.append(f"cell {cell!r}: its launches disagree on {col} "
                                f"({', '.join(sorted(seen))}); their launches are "
                                f"not repeats of the same experiment")
    return warnings


# ---------- Grouping for summaries and contrasts ----------
class GroupedSamples:
    """cell -> name -> launch -> {item: value}, for stages or metrics alike.

    "name" is a stage for the latency side, a metric for the response side;
    everything above this class is agnostic to which, so one summarizer and
    one contrast routine serve both.
    """

    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    def add(self, cell: str, name: str, launch: str, items: Dict[str, float]) -> None:
        if items:
            self.data.setdefault(cell, {}).setdefault(name, {})[launch] = items

    def names(self, cell: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        return self.data.get(cell, {})


def grouped_stage_samples(latency_runs: List[LatencyRun]) -> GroupedSamples:
    """Per launch, every stage's items with the warm-up item already dropped."""
    grouped = GroupedSamples()
    for run in latency_runs:
        warmup = set(run.warmup_items)
        for stage, per_item in run.durations.items():
            items = {item: value for item, value in per_item.items()
                     if item not in warmup}
            grouped.add(run.cell, stage, run.launch, items)
    return grouped


def grouped_metric_samples(evaluation_runs: List[EvaluationRun]) -> GroupedSamples:
    """Per launch, every metric's items (warm-up already dropped on read)."""
    grouped = GroupedSamples()
    for run in evaluation_runs:
        for metric, per_item in run.metrics.items():
            grouped.add(run.cell, metric, run.launch, per_item)
    return grouped


# ---------- Step 2: launch-level and cell-level summaries ----------
def _row(**kwargs: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {field_name: None for field_name in ROW_FIELDS}
    row.update(kwargs)
    return row


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return round(value, digits) if value is not None else None


LaunchValues = Dict[Tuple[str, str, str], float]  # (cell, name, launch) -> value


def summarize(grouped: GroupedSamples, kind: str, preferred: Sequence[str],
             launch_stat, centre_stat, digits: int) -> Tuple[List[Dict[str, Any]], LaunchValues]:
    """Per-launch values (`launch_stat` over items) and their centre/range.

    `launch_stat` and `centre_stat` take a list of floats and return one:
    the median (`rstat.percentile(values, 0.5)`) for a duration, so the
    result matches what `aggregate_logs.py` would report for that one
    launch's CSV; the mean (`statistics.fmean`) for a Tier 0/1 metric, so the
    result is the item-level pass/coverage rate rather than its midpoint.
    The plan is explicit that these differ ("the median of the R launch
    medians" for stages, "mean and range" for metrics) and conflating them
    would make a format-pass rate read like a duration instead of a fraction.

    Returns the raw (unrounded) per-launch values keyed by (cell, name,
    launch) alongside the display rows, because contrasts need the exact
    figures for their between-launch spread comparison - rounding those
    before the comparison could tip a borderline "established" call on a
    difference that is only a rounding artefact.
    """
    rows: List[Dict[str, Any]] = []
    launch_values: LaunchValues = {}

    for cell in sorted(grouped.data):
        for name in _ordered(grouped.data[cell], preferred):
            per_launch = grouped.data[cell][name]
            values: List[float] = []
            total_items = 0
            for launch in sorted(per_launch):
                items = per_launch[launch]
                if not items:
                    continue
                value = launch_stat(list(items.values()))
                launch_values[(cell, name, launch)] = value
                values.append(value)
                total_items += len(items)
                rows.append(_row(kind=kind, cell_id=cell, launch_id=launch, name=name,
                                 n_items=len(items), value=_round(value, digits)))
            if not values:
                continue
            centre = centre_stat(values)
            rows.append(_row(kind=kind, cell_id=cell, name=name,
                             n_items=total_items, n_launches=len(values),
                             value=_round(centre, digits),
                             min_value=_round(min(values), digits),
                             max_value=_round(max(values), digits)))
    return rows, launch_values


def _median(values: Sequence[float]) -> float:
    """The one and only percentile convention this module uses (see module
    docstring): never statistics.median, never a truncated-index shortcut."""
    return rstat.percentile(values, 0.5)


# ---------- Step 3: launch-blocked contrasts ----------
def _launch_range(launch_values: LaunchValues, cell: str, name: str) -> float:
    """Between-launch spread for one (cell, name), the step-2 range reused
    as the noise floor a contrast has to clear (see contrast_rows)."""
    values = [value for (c, n, _launch), value in launch_values.items()
             if c == cell and n == name]
    return (max(values) - min(values)) if len(values) >= 2 else 0.0


def contrast_rows(grouped: GroupedSamples, launch_values: LaunchValues,
                  baseline_cell: str, contrast_cell: str,
                  preferred: Sequence[str], digits: int) -> List[Dict[str, Any]]:
    """Launch r of `contrast_cell` against launch r of `baseline_cell`, per name.

    Reuses `rstat.compare_paired` exactly as it exists - the only thing added
    is running it once per matched launch instead of once over every item
    pooled - so shift estimates, Wilcoxon p-values and Hodges-Lehmann all
    still come from run_statistics.py, not from anything reimplemented here.

    shift = compare_paired(contrast_items, baseline_items).shift, i.e. a
    positive shift means `contrast_cell` measured higher than `baseline_cell`
    on that launch.

    "Established" is a plain, deterministic rule, not a new statistic: every
    matched launch's shift must share one sign, and the smallest of their
    magnitudes must exceed the larger of the two cells' own between-launch
    ranges from step 2. A shift that a cell's own launch-to-launch jitter
    could produce on its own is not distinguishable from that jitter, however
    consistent its sign looks across R launches.
    """
    rows: List[Dict[str, Any]] = []
    names_a = grouped.names(baseline_cell)
    names_b = grouped.names(contrast_cell)
    common_names = _ordered(set(names_a) & set(names_b), preferred)

    for name in common_names:
        launches_a = names_a[name]
        launches_b = names_b[name]
        matched = sorted(set(launches_a) & set(launches_b))
        unmatched = (set(launches_a) ^ set(launches_b))

        shifts: List[float] = []
        for launch in matched:
            comparison = rstat.compare_paired(launches_b[launch], launches_a[launch])
            if comparison.shift is not None:
                shifts.append(comparison.shift)
            rows.append(_row(kind="contrast", cell_id=baseline_cell,
                             contrast_cell_id=contrast_cell, launch_id=launch,
                             name=name, n_items=comparison.n_pairs,
                             value=_round(comparison.shift, digits),
                             p_value=_round(comparison.p_value, 4)))

        if not shifts:
            continue

        signs = {1 if s > 0 else (-1 if s < 0 else 0) for s in shifts}
        same_sign = signs == {1} or signs == {-1}
        noise_floor = max(_launch_range(launch_values, baseline_cell, name),
                          _launch_range(launch_values, contrast_cell, name))
        established = same_sign and min(abs(s) for s in shifts) > noise_floor

        notes = []
        if not same_sign:
            notes.append("sign disagreement across launches")
        if unmatched:
            notes.append(f"launches present on only one side, excluded: "
                         f"{', '.join(sorted(unmatched))}")

        rows.append(_row(kind="contrast", cell_id=baseline_cell,
                         contrast_cell_id=contrast_cell, name=name,
                         n_launches=len(shifts),
                         value=_round(_median(shifts), digits),
                         min_value=_round(min(shifts), digits),
                         max_value=_round(max(shifts), digits),
                         established=established, note="; ".join(notes)))
    return rows


# ---------- Output ----------
def write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value)
                             for key, value in row.items()})


def write_json(path: Path, root: Path, campaign: Campaign, warnings: List[str],
               rows: List[Dict[str, Any]]) -> None:
    payload = {
        "campaign_root": str(root),
        "warmup_items_per_launch": WARMUP_ITEMS,
        "latency_csvs": len(campaign.latency_runs),
        "evaluation_csvs": len(campaign.evaluation_runs),
        "cells": sorted({run.cell for run in campaign.latency_runs}
                        | {run.cell for run in campaign.evaluation_runs}),
        # The full list built in main(): load-time warnings plus anything
        # found while resolving --compare, not just campaign.warnings - a
        # --quiet, --json-only run must not lose a warning simply because the
        # text report was the only place it used to be printed.
        "warnings": warnings,
        "rows": rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_tidy_csv(path: Path, rows: List[Tuple[str, str, str, str, float]],
                   value_column: str, name_column: str) -> None:
    """The step-1 long table itself: cell_id, launch_id, item, <name>, <value>.

    Kept separate from the summary rows above (`ROW_FIELDS`) because this is
    the item-level table the plan's step 1 describes, one row per surviving
    item rather than per launch or per cell - what the summaries are built
    from, not the summaries.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "launch_id", "item", name_column, value_column])
        writer.writerows(rows)


def tidy_latency_rows(latency_runs: List[LatencyRun]) -> List[Tuple[str, str, str, str, float]]:
    out = []
    for run in latency_runs:
        warmup = set(run.warmup_items)
        for stage, per_item in run.durations.items():
            for item, value in per_item.items():
                if item not in warmup:
                    out.append((run.cell, run.launch, item, stage, value))
    return out


def tidy_metric_rows(evaluation_runs: List[EvaluationRun]) -> List[Tuple[str, str, str, str, float]]:
    out = []
    for run in evaluation_runs:
        for metric, per_item in run.metrics.items():
            for item, value in per_item.items():
                out.append((run.cell, run.launch, item, metric, value))
    return out


def render_text_report(root: Path, campaign: Campaign, warnings: List[str],
                       rows: List[Dict[str, Any]]) -> str:
    lines = [f"Campaign root: {root}"]

    cells = sorted({run.cell for run in campaign.latency_runs})
    launches = {(run.cell, run.launch) for run in campaign.latency_runs}
    dropped = sum(min(WARMUP_ITEMS, len(run.ordered_items))
                 for run in campaign.latency_runs)
    lines.append(f"{len(campaign.latency_runs)} latency CSV(s): "
                f"{len(cells)} cell(s), {len(launches)} (cell, launch) launch(es)")
    lines.append(f"warm-up exclusion: {dropped} item(s) dropped "
                f"({WARMUP_ITEMS} per launch; 00-INDEX.md - the first item of "
                f"every launch is bimodally fine or catastrophic)")

    if campaign.evaluation_runs:
        eval_dropped = sum(run.n_warmup_dropped for run in campaign.evaluation_runs)
        lines.append(f"{len(campaign.evaluation_runs)} evaluation_items.csv file(s): "
                    f"{eval_dropped} item(s) dropped by the same rule")

    for warning in warnings:
        lines.append(f"WARNING: {warning}")

    def section(kind: str, title: str, unit: str) -> None:
        cell_rows = [r for r in rows if r["kind"] == kind and r["launch_id"] is None]
        if not cell_rows:
            return
        lines.append("")
        lines.append(title)
        for r in cell_rows:
            lines.append(f"  {r['cell_id']:<28} {r['name']:<26} "
                        f"n_launches={r['n_launches']:<3} "
                        f"centre={r['value']}{unit}  "
                        f"range=[{r['min_value']}, {r['max_value']}]{unit}")

    section("stage", "Per-cell stage medians, median of R launch medians:", " ms")
    section("metric", "Per-cell response metrics, mean of R launch rates:", "")

    contrast_rollups = [r for r in rows if r["kind"] == "contrast" and r["launch_id"] is None]
    if contrast_rollups:
        lines.append("")
        lines.append("Launch-blocked contrasts (contrast minus baseline):")
        for r in contrast_rollups:
            verdict = "ESTABLISHED" if r["established"] else "not established"
            lines.append(f"  {r['contrast_cell_id']} vs {r['cell_id']:<20} {r['name']:<26} "
                        f"R={r['n_launches']:<2} shift={r['value']}  "
                        f"range=[{r['min_value']}, {r['max_value']}]  {verdict}"
                        + (f"  ({r['note']})" if r["note"] else ""))

    return "\n".join(lines)


# ---------- Entry point ----------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("campaign_root", type=str,
                        help="Directory to search recursively for latency_log_*.csv "
                             "and evaluation_items.csv files, or a single CSV file.")
    parser.add_argument("--compare", nargs=2, action="append", default=[],
                        metavar=("BASELINE_CELL", "CONTRAST_CELL"),
                        help="Launch-blocked contrast between two cell_id values "
                             "(or their path-fallback names). Repeatable.")
    parser.add_argument("--tsv", type=str, default=None,
                        help="Write the launch/cell summary table as TSV.")
    parser.add_argument("--json", type=str, default=None,
                        help="Write the launch/cell summary table as JSON.")
    parser.add_argument("--tidy-latency-csv", type=str, default=None,
                        help="Also write the item-level long table the latency "
                             "summaries are built from (cell_id,launch_id,item,"
                             "stage,duration_ms).")
    parser.add_argument("--tidy-metrics-csv", type=str, default=None,
                        help="Same, for the evaluation_items.csv metrics "
                             "(cell_id,launch_id,item,metric,value).")
    parser.add_argument("--quiet", action="store_true",
                        help="Skip the printed text report.")
    args = parser.parse_args()

    root = Path(args.campaign_root)
    if not root.exists():
        print(f"error: '{root}' does not exist", file=sys.stderr)
        return 1

    campaign = load_campaign(root)
    if not campaign.latency_runs and not campaign.evaluation_runs:
        print(f"no latency_log_*.csv or evaluation_items.csv files found under "
             f"'{root}'", file=sys.stderr)
        return 1

    warnings = list(campaign.warnings) + context_warnings(campaign.latency_runs)

    stage_grouped = grouped_stage_samples(campaign.latency_runs)
    metric_grouped = grouped_metric_samples(campaign.evaluation_runs)

    stage_rows, stage_launch_values = summarize(
        stage_grouped, "stage", KNOWN_STAGES, _median, _median, digits=1)
    metric_rows, metric_launch_values = summarize(
        metric_grouped, "metric", RESPONSE_METRICS, statistics.fmean,
        statistics.fmean, digits=4)
    rows = stage_rows + metric_rows

    for baseline, contrast in args.compare:
        known_cells = ({run.cell for run in campaign.latency_runs}
                      | {run.cell for run in campaign.evaluation_runs})
        missing = {baseline, contrast} - known_cells
        if missing:
            warnings.append(f"--compare {baseline} {contrast}: unknown cell(s) "
                            f"{sorted(missing)}, skipped")
            continue
        rows += contrast_rows(stage_grouped, stage_launch_values,
                              baseline, contrast, KNOWN_STAGES, digits=1)
        rows += contrast_rows(metric_grouped, metric_launch_values,
                              baseline, contrast, RESPONSE_METRICS, digits=4)

    # Printed to stderr unconditionally, the way aggregate_logs.py's own
    # main() prints analysis.warnings: --quiet silences the report, not the
    # warnings, since a --json-only run with a typo'd --compare cell must
    # still say so somewhere the caller will see it.
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if not args.quiet:
        print(render_text_report(root, campaign, warnings, rows))

    if args.tsv:
        write_tsv(Path(args.tsv), rows)
        print(f"Summary TSV written to: {Path(args.tsv).resolve()}")
    if args.json:
        write_json(Path(args.json), root, campaign, warnings, rows)
        print(f"Summary JSON written to: {Path(args.json).resolve()}")
    if args.tidy_latency_csv:
        write_tidy_csv(Path(args.tidy_latency_csv), tidy_latency_rows(campaign.latency_runs),
                       "duration_ms", "stage")
        print(f"Tidy latency table written to: {Path(args.tidy_latency_csv).resolve()}")
    if args.tidy_metrics_csv:
        write_tidy_csv(Path(args.tidy_metrics_csv), tidy_metric_rows(campaign.evaluation_runs),
                       "value", "metric")
        print(f"Tidy metrics table written to: {Path(args.tidy_metrics_csv).resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
