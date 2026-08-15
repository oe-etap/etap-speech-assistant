#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the latency logs a run leaves behind, one recording at a time.

A run over a folder of recordings writes one CSV row per stage per item. Each
item is one measurement of the pipeline, so the item is the unit everything
here counts in: a run over 120 recordings is a sample of 120, and its spread,
its quantiles and the precision of its estimates all follow from that.

Reports the distribution rather than the average alone - the median and the
upper quantiles with distribution-free confidence intervals, since a latency
sample is skewed and its mean sits somewhere in the tail nobody experiences.

With --compare it puts two runs side by side, pairing them recording by
recording and testing the shift between them, which is the only way to say
whether a change moved anything or the machine simply had a different morning.

Writes a formatted table for reading, a TSV for pasting into a spreadsheet and,
with --json-file, the whole thing including the per-item values for reanalysis.

Importable as much as runnable: aggregate() does from code what the command
line does, which is how mwe_assistant.py leaves a summary in the folder of
every run it finishes.
"""

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Any, Optional, Sequence, Set, Tuple, Union

import run_statistics as rstat


# Preferred order of stages for clear reporting
KNOWN_STAGES = ["stt", "stt_endpoint_delay", "llm_prompt_eval", "llm_ttft",
                "llm_first_chunk_fill", "llm_ttfc", "tts_first_chunk", "ttfa",
                "llm_eval", "tts_total", "e2e_response_ready"]

# Columns whose value decides how a row is to be read. Runs that disagree on
# them are not the same experiment, and averaging over the difference produces
# a number describing nothing.
RUN_CONTEXT_COLUMNS = ["input_mode", "audio_pacing", "utterance_trigger",
                       "stt_engine", "tts_engine", "mode"]

# Rates, each measured over the stage of the row it sits on. Averaging these
# across stages says little, since e2e_response_ready spans all the others.
PER_STAGE_RESOURCES = {
    "cpu_percent": "CPU (% system)",
    "gpu_util_percent": "GPU Util (%)",
}

# Levels that drift slowly over a run, so one pooled figure is informative.
# The pipeline's RSS and the LLM's are separate processes and add up; the LLM's
# VRAM is part of the device-wide figure above it, and does not. The model's own
# share is in turn part of what its process holds, the rest being the CUDA
# context and the runtime loaded beside it.
POOLED_RESOURCES = {
    "ram_percent": "RAM Usage (%)",
    "rss_mb": "RAM RSS (MB, pipeline)",
    "llm_rss_mb": "RAM RSS (MB, LLM server)",
    "gpu_mem_used_mb": "GPU Mem Used (MB, device)",
    "llm_vram_mb": "VRAM (MB, LLM process)",
    "llm_model_vram_mb": "VRAM (MB, LLM model)",
}

EXTRA_LABELS = {
    "stt_rtf": "STT Real Time Factor",
    "tokens_per_sec": "LLM Tokens/sec",
}

# Reported for every stage. 0.99 is deliberately absent: a distribution-free
# interval for it needs 368 items before both of its ends exist, so at
# the sizes these runs come in it is the largest observation wearing a label.
DEFAULT_QUANTILES = (0.5, 0.9, 0.95)

# Defaults shared by the command line and by whatever imports this module, so
# that a summary written during a run and one written afterwards agree.
DEFAULT_LOG_COUNT = 4
DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 5000

# What the three outputs are called when the caller names a folder rather than
# a file. A run folder carries them under these names.
SUMMARY_FILENAME = "log_averages_summary.txt"
TSV_FILENAME = "log_averages.tsv"
JSON_FILENAME = "log_averages.json"


@dataclass
class RunLog:
    """One CSV: what each recording measured, and under what configuration."""
    path: Path
    folder: str
    items: List[str] = field(default_factory=list)
    durations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stage_resources: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    pooled_resources: Dict[str, List[float]] = field(default_factory=dict)
    extras: Dict[str, Dict[str, float]] = field(default_factory=dict)
    context: Dict[str, Set[str]] = field(default_factory=dict)
    anchors: Set[str] = field(default_factory=set)
    prompts: Set[str] = field(default_factory=set)
    collapsed: int = 0
    skipped_warmup: List[str] = field(default_factory=list)


@dataclass
class Analysis:
    """Everything the report is built from."""
    runs: List[RunLog]
    requested_count: int
    warmup: int
    confidence: float
    quantiles: Sequence[float]
    samples: Dict[str, List[float]]                 # stage -> one value per item
    summaries: Dict[str, Optional[rstat.Summary]]
    resource_samples: Dict[str, List[float]]
    stage_resources: Dict[str, Dict[str, List[float]]]
    extra_samples: Dict[str, List[float]]
    context: Dict[str, List[str]]
    resamples: int = 0
    warnings: List[str] = field(default_factory=list)
    baseline: Optional["Analysis"] = None
    comparison: Dict[str, Any] = field(default_factory=dict)
    paired: bool = True
    primary: Optional[str] = None


# ---------- Reading ----------
def find_latest_csv_logs(outputs_dir: Path, count: int) -> List[Path]:
    """Find the latest `count` CSV logs under the outputs directory.

    A path pointing straight at a CSV, or at a single run folder, is taken as
    given so a specific run can be named without hunting for its timestamp.
    """
    if outputs_dir.is_file():
        return [outputs_dir]

    if not outputs_dir.exists():
        print(f"Error: Directory '{outputs_dir}' does not exist.", file=sys.stderr)
        return []

    # Search for all latency_log_*.csv files recursively
    csv_files = list(outputs_dir.glob("**/latency_log_*.csv"))

    # Fallback to any .csv files if specific pattern yields nothing
    if not csv_files:
        csv_files = [p for p in outputs_dir.glob("**/*.csv") if not p.name.endswith("_summary.csv")]

    if not csv_files:
        print(f"Warning: No CSV log files found in '{outputs_dir}'.", file=sys.stderr)
        return []

    # Sort files by modification time descending (latest first)
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return csv_files[:count]


def parse_csv_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a single latency log CSV file and return a list of record dicts."""
    records = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        print(f"Warning: Could not read '{file_path}': {e}", file=sys.stderr)
    return records


def read_run(file_path: Path, warmup: int = 0) -> Optional[RunLog]:
    """Read one CSV into per-item measurements.

    warmup drops the first N recordings of the run. The first item pays for
    whatever the run loads lazily - model weights, a cold page cache, a server
    that has not compiled a kernel yet - and in the logged runs its llm_ttfc
    lands at twice the median. It is a real cost, but it is a cost of starting,
    not of answering, and pooling it with the rest describes neither.
    """
    rows = parse_csv_file(file_path)
    if not rows:
        return None

    run = RunLog(path=file_path, folder=file_path.parent.name)
    run.context = {col: set() for col in RUN_CONTEXT_COLUMNS}
    run.pooled_resources = {key: [] for key in POOLED_RESOURCES}

    ordered_items: List[str] = []
    for row in rows:
        item = (row.get("item") or "").strip()
        if item and item not in ordered_items:
            ordered_items.append(item)

    skipped = set(ordered_items[:warmup]) if warmup > 0 else set()
    run.skipped_warmup = ordered_items[:warmup] if warmup > 0 else []
    run.items = [i for i in ordered_items if i not in skipped]

    # More than one row for the same item and stage means the item held several
    # utterances, each timed separately. Their median stands in for the item so
    # that every item weighs the same, whatever it happened to contain.
    repeated: Dict[Tuple[str, str], List[float]] = {}

    for row in rows:
        item = (row.get("item") or "").strip()
        stage = (row.get("stage") or "").strip()

        for col in RUN_CONTEXT_COLUMNS:
            value = (row.get(col) or "").strip()
            run.context[col].add(value or "(unset)")

        if item in skipped:
            continue

        duration_str = (row.get("duration_ms") or "").strip()
        if stage and item and duration_str:
            # Parsed before the key is touched: setdefault would otherwise leave
            # an empty list behind for a row whose duration does not convert,
            # and an empty list has no median.
            try:
                duration = float(duration_str)
            except ValueError:
                duration = None
            if duration is not None:
                repeated.setdefault((stage, item), []).append(duration)

        # Non-numeric values, such as the '[N/A]' markers nvidia-smi can emit,
        # are skipped.
        for res_key in (*POOLED_RESOURCES, *PER_STAGE_RESOURCES):
            val_str = (row.get(res_key) or "").strip()
            if not val_str:
                continue
            try:
                value = float(val_str)
            except ValueError:
                continue
            if res_key in POOLED_RESOURCES:
                run.pooled_resources[res_key].append(value)
            elif stage:
                run.stage_resources.setdefault(stage, {}).setdefault(res_key, []).append(value)

        extra_raw = (row.get("extra_json") or "").strip()
        if not extra_raw:
            continue
        try:
            extra = json.loads(extra_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(extra, dict):
            continue

        for key in EXTRA_LABELS:
            value = extra.get(key)
            if isinstance(value, (int, float)) and item:
                run.extras.setdefault(key, {})[item] = float(value)
        if extra.get("speech_end_source"):
            run.anchors.add(str(extra["speech_end_source"]))
        if extra.get("system_prompt"):
            run.prompts.add(str(extra["system_prompt"]))

    for (stage, item), values in repeated.items():
        if len(values) > 1:
            run.collapsed += 1
        run.durations.setdefault(stage, {})[item] = rstat.percentile(values, 0.5)

    return run


# ---------- Analysis ----------
def build_analysis(csv_files: List[Path], requested_count: int, warmup: int,
                   confidence: float, quantiles: Sequence[float],
                   resamples: int, seed: int) -> Optional[Analysis]:
    """Pool the selected runs into one sample per stage, one value per item."""
    runs = [run for run in (read_run(path, warmup) for path in csv_files) if run]
    if not runs:
        return None

    samples: Dict[str, List[float]] = {}
    extra_samples: Dict[str, List[float]] = {}
    resource_samples: Dict[str, List[float]] = {key: [] for key in POOLED_RESOURCES}
    stage_resources: Dict[str, Dict[str, List[float]]] = {}

    for run in runs:
        for stage, per_item in run.durations.items():
            samples.setdefault(stage, []).extend(per_item[i] for i in run.items if i in per_item)
        for key, per_item in run.extras.items():
            extra_samples.setdefault(key, []).extend(per_item[i] for i in run.items if i in per_item)
        for key, values in run.pooled_resources.items():
            resource_samples.setdefault(key, []).extend(values)
        for stage, metrics in run.stage_resources.items():
            for key, values in metrics.items():
                stage_resources.setdefault(stage, {}).setdefault(key, []).extend(values)

    context: Dict[str, List[str]] = {}
    for col in RUN_CONTEXT_COLUMNS:
        seen: Set[str] = set()
        for run in runs:
            seen |= run.context.get(col, set())
        if seen:
            context[col] = sorted(seen)

    summaries = {
        stage: rstat.summarize(values, quantiles, confidence, resamples, seed)
        for stage, values in samples.items()
    }

    analysis = Analysis(
        runs=runs, requested_count=requested_count, warmup=warmup,
        confidence=confidence, quantiles=quantiles, samples=samples,
        summaries=summaries, resource_samples=resource_samples,
        stage_resources=stage_resources, extra_samples=extra_samples,
        context=context, resamples=resamples,
    )
    analysis.warnings = collect_warnings(analysis)
    return analysis


def item_values(runs: List[RunLog], stage: str) -> Dict[str, float]:
    """One value per recording for a stage, across however many runs there are.

    Several runs over the same folder give each recording several readings; the
    median of them stands in, so that pairing against another set of runs still
    compares like with like.
    """
    gathered: Dict[str, List[float]] = {}
    for run in runs:
        per_item = run.durations.get(stage, {})
        for item in run.items:
            if item in per_item:
                gathered.setdefault(item, []).append(per_item[item])
    return {item: rstat.percentile(values, 0.5) for item, values in gathered.items()}


def compare_analyses(current: Analysis, baseline: Analysis, confidence: float,
                     primary: Optional[str] = None) -> Tuple[Dict[str, Any], bool]:
    """Test every stage between the two sides, correcting for the family size.

    Pairs by recording name where the two sides share one. Where they do not,
    falls back to the rank-sum test over the two independent samples, which is
    a far blunter instrument: the spread between recordings, which the pairing
    would have cancelled entirely, stays in as noise.

    Eleven stages tested at once is eleven chances to be unlucky, so the
    p-values carry a Holm correction over the family. A stage named as primary
    is held out of it: naming the metric the experiment is about, before seeing
    the result, is what buys the right to read it at full strength, and the
    remaining stages stay a family of their own.
    """
    stages = [s for s in KNOWN_STAGES if s in current.samples or s in baseline.samples]
    stages += sorted(s for s in current.samples
                     if s not in KNOWN_STAGES and s in baseline.samples)

    side_a = {stage: item_values(current.runs, stage) for stage in stages}
    side_b = {stage: item_values(baseline.runs, stage) for stage in stages}

    # One name shared by any stage is enough to pair on: the two sides ran over
    # the same recordings, and every stage they both carry pairs up the same way.
    paired = any(set(side_a[stage]) & set(side_b[stage]) for stage in stages)

    results: Dict[str, Any] = {}
    for stage in stages:
        if paired:
            results[stage] = rstat.compare_paired(side_a[stage], side_b[stage], confidence)
        else:
            results[stage] = rstat.compare_unpaired(
                current.samples.get(stage, []), baseline.samples.get(stage, []))

    family = [s for s in stages if s in results and s != primary]
    adjusted = rstat.holm_adjust([results[s].p_value for s in family])
    for stage, value in zip(family, adjusted):
        results[stage].p_adjusted = value
    if primary in results:
        results[primary].p_adjusted = results[primary].p_value

    return results, paired


def collect_warnings(analysis: Analysis) -> List[str]:
    """Everything about the selection that would make a reader misread it.

    The first line of each warning stands alone; anything indented under it
    continues that same warning rather than starting a new one.
    """
    warnings: List[str] = []

    mixed = {col: values for col, values in analysis.context.items() if len(values) > 1}
    if mixed:
        warnings += ["these runs are not the same experiment. Averaging over them",
                     "  produces numbers that describe no actual configuration:"]
        warnings += [f"    {col}: {', '.join(values)}" for col, values in mixed.items()]
        warnings.append("  Narrow the selection with --outputs-dir, or lower --log-count.")

    if "(unset)" in analysis.context.get("input_mode", []):
        warnings += ["some runs predate the input_mode/audio_pacing columns. Their",
                     "  'ttfa' was measured from the start of processing and therefore includes",
                     "  however long the speaker talked. It is not the speech-end anchored TTFA",
                     "  reported here, and the two must not be pooled. Re-run to compare."]

    anchors = set().union(*(run.anchors for run in analysis.runs)) if analysis.runs else set()
    if len(anchors) > 1:
        warnings += [f"the end of speech was located more than one way in this selection: "
                     f"{', '.join(sorted(anchors))}.",
                     "  'ttfa' and 'stt_endpoint_delay' are measured from that instant, so the",
                     "  items are not reporting the same quantity and part of their spread is",
                     "  the difference between the anchors rather than anything the code did."]

    prompts = set().union(*(run.prompts for run in analysis.runs)) if analysis.runs else set()
    if len(prompts) > 1:
        warnings.append(f"more than one system prompt is present ({len(prompts)}), which moves "
                        f"llm_prompt_eval and llm_eval directly.")

    collapsed = sum(run.collapsed for run in analysis.runs)
    if collapsed:
        warnings.append(f"{collapsed} item/stage pair{'s' if collapsed > 1 else ''} held more "
                        f"than one utterance; each is represented by its median so every "
                        f"recording carries equal weight.")

    counts = {len(run.items) for run in analysis.runs}
    if len(analysis.runs) > 1 and len(counts) > 1:
        warnings.append(f"the runs cover different numbers of recordings ({sorted(counts)}), so "
                        f"the pooled quantiles weigh the longer runs more heavily.")

    smallest = min((s.n for s in analysis.summaries.values() if s), default=0)
    if 0 < smallest < 20:
        warnings.append(f"the smallest stage sample holds {smallest} recording"
                        f"{'s' if smallest > 1 else ''}. Below about 20 the upper quantiles have "
                        f"no interval to report and the median's is wider than most differences "
                        f"worth chasing.")

    return warnings


# ---------- Formatting ----------
def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> Tuple[List[str], int]:
    """Pad every column to its widest cell, then join with tabs.

    The padding is what makes it readable in Notepad; the tabs are what make
    the same text paste into a spreadsheet as columns.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: Sequence[str]) -> str:
        first = cells[0].ljust(widths[0])
        rest = [cell.rjust(widths[index + 1]) for index, cell in enumerate(cells[1:])]
        return "\t".join([first] + rest)

    width = sum(widths) + len(widths) - 1
    return [line(headers)] + [line(row) for row in rows], width


def number(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def interval_text(interval: rstat.Interval, digits: int = 0) -> str:
    """An interval, or which end of it the sample could not place."""
    if interval.bounded:
        return f"{interval.low:.{digits}f} - {interval.high:.{digits}f}"
    if interval.low is None and interval.high is None:
        return "n too small"
    if interval.high is None:
        return f">= {interval.low:.{digits}f}"
    return f"<= {interval.high:.{digits}f}"


def p_text(p_value: Optional[float]) -> str:
    if p_value is None:
        return "N/A"
    if p_value < 0.0001:
        return "<0.0001"
    return f"{p_value:.4f}"


def run_labels(runs: Sequence[RunLog]) -> List[str]:
    """Name each run by its folder, falling back to the file where that repeats.

    Runs normally sit one to a timestamped folder, so the folder names them.
    Several logs gathered into one directory do not, and two rows both reading
    'cur' would be worse than useless.
    """
    folders = [run.folder for run in runs]
    return [run.folder if folders.count(run.folder) == 1 else f"{run.folder}/{run.path.stem}"
            for run in runs]


def stage_order(*sources: Dict[str, Any]) -> List[str]:
    """KNOWN_STAGES first, anything new after it, nothing twice."""
    present = set()
    for source in sources:
        present |= set(source)
    return [s for s in KNOWN_STAGES if s in present] + sorted(present - set(KNOWN_STAGES))


def format_report(analysis: Analysis) -> str:
    """The whole report, as one block of text."""
    lines: List[str] = []
    thick = "=" * 100
    thin = "-" * 100
    confidence_label = f"{analysis.confidence:.0%}"

    # Sections number themselves, so the ones that have nothing to report can
    # drop out without leaving a gap in the sequence.
    counter = [0]

    def section(title: str, headers: Sequence[str], rows: Sequence[Sequence[str]],
                notes: Sequence[str] = ()) -> None:
        if not rows:
            return
        counter[0] += 1
        numbered = f"{counter[0]}. {title}"
        body, width = render_table(headers, rows)
        rule = "=" * max(width, len(numbered))
        lines.append(numbered)
        lines.append(rule)
        lines.append(body[0])
        lines.append("-" * max(width, len(numbered)))
        lines.extend(body[1:])
        lines.append(rule)
        lines.extend(notes)
        lines.append("")

    # ----- Header -----
    lines.append(thick)
    lines.append("                        PERFORMANCE LOG STATISTICAL SUMMARY")
    lines.append(thick)
    lines.append(f"Generated On         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Requested Log Count  : {analysis.requested_count}")
    lines.append(f"Analyzed Log Count   : {len(analysis.runs)}")
    lines.append(f"Recordings Analyzed  : {sum(len(run.items) for run in analysis.runs)}")
    if analysis.warmup:
        dropped = sum(len(run.skipped_warmup) for run in analysis.runs)
        lines.append(f"Warm-up Dropped      : {dropped} "
                     f"(first {analysis.warmup} of each run)")
    lines.append(f"Confidence Level     : {confidence_label}")
    lines.append("")
    lines.append("Configuration:")
    for col in RUN_CONTEXT_COLUMNS:
        values = analysis.context.get(col)
        if values:
            lines.append(f"  {col:<18}: {', '.join(values)}")

    if analysis.warnings:
        lines.append("")
        lines.append("!" * 100)
        for warning in analysis.warnings:
            lines.append(f"WARNING: {warning}" if not warning.startswith(" ") else warning)
        lines.append("!" * 100)

    lines.append("")
    lines.append("Included Log Files:")
    for idx, run in enumerate(analysis.runs, 1):
        lines.append(f"  {idx}. {run.path}  ({len(run.items)} recordings)")
    if analysis.baseline:
        lines.append("")
        lines.append("Baseline Log Files (--compare):")
        for idx, run in enumerate(analysis.baseline.runs, 1):
            lines.append(f"  {idx}. {run.path}  ({len(run.items)} recordings)")
    lines.append(thick)
    lines.append("")

    # ----- 1. Distribution -----
    quantile_headers = [f"p{int(q * 100)}" for q in analysis.quantiles]
    headers = ["Stage", "n", "Mean", "SD", "Min"] + quantile_headers + ["Max", "IQR"]
    rows = []
    for stage in stage_order(analysis.samples):
        summary = analysis.summaries.get(stage)
        if not summary:
            continue
        rows.append([stage, str(summary.n), number(summary.mean), number(summary.sd),
                     number(summary.minimum)]
                    + [number(summary.quantiles.get(q)) for q in analysis.quantiles]
                    + [number(summary.maximum), number(summary.iqr)])
    section("STAGE LATENCY DISTRIBUTION (per recording, ms)", headers, rows, notes=[
        "One observation per recording. The median is the figure to quote: the mean of a",
        "latency sample is pulled into a tail that no single recording occupies.",
    ])

    # ----- 2. Precision -----
    headers = ["Stage", "n", "Median", f"{confidence_label} CI (median)", "+/-",
               "Mean", f"{confidence_label} CI (mean)"]
    rows = []
    for stage in stage_order(analysis.samples):
        summary = analysis.summaries.get(stage)
        if not summary:
            continue
        median_ci = summary.quantile_intervals.get(0.5, rstat.Interval(None, None))
        half = median_ci.width / 2 if median_ci.bounded else None
        relative = f"{half / summary.median:.1%}" if half and summary.median else "N/A"
        # An absent mean interval means one of two different things, and saying
        # the sample was too small when the bootstrap was simply turned off
        # would send the reader after more recordings for no reason.
        mean_ci = ("(off)" if not analysis.resamples
                   else interval_text(summary.mean_interval))
        rows.append([stage, str(summary.n), number(summary.median),
                     interval_text(median_ci), relative,
                     number(summary.mean), mean_ci])
    mean_note = (f"The mean's is a percentile bootstrap over {analysis.resamples} resamples, "
                 f"seeded so the report does not move; it needs "
                 f"{rstat.MIN_N_FOR_BOOTSTRAP} recordings to mean anything."
                 if analysis.resamples else
                 "The mean carries no interval here: --bootstrap 0 turned the resampling off.")
    section(f"ESTIMATE PRECISION ({confidence_label} confidence intervals, ms)",
            headers, rows, notes=[
        "The median's interval is distribution-free: it comes from which order statistics",
        "bracket the true median, which holds whatever shape the data has.",
        mean_note,
        "'+/-' is half the median's interval as a fraction of the median: the precision this",
        "many recordings bought. A difference smaller than it is not resolvable here.",
    ])

    # ----- 3. Upper quantile intervals -----
    upper = [q for q in analysis.quantiles if q > 0.5]
    if upper:
        headers = ["Stage", "n"]
        for q in upper:
            headers += [f"p{int(q * 100)}", f"{confidence_label} CI"]
        rows = []
        for stage in stage_order(analysis.samples):
            summary = analysis.summaries.get(stage)
            if not summary:
                continue
            row = [stage, str(summary.n)]
            for q in upper:
                row.append(number(summary.quantiles.get(q)))
                row.append(interval_text(summary.quantile_intervals.get(q, rstat.Interval(None, None))))
            rows.append(row)
        section("TAIL QUANTILES AND WHAT THEY REST ON", headers, rows, notes=[
            "An upper quantile is held up by the handful of recordings above it, so its",
            "interval is wide even where the median's is tight. A one-sided '>= x' means the",
            "run has nothing above the quantile to bound it with, and 'n too small' that it",
            "cannot place either end: p95 needs 72 recordings before both exist, p99 needs 368,",
            "which is why p99 is not in the default set.",
        ])

    # ----- 4. Per-run breakdown -----
    summary_stages = [s for s in ["stt", "stt_endpoint_delay", "llm_ttfc", "ttfa",
                                  "e2e_response_ready"] if s in analysis.samples]
    if summary_stages:
        headers = ["Run", "n"] + [f"{s} p50" for s in summary_stages]
        rows = []
        for run, label in zip(analysis.runs, run_labels(analysis.runs)):
            row = [label, str(len(run.items))]
            for stage in summary_stages:
                values = [run.durations[stage][i] for i in run.items
                          if i in run.durations.get(stage, {})]
                row.append(number(rstat.percentile(values, 0.5)) if values else "N/A")
            rows.append(row)
        if len(analysis.runs) > 1:
            pooled = ["POOLED", str(max((len(analysis.samples.get(s, []))
                                         for s in summary_stages), default=0))]
            pooled += [number(rstat.percentile(analysis.samples.get(s, []), 0.5))
                       for s in summary_stages]
            rows.append(pooled)
        notes = ["TTFA is measured from the end of the speech. It is blank under 'fast'",
                 "pacing, where no point inside the audio maps to a wall-clock instant."]
        if len(analysis.runs) > 1:
            notes += [
                "Read the per-run medians before the pooled one. Runs differ by more than the",
                "code between them - thermal state, what else the machine was doing, which",
                "weights were still cached - and a spread across these rows that rivals the",
                "difference being investigated means the pooled interval understates it.",
            ]
        section("PER-RUN BREAKDOWN (median per run, ms)", headers, rows, notes)

    # ----- 5. Comparison -----
    if analysis.comparison:
        counter[0] += 1
        lines.extend(format_comparison(analysis, counter[0]))

    # ----- 6. Per-stage resources -----
    if analysis.stage_resources:
        headers = ["Stage", "CPU p50", "CPU p90", "CPU max", "GPU p50", "GPU max", "Samples"]
        rows = []
        for stage in stage_order(analysis.stage_resources):
            metrics = analysis.stage_resources.get(stage) or {}
            cpu = metrics.get("cpu_percent", [])
            gpu = metrics.get("gpu_util_percent", [])
            if not cpu and not gpu:
                continue
            rows.append([
                stage,
                number(rstat.percentile(cpu, 0.5)), number(rstat.percentile(cpu, 0.9)),
                number(max(cpu)) if cpu else "N/A",
                number(rstat.percentile(gpu, 0.5)), number(max(gpu)) if gpu else "N/A",
                str(len(cpu) or len(gpu)),
            ])
        section("PER-STAGE RESOURCE USAGE", headers, rows, notes=[
            "Each value is averaged over the stage of its own row. Both figures are",
            "machine-wide, so concurrent stages share the same load and the numbers",
            "cannot be added up; e2e_response_ready already spans the whole item.",
            "GPU utilisation carries the driver's own lag, which outlasts the work",
            "and so reaches rows whose stage only overlapped it.",
        ])

    # ----- 7. Pooled resources -----
    rows = []
    for key, label in POOLED_RESOURCES.items():
        values = analysis.resource_samples.get(key, [])
        if values:
            rows.append([label, number(rstat.percentile(values, 0.5)),
                         number(rstat.percentile(values, 0.9)),
                         number(min(values)), number(max(values)), str(len(values))])
    section("OVERALL RESOURCE LEVELS", ["Resource Metric", "p50", "p90", "Min", "Max", "Samples"],
            rows)

    # ----- 8. Extra metrics -----
    rows = []
    for key, label in EXTRA_LABELS.items():
        values = analysis.extra_samples.get(key, [])
        if not values:
            continue
        summary = rstat.summarize(values, analysis.quantiles, analysis.confidence, 0)
        median_ci = summary.quantile_intervals.get(0.5, rstat.Interval(None, None))
        rows.append([label, str(summary.n), number(summary.median, 3),
                     interval_text(median_ci, 3), number(summary.mean, 3),
                     number(summary.minimum, 3), number(summary.maximum, 3)])
    section("EXTRA METRICS (STT RTF & LLM Tokens/sec)",
            ["Extra Metric", "n", "Median", f"{confidence_label} CI", "Mean", "Min", "Max"], rows)

    return "\n".join(lines)


def format_comparison(analysis: Analysis, index: int) -> List[str]:
    """The A/B table: what moved, by how much, and whether it can be believed."""
    lines: List[str] = []
    confidence_label = f"{analysis.confidence:.0%}"
    results = analysis.comparison

    if analysis.paired:
        title = f"{index}. A/B COMPARISON, PAIRED BY RECORDING (current - baseline, ms)"
        headers = ["Stage", "pairs", "Current p50", "Baseline p50", "Shift",
                   f"{confidence_label} CI (shift)", "Change", "p", "p (Holm)", "Detectable"]
    else:
        title = f"{index}. A/B COMPARISON, UNPAIRED (current - baseline, ms)"
        headers = ["Stage", "n cur", "n base", "Current p50", "Baseline p50", "Shift",
                   "Cliff's d", "p", "p (Holm)"]

    rows = []
    for stage in stage_order(results):
        result = results[stage]
        # The primary stage is marked so nobody has to remember which column of
        # the two p-values applies to it.
        label = f"{stage} *" if stage == analysis.primary else stage
        if analysis.paired:
            if not result.n_pairs:
                continue
            change = (f"{result.relative_shift:+.1%}"
                      if result.relative_shift is not None else "N/A")
            rows.append([
                label, str(result.n_pairs), number(result.median_a), number(result.median_b),
                f"{result.shift:+.1f}" if result.shift is not None else "N/A",
                interval_text(result.shift_interval, 1),
                change, p_text(result.p_value), p_text(result.p_adjusted),
                number(result.detectable_shift),
            ])
        else:
            if not result.n_a or not result.n_b:
                continue
            rows.append([
                label, str(result.n_a), str(result.n_b),
                number(result.median_a), number(result.median_b),
                f"{result.shift:+.1f}" if result.shift is not None else "N/A",
                number(result.effect_size, 2), p_text(result.p_value), p_text(result.p_adjusted),
            ])

    if not rows:
        return []

    body, width = render_table(headers, rows)
    rule = "=" * max(width, len(title))
    lines.append(title)
    lines.append(rule)
    lines.append(body[0])
    lines.append("-" * max(width, len(title)))
    lines.extend(body[1:])
    lines.append(rule)

    if analysis.paired:
        # Stages that could not be tested report why in place of a method; those
        # belong in their own row, not in the line naming the test that ran.
        methods = sorted({r.method for r in results.values()
                          if getattr(r, "n_pairs", 0) and r.p_value is not None})
        lines += [
            f"Test: {', '.join(methods)}, two-sided." if methods
            else "No stage had enough paired recordings to test.",
            "'Shift' is the Hodges-Lehmann estimate of the median difference, and a negative",
            "one means the current run is faster. Its interval clearing zero and a small p are",
            "the same statement made twice.",
            "'p (Holm)' holds the chance of any false positive across all stages at "
            f"{1 - analysis.confidence:.0%}; read it,",
            "not the raw p, when scanning the column for something that moved.",
        ]
        if analysis.primary:
            lines += [
                f"'{analysis.primary}' is marked * as the primary endpoint and is not in that",
                "family: its p stands uncorrected, which it is entitled to only because it was",
                "named before the result was seen. The rest remain exploratory.",
            ]
        else:
            lines += [
                "With no --primary named, every stage pays for all eleven being tested. A real",
                "effect can sit at p = 0.006 and still not clear the corrected line, so name the",
                "stage the experiment is about up front if there is one.",
            ]
        lines += [
            "'Detectable' is the smallest true shift this many pairs would catch four times in",
            "five. A shift well under it is not evidence of no change - it is a sample that",
            "could not have shown one.",
            "",
            "Both sides are single sets of runs on one machine, so a difference here is the",
            "difference between these runs, which includes everything else that changed with",
            "them. To attribute it to the code, run each side more than once, interleaved.",
        ]
    else:
        lines += [
            "The two sides share no recording names, so this is the rank-sum test over",
            "independent samples. It is much the weaker comparison: at 120 recordings a side it",
            "needs a shift some four times larger than the paired test to reach the same",
            "confidence, because the differences between recordings stay in as noise.",
            "Run both sides over the same audio folder to get the pairing back.",
            "Cliff's d is how much more often a current recording beats a baseline one than the",
            "reverse: 0 is no separation, +/-1 is complete.",
        ]
    lines.append("")
    return lines


def format_tsv(analysis: Analysis) -> str:
    """The stage summary as a plain TSV, one row per stage."""
    quantile_headers = [f"p{int(q * 100)}_ms" for q in analysis.quantiles]
    header = (["stage", "n", "mean_ms", "sd_ms", "min_ms"] + quantile_headers
              + ["max_ms", "median_ci_low_ms", "median_ci_high_ms",
                 "mean_ci_low_ms", "mean_ci_high_ms"])
    if analysis.comparison:
        header += ["baseline_p50_ms", "shift_ms", "shift_ci_low_ms", "shift_ci_high_ms",
                   "p_value", "p_holm"]

    lines = ["\t".join(header)]
    for stage in stage_order(analysis.samples):
        summary = analysis.summaries.get(stage)
        if not summary:
            continue
        median_ci = summary.quantile_intervals.get(0.5, rstat.Interval(None, None))
        row = [stage, str(summary.n), f"{summary.mean:.2f}",
               number(summary.sd, 2), f"{summary.minimum:.2f}"]
        row += [number(summary.quantiles.get(q), 2) for q in analysis.quantiles]
        row += [f"{summary.maximum:.2f}",
                number(median_ci.low, 2), number(median_ci.high, 2),
                number(summary.mean_interval.low, 2), number(summary.mean_interval.high, 2)]
        if analysis.comparison:
            result = analysis.comparison.get(stage)
            if result is None:
                row += ["N/A"] * 6
            else:
                interval = getattr(result, "shift_interval", rstat.Interval(None, None))
                row += [number(result.median_b, 2), number(result.shift, 2),
                        number(interval.low, 2), number(interval.high, 2),
                        p_text(result.p_value), p_text(result.p_adjusted)]
        lines.append("\t".join(row))
    return "\n".join(lines)


def build_json(analysis: Analysis) -> Dict[str, Any]:
    """The whole analysis, per-item values included, for reanalysis elsewhere."""

    def interval_pair(interval: rstat.Interval) -> Dict[str, Optional[float]]:
        return {"low": interval.low, "high": interval.high}

    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "confidence": analysis.confidence,
        "warmup_dropped_per_run": analysis.warmup,
        "configuration": analysis.context,
        "warnings": analysis.warnings,
        "runs": [{"path": str(run.path), "folder": run.folder,
                  "recordings": len(run.items),
                  "skipped_warmup": run.skipped_warmup,
                  "speech_end_sources": sorted(run.anchors)} for run in analysis.runs],
        "stages": {},
        "items": {},
    }

    for stage in stage_order(analysis.samples):
        summary = analysis.summaries.get(stage)
        if not summary:
            continue
        payload["stages"][stage] = {
            "n": summary.n,
            "mean": summary.mean,
            "sd": summary.sd,
            "min": summary.minimum,
            "max": summary.maximum,
            "quantiles": {str(q): v for q, v in summary.quantiles.items()},
            "quantile_intervals": {str(q): interval_pair(iv)
                                   for q, iv in summary.quantile_intervals.items()},
            "mean_interval": interval_pair(summary.mean_interval),
        }
        payload["items"][stage] = item_values(analysis.runs, stage)

    if analysis.comparison:
        payload["comparison"] = {"paired": analysis.paired, "primary": analysis.primary,
                                 "stages": {}}
        if analysis.baseline:
            payload["comparison"]["baseline_runs"] = [str(r.path) for r in analysis.baseline.runs]
        for stage, result in analysis.comparison.items():
            entry = {
                "median_current": result.median_a,
                "median_baseline": result.median_b,
                "shift": result.shift,
                "p_value": result.p_value,
                "p_holm": result.p_adjusted,
                "method": result.method,
            }
            if analysis.paired:
                entry.update({
                    "pairs": result.n_pairs,
                    "zero_differences": result.n_zero,
                    "shift_interval": interval_pair(result.shift_interval),
                    "detectable_shift": result.detectable_shift,
                    "relative_shift": result.relative_shift,
                })
            else:
                entry.update({"n_current": result.n_a, "n_baseline": result.n_b,
                              "cliffs_delta": result.effect_size})
            payload["comparison"]["stages"][stage] = entry

    return payload


# ---------- Callable interface ----------
PathLike = Union[str, os.PathLike]

# How a selection of logs may be named: a directory to search, a single run
# folder, one CSV, or the paths outright.
LogSelection = Union[PathLike, Sequence[PathLike]]


@dataclass
class Report:
    """A rendered analysis, and wherever it was written."""
    analysis: Analysis
    text: str
    tsv: str
    files: Dict[str, Path] = field(default_factory=dict)

    @property
    def warnings(self) -> List[str]:
        return self.analysis.warnings

    def as_json(self) -> Dict[str, Any]:
        """The whole analysis, the per-recording values included."""
        return build_json(self.analysis)


def normalize_quantiles(values: Iterable[float]) -> List[float]:
    """Percentages or fractions, in any order, to the fractions reported.

    The median is always among them: the report quotes it as the figure to
    read, and a caller asking only for the tail did not mean to remove it.
    """
    quantiles = [value / 100 if value > 1 else value for value in map(float, values)]
    for quantile in quantiles:
        if not 0 < quantile < 1:
            raise ValueError(f"quantile out of range: {quantile}")
    return sorted(set(quantiles) | {0.5})


def check_options(log_count: int, confidence: float, warmup: int) -> None:
    """Reject the arguments that describe no analysis, whatever asked for one."""
    if log_count <= 0:
        raise ValueError("log count must be a positive integer")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must sit between 0.5 and 1")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")


def resolve_logs(logs: LogSelection, log_count: int = DEFAULT_LOG_COUNT) -> List[Path]:
    """The CSVs a selection names.

    A directory is searched for its latest `log_count` logs; a run folder or a
    single CSV names itself. A sequence is taken as given, since a caller that
    assembled one has already chosen.
    """
    if isinstance(logs, (str, os.PathLike)):
        return find_latest_csv_logs(Path(logs), log_count)
    return [Path(path) for path in logs]


def analyze_logs(logs: LogSelection, *, log_count: int = DEFAULT_LOG_COUNT, warmup: int = 0,
                 confidence: float = DEFAULT_CONFIDENCE,
                 quantiles: Iterable[float] = DEFAULT_QUANTILES,
                 resamples: int = DEFAULT_RESAMPLES, seed: int = rstat.DEFAULT_SEED,
                 compare: Optional[LogSelection] = None, compare_count: int = 1,
                 primary: Optional[str] = None) -> Optional[Analysis]:
    """Pool a selection of runs into one analysis, against a baseline if given.

    Returns None where the selection holds nothing readable. That is a state to
    handle rather than an error: a run that logged nothing has nothing to say
    about itself. A baseline that cannot be read is the lesser problem, the
    current runs still standing on their own, so it warns and reports them.

    Raises ValueError on arguments that describe no analysis.
    """
    check_options(log_count, confidence, warmup)
    quantiles = normalize_quantiles(quantiles)

    csv_files = resolve_logs(logs, log_count)
    if not csv_files:
        return None

    analysis = build_analysis(csv_files, log_count, warmup, confidence,
                              quantiles, resamples, seed)
    if not analysis or not compare:
        return analysis

    baseline_files = resolve_logs(compare, max(1, compare_count))
    if not baseline_files:
        print(f"Warning: no baseline logs found in '{compare}'; "
              f"reporting the current runs alone.", file=sys.stderr)
        return analysis

    chosen = {path.resolve() for path in csv_files}
    baseline_files = [path for path in baseline_files if path.resolve() not in chosen]
    if not baseline_files:
        print("Warning: the baseline selected the same logs as the analysis; "
              "skipping the comparison.", file=sys.stderr)
        return analysis

    baseline = build_analysis(baseline_files, compare_count, warmup, confidence,
                              quantiles, resamples, seed)
    if not baseline:
        return analysis

    analysis.baseline = baseline
    analysis.primary = primary
    analysis.comparison, analysis.paired = compare_analyses(
        analysis, baseline, confidence, primary)
    if primary and primary not in analysis.comparison:
        print(f"Warning: the primary stage '{primary}' is not one either run recorded; "
              f"no stage was held out of the correction.", file=sys.stderr)
        analysis.primary = None

    return analysis


def render_report(analysis: Analysis) -> Report:
    """Everything the analysis renders to, written nowhere yet."""
    return Report(analysis=analysis, text=format_report(analysis), tsv=format_tsv(analysis))


def write_report(report: Report, output_file: Optional[PathLike] = None,
                 tsv_file: Optional[PathLike] = None,
                 json_file: Optional[PathLike] = None) -> Dict[str, Path]:
    """Write whichever outputs were asked for; answer with the ones that landed.

    A path that cannot be written is reported and skipped rather than raised:
    the analysis behind it is the expensive part, and losing the other two
    outputs, or a caller's run, over one unwritable directory helps nobody.
    """
    targets = [("text", output_file, report.text), ("tsv", tsv_file, report.tsv)]
    if json_file:
        targets.append(("json", json_file, json.dumps(report.as_json(), indent=2)))

    for kind, target, content in targets:
        if not target:
            continue
        path = Path(target)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            print(f"Error writing {kind} file '{path}': {e}", file=sys.stderr)
            continue
        report.files[kind] = path

    return report.files


def aggregate(logs: LogSelection, output_file: Optional[PathLike] = None,
              tsv_file: Optional[PathLike] = None,
              json_file: Optional[PathLike] = None, **options) -> Optional[Report]:
    """Analyze a selection of runs and write the report where asked for.

    The whole of what the command line does, in one call, so that a run can
    summarize itself the moment it finishes rather than waiting for somebody to
    remember the script. `options` are analyze_logs's, and None comes back for
    the same reason it does there.
    """
    analysis = analyze_logs(logs, **options)
    if analysis is None:
        return None

    report = render_report(analysis)
    write_report(report, output_file, tsv_file, json_file)
    return report


def summarize_run(run_dir: PathLike, logs: Optional[LogSelection] = None,
                  **options) -> Optional[Report]:
    """Summarize one run into its own folder, under the standard three names.

    `logs` defaults to the folder itself, which is where a run leaves its CSV;
    it is worth passing when the log was directed elsewhere.
    """
    run_dir = Path(run_dir)
    options.setdefault("log_count", 1)
    return aggregate(run_dir if logs is None else logs,
                     output_file=run_dir / SUMMARY_FILENAME,
                     tsv_file=run_dir / TSV_FILENAME,
                     json_file=run_dir / JSON_FILENAME,
                     **options)


# ---------- Entry point ----------
def parse_quantiles(text: str) -> List[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    try:
        return normalize_quantiles(values)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-c", "--log-count", type=int, default=DEFAULT_LOG_COUNT,
                        help=f"Number of latest CSV run log files to analyze "
                             f"(default: {DEFAULT_LOG_COUNT}).")
    parser.add_argument("-d", "--outputs-dir", type=str, default="outputs",
                        help="Directory holding the run folders, a single run folder, or one\n"
                             "CSV file (default: 'outputs').")
    parser.add_argument("-o", "--output-file", type=str,
                        default=os.path.join("outputs", SUMMARY_FILENAME),
                        help="Where to write the formatted report.")
    parser.add_argument("-t", "--tsv-file", type=str,
                        default=os.path.join("outputs", TSV_FILENAME),
                        help="Where to write the stage summary as TSV.")
    parser.add_argument("-j", "--json-file", type=str, default=None,
                        help="Also write the full analysis, per-recording values included,\n"
                             "as JSON for reanalysis elsewhere.")
    parser.add_argument("--compare", type=str, default=None,
                        help="A baseline to test against: a run folder, a CSV, or a directory\n"
                             "of runs. Pairs recording by recording where the names match.")
    parser.add_argument("--compare-count", type=int, default=1,
                        help="How many logs to take from --compare (default: 1).")
    parser.add_argument("--primary", type=str, default=None, metavar="STAGE",
                        help="The stage the comparison is about, e.g. 'ttfa'. Its p-value is\n"
                             "read at full strength instead of paying for the other ten being\n"
                             "tested beside it. Only legitimate when chosen before the result\n"
                             "is seen; picking the winner afterwards is how noise gets published.")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Drop the first N recordings of every run. The first pays for\n"
                             "loading whatever the run loads lazily; 1 is usually right when\n"
                             "the question is steady-state latency (default: 0).")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Confidence level for every interval, and 1 minus the level at\n"
                             f"which the Holm correction holds the family "
                             f"(default: {DEFAULT_CONFIDENCE}).")
    default_percentiles = ",".join(str(int(q * 100)) for q in DEFAULT_QUANTILES)
    parser.add_argument("--percentiles", type=str, default=default_percentiles,
                        help=f"Which percentiles to report (default: '{default_percentiles}').\n"
                             f"99 needs 368 recordings before an interval for it exists.")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_RESAMPLES,
                        help=f"Resamples behind the mean's interval; 0 turns it off "
                             f"(default: {DEFAULT_RESAMPLES}).")
    parser.add_argument("--seed", type=int, default=rstat.DEFAULT_SEED,
                        help="Seed for the bootstrap, so the report is reproducible.")

    args = parser.parse_args()

    try:
        check_options(args.log_count, args.confidence, args.warmup)
    except ValueError as e:
        print(f"Error: {e}.", file=sys.stderr)
        sys.exit(1)

    try:
        quantiles = parse_quantiles(args.percentiles)
    except (ValueError, argparse.ArgumentTypeError) as e:
        print(f"Error: could not read --percentiles: {e}", file=sys.stderr)
        sys.exit(1)

    outputs_dir = Path(args.outputs_dir)
    csv_files = resolve_logs(outputs_dir, args.log_count)
    if not csv_files:
        print(f"No log files found in '{outputs_dir}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing the last {len(csv_files)} CSV log file(s) from '{outputs_dir}'...")

    analysis = analyze_logs(csv_files, log_count=args.log_count, warmup=args.warmup,
                            confidence=args.confidence, quantiles=quantiles,
                            resamples=args.bootstrap, seed=args.seed,
                            compare=args.compare, compare_count=args.compare_count,
                            primary=args.primary)
    if not analysis:
        print("Every selected log was empty or unreadable. Exiting.", file=sys.stderr)
        sys.exit(1)

    for warning in analysis.warnings:
        print(warning if warning.startswith(" ") else f"WARNING: {warning}", file=sys.stderr)

    report = render_report(analysis)
    print("\n" + report.text + "\n")

    written = write_report(report, args.output_file, args.tsv_file, args.json_file)
    labels = {"text": "Summary text table saved to",
              "tsv": "Tab-separated TSV table saved to",
              "json": "JSON analysis saved to"}
    for kind, path in written.items():
        print(f"{labels[kind]} : {path.resolve()}")


if __name__ == "__main__":
    main()
