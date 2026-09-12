#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct per-item TTFA from a frozen ASR pass and a text-mode arm.

    TTFA(item) = stt_endpoint_delay(item)   # the canonical --asr-only run
               + llm_ttfc(item)             # one --input-mode text launch
               + tts_first_chunk(item)      # the same text-mode launch
               + QUEUE_HANDOFF_MS

Once the recognizer is out of the per-cell loop no single run measures TTFA
end to end, so it has to be put back together from the two arms. The sum is
an empirical property of the instrumentation, not a definition: `ttfa` is
measured directly as `tts_first_chunk_t - speech_end_t`. What licenses the
sum is that the decomposition holds within a run to a median +5 ms over 13090
archived items -- `ttfa_residual_check.py` is that check, and this script is
its consumer, not a second copy of it.

**Sums per item, never median to median.** The median of a sum equals the sum
of medians only if the stages are perfectly rank-correlated, and they are not:
a slow endpoint and a slow first token happen to different recordings. Every
statistic below is taken over per-item reconstructed values that were summed
first. `tests/test_ttfa_reconstruct.py` pins this with deliberately
anti-correlated stages where the two routes give 1042 ms and 86 ms.

**One ASR run is frozen as canonical.** The text arm ran on one particular
set of transcripts, so the endpoint delay paired with each item has to come
from the run that produced that item's text -- a delay from another pass
belongs to a recognition that produced different words. Other runs in
`--asr-arm` are read only for the cross-run spread figure, which is reported
beside the reconstruction and never added into it.

Standard library only, for the reason `run_statistics.py` gives for itself:
this runs over archived logs on whatever interpreter is at hand. The three
sibling modules it imports are stdlib-only too, and importing them is
deliberate -- `ttfa_residual_check.py` already re-derives `aggregate_logs.py`'s
reader once, and a third copy of the median collapse or of the warm-up
exclusion is exactly the drift those two were written to avoid.

Usage:
    # One ASR pass, one text-mode campaign tree.
    python3 ttfa_reconstruct.py --asr-arm outputs/asr --text-arm outputs/text

    # Several ASR passes archived together: name the one the transcripts
    # came from, and the rest become the spread figure.
    python3 ttfa_reconstruct.py --asr-arm outputs/asr --canonical 20260912_051500 \\
        --text-arm outputs/text --per-item-csv reconstructed.csv

    # Section 3's validation: the same items also run as a full pipeline.
    python3 ttfa_reconstruct.py --asr-arm outputs/asr --text-arm outputs/text \\
        --measured outputs/file --comparison-csv measured_vs_reconstructed.csv
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import campaign_report as creport
import run_statistics as rstat
import ttfa_residual_check as residual

# The reconstruction is biased low by a constant without this. Over 13090
# archived items across 147 distinct runs the within-run residual
# `ttfa - (stt_endpoint_delay + llm_ttfc + tts_first_chunk)` sits in a 3-6 ms
# band with a median of +5 and a p95 of +6 -- a constant, not a spread around
# zero. It is the queue handoff on either side of the LLM request: `mailbox`
# between the STT finalising and `llm_worker` picking it up, and `tts_queue`
# between the first chunk being ready and `tts_worker` writing it. Neither
# interval is instrumented, so neither lands in any of the three terms.
# Carried rather than ignored because leaving it out is a known-sign error
# that costs nothing to remove, and because a named constant is the only
# form in which it stays visible: the report prints it, `--offset-ms 0` drops
# it, and the per-item CSV carries the raw sum in its own column so the
# offset is one subtraction away from the data. Re-measure it on campaign
# data: these runs predate the Ollama 0.33.3 upgrade of 2026-09-09, whose
# 0.15-1.8 s per-request overhead sat inside both sides of that subtraction.
QUEUE_HANDOFF_MS = 5.0

ASR_STAGE = "stt_endpoint_delay"
TEXT_STAGES = ("llm_ttfc", "tts_first_chunk")
RECONSTRUCTION_STAGES = (ASR_STAGE,) + TEXT_STAGES
MEASURED_STAGE = "ttfa"

# What section 3 compares arm by arm. `llm_ttft` is in here and not in the
# sum: it is the first *token*, which `llm_ttfc` (first chunk) already
# contains, so adding it would double-count -- but it is the stage that says
# whether a difference between the arms came from the model or from
# everything after it.
COMPARISON_STAGES = ("llm_ttft", "llm_ttfc", "tts_first_chunk")

QUANTILES = (0.5, 0.9, 0.95)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_UNMATCHED = 2


# ---------- Reading arms ----------
@dataclass
class Arm:
    """Every launch of one run mode, and what was dropped reading them."""
    label: str
    root: Path
    runs: List[creport.LatencyRun] = field(default_factory=list)
    duplicate_csvs: List[Path] = field(default_factory=list)

    @property
    def cells(self) -> List[str]:
        return sorted({run.cell for run in self.runs})

    def by_cell(self, cell: str) -> List[creport.LatencyRun]:
        return [run for run in self.runs if run.cell == cell]


def load_arm(label: str, root: Path) -> Arm:
    """Read one arm's launches, counting a run archived at two paths once.

    Discovery is `ttfa_residual_check.find_csvs` (byte-identical copies
    dropped -- 28 of the 175 files under the archive are one) and parsing is
    `campaign_report.read_latency_csv` (repeated `(stage, item)` rows
    median-collapsed, cell/launch identity from the columns with the path
    fallback). Neither is reimplemented here.
    """
    paths, duplicates = residual.find_csvs(root)
    runs = []
    for path in paths:
        run = creport.read_latency_csv(path)
        if run is not None:
            runs.append(run)
    return Arm(label=label, root=root, runs=runs, duplicate_csvs=duplicates)


def timed_items(run: creport.LatencyRun) -> List[str]:
    """A launch's items with its warm-up prefix removed.

    `campaign_report.WARMUP_ITEMS`, not a local 1: the first item of every
    launch is bimodal (136 of 144 archived first items are single-digit
    milliseconds, 8 are 3.5-50 s of model load landing inside `ttfa` and
    inside none of the three terms), and `aggregate_logs.py`,
    `ttfa_residual_check.py` and `campaign_report.py` all already exclude it.
    A fourth definition of the exclusion is a fourth thing to drift.
    """
    return run.ordered_items[creport.WARMUP_ITEMS:]


def stage_value(run: creport.LatencyRun, stage: str, item: str) -> Optional[float]:
    return run.durations.get(stage, {}).get(item)


def item_series(run: creport.LatencyRun, stage: str) -> Dict[str, float]:
    """One launch's timed items that carry a value for this stage.

    An item with no row for the stage is absent from the result rather than
    zero, the same distinction `aggregate_logs.py` needs the pipeline to
    write: a zero here cannot be told from a measurement later.
    """
    series = {}
    for item in timed_items(run):
        value = stage_value(run, stage, item)
        if value is not None:
            series[item] = value
    return series


def pick_canonical(arm: Arm, wanted: Optional[str]) -> Tuple[Optional[creport.LatencyRun], str]:
    """The one ASR run the transcripts came from, or a refusal explaining why.

    A silent pick here would pair each item's endpoint delay with a
    recognition that may have produced different text, which is the one
    correspondence the whole reconstruction rests on -- so more than one
    candidate and no `--canonical` is an error, not a default.
    """
    if not arm.runs:
        return None, f"no latency CSV under {arm.root}"
    if wanted is None:
        if len(arm.runs) == 1:
            return arm.runs[0], ""
        names = ", ".join(f"{run.cell}/{run.launch}" for run in arm.runs)
        return None, (f"{len(arm.runs)} ASR runs under {arm.root}; name the one the "
                      f"transcripts came from with --canonical (candidates: {names})")
    matches = [run for run in arm.runs
               if wanted in (run.launch, f"{run.cell}/{run.launch}") or wanted in str(run.path)]
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        names = ", ".join(f"{run.cell}/{run.launch}" for run in arm.runs)
        return None, f"--canonical {wanted!r} matches no run under {arm.root} (have: {names})"
    names = ", ".join(str(run.path) for run in matches)
    return None, f"--canonical {wanted!r} matches {len(matches)} runs: {names}"


# ---------- The reconstruction ----------
@dataclass
class Join:
    """Which items the two arms agreed on, and which they did not."""
    matched: List[str] = field(default_factory=list)
    asr_only: List[str] = field(default_factory=list)
    text_only: List[str] = field(default_factory=list)
    incomplete: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not (self.asr_only or self.text_only or self.incomplete)


@dataclass
class Reconstruction:
    """One text-mode launch joined to the canonical ASR run, item by item."""
    cell: str
    launch: str
    path: Path
    offset_ms: float
    stage_sum: Dict[str, float] = field(default_factory=dict)
    parts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    join: Join = field(default_factory=Join)

    @property
    def values(self) -> Dict[str, float]:
        """Reconstructed TTFA per item, offset applied."""
        return {item: total + self.offset_ms for item, total in self.stage_sum.items()}


def reconstruct(asr_run: creport.LatencyRun, text_run: creport.LatencyRun,
                offset_ms: float) -> Reconstruction:
    """Join one text-mode launch to the frozen ASR run and sum per item.

    Each arm's own warm-up item is dropped before the join, so an arm whose
    first item differs from the other's shows up as an unmatched item on both
    sides rather than as a silently missing row.
    """
    result = Reconstruction(cell=text_run.cell, launch=text_run.launch,
                            path=text_run.path, offset_ms=offset_ms)
    asr_items = timed_items(asr_run)
    text_items = timed_items(text_run)
    asr_set, text_set = set(asr_items), set(text_items)

    result.join.asr_only = [item for item in asr_items if item not in text_set]
    result.join.text_only = [item for item in text_items if item not in asr_set]

    for item in text_items:
        if item not in asr_set:
            continue
        parts: Dict[str, float] = {}
        missing: List[str] = []
        for stage in RECONSTRUCTION_STAGES:
            source = asr_run if stage == ASR_STAGE else text_run
            value = stage_value(source, stage, item)
            if value is None:
                missing.append(stage)
            else:
                parts[stage] = value
        if missing:
            result.join.incomplete[item] = missing
            continue
        result.join.matched.append(item)
        result.parts[item] = parts
        # Per item, and only then a statistic. Summing the stage medians
        # instead would give the right answer only under perfect rank
        # correlation between the stages, which is not what the arms do.
        result.stage_sum[item] = sum(parts[stage] for stage in RECONSTRUCTION_STAGES)

    return result


# ---------- Statistics ----------
def describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """n, quantiles and range for one sample.

    Quantiles come from `rstat.percentile` and nothing else: it interpolates
    linearly between order statistics, while a truncated index
    (`sorted[int(n*q)]`) is a different estimator that disagrees on short
    heavy-tailed samples -- 12811 ms against 7669 ms on one real stratum
    here. Two conventions in one project make two scripts' figures silently
    incomparable.
    """
    if not values:
        return {"n": 0}
    described: Dict[str, Optional[float]] = {"n": len(values)}
    for q in QUANTILES:
        described[f"p{int(q * 100)}"] = rstat.percentile(values, q)
    described["min"] = min(values)
    described["max"] = max(values)
    return described


def per_item_median(series: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """One value per item, collapsing replicate launches of the same item.

    This is a median over launches of a value that was already summed per
    item -- the launch is the unit of replication, so collapsing it is what
    the analysis layer does everywhere. It is not the median-to-median
    shortcut the module docstring refuses: that one would collapse the
    *stages* before they were added.
    """
    gathered: Dict[str, List[float]] = {}
    for mapping in series:
        for item, value in mapping.items():
            gathered.setdefault(item, []).append(value)
    return {item: rstat.percentile(values, 0.5) for item, values in gathered.items()}


def cell_rollup(launch_medians: Sequence[float]) -> Dict[str, Optional[float]]:
    """A cell's centre and the range its launches actually spanned."""
    if not launch_medians:
        return {"launches": 0}
    return {"launches": len(launch_medians),
            "median": rstat.percentile(launch_medians, 0.5),
            "min": min(launch_medians),
            "max": max(launch_medians)}


# ---------- Cross-run endpoint-delay spread ----------
@dataclass
class EndpointSpread:
    """How far apart repeated ASR passes put the same item's endpoint delay."""
    run_medians: List[Tuple[str, float]] = field(default_factory=list)
    per_item_spread: List[float] = field(default_factory=list)
    items_on_every_run: int = 0
    runs: int = 0


def endpoint_spread(runs: Sequence[creport.LatencyRun]) -> EndpointSpread:
    """Descriptive only: never added to a reconstructed value.

    Propagating it would mean pairing an item with an endpoint delay from a
    recognition that produced different text. Reported because it is the
    size of the error the freeze avoids, which is worth knowing even though
    it is deliberately not carried.
    """
    spread = EndpointSpread(runs=len(runs))
    gathered: Dict[str, List[float]] = {}
    for run in runs:
        series = item_series(run, ASR_STAGE)
        if series:
            spread.run_medians.append((f"{run.cell}/{run.launch}",
                                       rstat.percentile(list(series.values()), 0.5)))
        for item, value in series.items():
            gathered.setdefault(item, []).append(value)

    for values in gathered.values():
        if len(values) == len(runs) and len(values) > 1:
            spread.items_on_every_run += 1
            spread.per_item_spread.append(max(values) - min(values))
    return spread


# ---------- Section 3: measured against reconstructed ----------
@dataclass
class CellComparison:
    """One cell's validation rows: the same items run both ways."""
    cell: str
    text_launches: List[str] = field(default_factory=list)
    measured_launches: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    reconstructed: Dict[str, float] = field(default_factory=dict)
    measured: Dict[str, float] = field(default_factory=dict)
    stage_text: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stage_measured: Dict[str, Dict[str, float]] = field(default_factory=dict)
    only_reconstructed: List[str] = field(default_factory=list)
    only_measured: List[str] = field(default_factory=list)

    @property
    def differences(self) -> List[float]:
        return [self.measured[item] - self.reconstructed[item] for item in self.items]


def stage_series(runs: Sequence[creport.LatencyRun], stage: str) -> Dict[str, float]:
    """One value per item for a stage, over a cell's launches."""
    return per_item_median([item_series(run, stage) for run in runs])


def compare_cell(cell: str, reconstructions: Sequence[Reconstruction],
                 text_runs: Sequence[creport.LatencyRun],
                 measured_runs: Sequence[creport.LatencyRun]) -> CellComparison:
    """Measured TTFA against reconstructed TTFA for one cell, item by item.

    What this difference contains is *cross-launch* variance -- the two arms
    are separate launches of a CPU-bound model -- plus whatever overlap the
    reconstruction misses, namely the recognizer decoding trailing silence
    while the LLM is already producing tokens. It is a different quantity
    from the within-run additivity residual (+5 ms, `ttfa_residual_check.py`),
    which compares terms recorded inside one launch. Three recordings
    measured this way came out at -187/-160/-113 ms; that is the size of the
    launch-to-launch term, not a broken decomposition.
    """
    comparison = CellComparison(cell=cell)
    comparison.text_launches = sorted({r.launch for r in reconstructions})
    comparison.measured_launches = sorted({r.launch for r in measured_runs})

    comparison.reconstructed = per_item_median([r.values for r in reconstructions])
    comparison.measured = stage_series(measured_runs, MEASURED_STAGE)
    for stage in COMPARISON_STAGES:
        comparison.stage_text[stage] = stage_series(text_runs, stage)
        comparison.stage_measured[stage] = stage_series(measured_runs, stage)

    both = set(comparison.reconstructed) & set(comparison.measured)
    comparison.items = sorted(both)
    comparison.only_reconstructed = sorted(set(comparison.reconstructed) - both)
    comparison.only_measured = sorted(set(comparison.measured) - both)
    return comparison


# ---------- Output files ----------
PER_ITEM_FIELDS = ["cell_id", "launch_id", "item", "asr_launch_id",
                   "stt_endpoint_delay_ms", "llm_ttfc_ms", "tts_first_chunk_ms",
                   "stage_sum_ms", "offset_ms", "reconstructed_ttfa_ms"]


def write_per_item_csv(path: Path, asr_run: creport.LatencyRun,
                       reconstructions: Sequence[Reconstruction]) -> int:
    """One row per (cell, launch, item): the three terms and their sum.

    `stage_sum_ms` and `offset_ms` are separate columns so the offset can be
    taken back out of the data without re-running anything.
    """
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_ITEM_FIELDS)
        writer.writeheader()
        for rec in reconstructions:
            for item in rec.join.matched:
                parts = rec.parts[item]
                writer.writerow({
                    "cell_id": rec.cell,
                    "launch_id": rec.launch,
                    "item": item,
                    "asr_launch_id": asr_run.launch,
                    "stt_endpoint_delay_ms": round(parts[ASR_STAGE], 3),
                    "llm_ttfc_ms": round(parts["llm_ttfc"], 3),
                    "tts_first_chunk_ms": round(parts["tts_first_chunk"], 3),
                    "stage_sum_ms": round(rec.stage_sum[item], 3),
                    "offset_ms": rec.offset_ms,
                    "reconstructed_ttfa_ms": round(rec.stage_sum[item] + rec.offset_ms, 3),
                })
                rows += 1
    return rows


def comparison_fields() -> List[str]:
    fields = ["cell_id", "item", "n_text_launches", "n_measured_launches"]
    for stage in COMPARISON_STAGES:
        fields += [f"{stage}_text_ms", f"{stage}_measured_ms", f"{stage}_diff_ms"]
    fields += ["reconstructed_ttfa_ms", "measured_ttfa_ms", "ttfa_diff_ms"]
    return fields


def write_comparison_csv(path: Path, comparisons: Sequence[CellComparison]) -> int:
    """Section 3's per-item table: the same item, both ways, stage by stage.

    A cell on only one side of the comparison writes no row at all rather
    than a row of blanks, on the same grounds `aggregate_logs.py` gives for
    stages: a placeholder cannot be told apart from a measurement later.
    """
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields())
        writer.writeheader()
        for comparison in comparisons:
            for item in comparison.items:
                row = {"cell_id": comparison.cell,
                       "item": item,
                       "n_text_launches": len(comparison.text_launches),
                       "n_measured_launches": len(comparison.measured_launches)}
                for stage in COMPARISON_STAGES:
                    text = comparison.stage_text[stage].get(item)
                    measured = comparison.stage_measured[stage].get(item)
                    row[f"{stage}_text_ms"] = "" if text is None else round(text, 3)
                    row[f"{stage}_measured_ms"] = "" if measured is None else round(measured, 3)
                    row[f"{stage}_diff_ms"] = ("" if text is None or measured is None
                                               else round(measured - text, 3))
                row["reconstructed_ttfa_ms"] = round(comparison.reconstructed[item], 3)
                row["measured_ttfa_ms"] = round(comparison.measured[item], 3)
                row["ttfa_diff_ms"] = round(comparison.measured[item]
                                            - comparison.reconstructed[item], 3)
                writer.writerow(row)
                rows += 1
    return rows


# ---------- Reporting ----------
def _fmt(value: Optional[float], digits: int = 0) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _describe_line(label: str, values: Sequence[float], width: int = 34) -> str:
    stats = describe(values)
    if not stats.get("n"):
        return f"{label:<{width}} n=0"
    return (f"{label:<{width}} n={stats['n']:<5} median={_fmt(stats['p50'])} ms"
            f"  p90={_fmt(stats['p90'])} ms  p95={_fmt(stats['p95'])} ms"
            f"  min={_fmt(stats['min'])} ms  max={_fmt(stats['max'])} ms")


def render_report(asr_run: creport.LatencyRun, asr_arm: Arm, text_arm: Arm,
                  reconstructions: Sequence[Reconstruction], offset_ms: float,
                  spread: EndpointSpread,
                  comparisons: Sequence[CellComparison]) -> str:
    lines: List[str] = []
    terms = " + ".join(RECONSTRUCTION_STAGES)
    lines.append(f"Reconstructed TTFA = {terms} + queue handoff")
    if offset_ms:
        lines.append(f"  queue-handoff offset applied: {offset_ms:+.0f} ms "
                     f"(median within-run residual over 13090 archived items; "
                     f"--offset-ms 0 drops it)")
    else:
        lines.append("  queue-handoff offset applied: none "
                     f"(the archived median residual is {QUEUE_HANDOFF_MS:+.0f} ms, "
                     "so these values are biased low by about that much)")

    warmup = ", ".join(asr_run.warmup_items) or "(none)"
    lines += ["",
              f"Canonical ASR run: {asr_run.cell}/{asr_run.launch}  ({asr_run.path})",
              f"  {len(timed_items(asr_run))} timed items, warm-up excluded: {warmup}"]
    if len(asr_arm.runs) > 1:
        lines.append(f"  {len(asr_arm.runs) - 1} further ASR run(s) in the arm are read only "
                     "for the spread figure below, never for a reconstructed value")
    if asr_arm.duplicate_csvs:
        lines.append(f"  {len(asr_arm.duplicate_csvs)} byte-identical copy/copies dropped")

    lines += ["",
              f"Text arm: {len(text_arm.runs)} launch(es) over {len(text_arm.cells)} cell(s)"
              + (f", {len(text_arm.duplicate_csvs)} byte-identical copy/copies dropped"
                 if text_arm.duplicate_csvs else "")]

    lines += ["", "Join (each arm's own warm-up item dropped first)"]
    unmatched_total = 0
    for rec in reconstructions:
        join = rec.join
        unmatched = len(join.asr_only) + len(join.text_only) + len(join.incomplete)
        unmatched_total += unmatched
        lines.append(f"  {rec.cell}/{rec.launch:<20} matched {len(join.matched):<5} "
                     f"unmatched {unmatched}")
        for item in join.asr_only:
            lines.append(f"      only in the ASR run: {item}")
        for item in join.text_only:
            lines.append(f"      only in the text run: {item}")
        for item, stages in sorted(join.incomplete.items()):
            lines.append(f"      {item}: no {', '.join(stages)} value")
    if unmatched_total:
        lines.append("  unmatched items are an item-naming bug in the run modes, not "
                     "something to average over; fix the naming and re-run")

    lines += ["", "Reconstructed TTFA per launch"]
    for rec in reconstructions:
        lines.append("  " + _describe_line(f"{rec.cell}/{rec.launch}",
                                           sorted(rec.values.values())))

    by_cell: Dict[str, List[float]] = {}
    for rec in reconstructions:
        values = list(rec.values.values())
        if values:
            by_cell.setdefault(rec.cell, []).append(rstat.percentile(values, 0.5))
    if by_cell:
        lines += ["", "Reconstructed TTFA per cell (median of launch medians, "
                      "range over launches)"]
        for cell, medians in sorted(by_cell.items()):
            rollup = cell_rollup(medians)
            lines.append(f"  {cell:<34} launches={rollup['launches']:<3} "
                         f"median={_fmt(rollup['median'])} ms  "
                         f"range={_fmt(rollup['min'])}..{_fmt(rollup['max'])} ms")

    lines += ["", "Cross-run endpoint-delay spread (descriptive; NOT propagated above)"]
    if spread.runs < 2:
        lines.append("  only one ASR run in the arm; nothing to compare it against")
    else:
        for name, median in spread.run_medians:
            lines.append(f"  {name:<34} median {ASR_STAGE} = {_fmt(median)} ms")
        lines.append("  " + _describe_line("per-item spread (max-min)",
                                           spread.per_item_spread))
        lines.append(f"  over the {spread.items_on_every_run} item(s) present on all "
                     f"{spread.runs} runs. Freezing one run is what keeps this out of "
                     "the reconstruction.")

    if comparisons:
        lines += ["", "Validation: directly measured TTFA - reconstructed TTFA",
                  "  cross-launch difference between two separate launches, NOT the "
                  "within-run additivity residual"]
        for comparison in comparisons:
            lines.append(f"  cell {comparison.cell}: {len(comparison.text_launches)} text "
                         f"launch(es), {len(comparison.measured_launches)} measured launch(es), "
                         f"{len(comparison.items)} shared item(s)")
            for item in comparison.only_reconstructed:
                lines.append(f"      reconstructed only: {item}")
            for item in comparison.only_measured:
                lines.append(f"      measured only: {item}")
            lines.append("    " + _describe_line("measured - reconstructed",
                                                 comparison.differences, width=32))
            for stage in COMPARISON_STAGES:
                diffs = [comparison.stage_measured[stage][item]
                         - comparison.stage_text[stage][item]
                         for item in comparison.items
                         if item in comparison.stage_text[stage]
                         and item in comparison.stage_measured[stage]]
                lines.append("    " + _describe_line(f"{stage}: measured - text",
                                                     diffs, width=32))

    return "\n".join(lines)


def build_json(asr_run: creport.LatencyRun, reconstructions: Sequence[Reconstruction],
               offset_ms: float, spread: EndpointSpread,
               comparisons: Sequence[CellComparison]) -> Dict[str, object]:
    by_cell: Dict[str, List[float]] = {}
    launches = []
    for rec in reconstructions:
        values = list(rec.values.values())
        launches.append({
            "cell_id": rec.cell,
            "launch_id": rec.launch,
            "path": str(rec.path),
            "matched": len(rec.join.matched),
            "unmatched_asr_only": rec.join.asr_only,
            "unmatched_text_only": rec.join.text_only,
            "incomplete": rec.join.incomplete,
            "reconstructed_ttfa_ms": describe(values),
        })
        if values:
            by_cell.setdefault(rec.cell, []).append(rstat.percentile(values, 0.5))

    return {
        "offset_ms": offset_ms,
        "canonical_asr_run": {"cell_id": asr_run.cell, "launch_id": asr_run.launch,
                              "path": str(asr_run.path),
                              "warmup_items": asr_run.warmup_items},
        "launches": launches,
        "cells": {cell: cell_rollup(medians) for cell, medians in sorted(by_cell.items())},
        "endpoint_delay_spread": {
            "runs": spread.runs,
            "items_on_every_run": spread.items_on_every_run,
            "run_medians_ms": dict(spread.run_medians),
            "per_item_spread_ms": describe(spread.per_item_spread),
        },
        "validation": [
            {"cell_id": c.cell,
             "text_launches": c.text_launches,
             "measured_launches": c.measured_launches,
             "items": len(c.items),
             "measured_minus_reconstructed_ms": describe(c.differences)}
            for c in comparisons
        ],
    }


# ---------- CLI ----------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--asr-arm", required=True,
                        help="Directory tree or single latency CSV of --asr-only runs.")
    parser.add_argument("--canonical", default=None,
                        help="Launch id, cell/launch, or path fragment of the ASR run the "
                             "transcripts came from. Required when the arm holds more "
                             "than one run.")
    parser.add_argument("--text-arm", required=True,
                        help="Directory tree or single latency CSV of --input-mode text runs.")
    parser.add_argument("--measured", default=None,
                        help="Directory tree or single latency CSV of full-pipeline runs "
                             "over the same items, for the validation comparison.")
    parser.add_argument("--offset-ms", type=float, default=QUEUE_HANDOFF_MS,
                        help="Queue-handoff constant added to every reconstructed value "
                             f"(default: {QUEUE_HANDOFF_MS:g}; pass 0 to report the bare sum).")
    parser.add_argument("--per-item-csv", default=None,
                        help="Write the per-item reconstruction table here.")
    parser.add_argument("--comparison-csv", default=None,
                        help="Write the per-item measured-versus-reconstructed table here.")
    parser.add_argument("--json", default=None,
                        help="Write the same figures as JSON here.")
    args = parser.parse_args(argv)

    asr_root, text_root = Path(args.asr_arm), Path(args.text_arm)
    for root in (asr_root, text_root):
        if not root.exists():
            print(f"error: '{root}' does not exist", file=sys.stderr)
            return EXIT_USAGE

    asr_arm = load_arm("asr", asr_root)
    text_arm = load_arm("text", text_root)
    asr_run, problem = pick_canonical(asr_arm, args.canonical)
    if asr_run is None:
        print(f"error: {problem}", file=sys.stderr)
        return EXIT_USAGE
    if not text_arm.runs:
        print(f"error: no latency CSV under {text_root}", file=sys.stderr)
        return EXIT_USAGE

    reconstructions = [reconstruct(asr_run, run, args.offset_ms) for run in text_arm.runs]
    spread = endpoint_spread(asr_arm.runs)

    comparisons: List[CellComparison] = []
    if args.measured:
        measured_root = Path(args.measured)
        if not measured_root.exists():
            print(f"error: '{measured_root}' does not exist", file=sys.stderr)
            return EXIT_USAGE
        measured_arm = load_arm("measured", measured_root)
        if not measured_arm.runs:
            print(f"error: no latency CSV under {measured_root}", file=sys.stderr)
            return EXIT_USAGE
        for cell in text_arm.cells:
            measured_runs = measured_arm.by_cell(cell)
            if not measured_runs:
                print(f"warning: cell {cell!r} has no measured launch to compare against",
                      file=sys.stderr)
                continue
            comparisons.append(compare_cell(
                cell,
                [rec for rec in reconstructions if rec.cell == cell],
                text_arm.by_cell(cell),
                measured_runs))
        for cell in measured_arm.cells:
            if cell not in text_arm.cells:
                print(f"warning: measured cell {cell!r} has no text-arm launch",
                      file=sys.stderr)

    print(render_report(asr_run, asr_arm, text_arm, reconstructions,
                        args.offset_ms, spread, comparisons))

    if args.per_item_csv:
        rows = write_per_item_csv(Path(args.per_item_csv), asr_run, reconstructions)
        print(f"\nwrote {rows} per-item row(s) to {args.per_item_csv}")
    if args.comparison_csv:
        rows = write_comparison_csv(Path(args.comparison_csv), comparisons)
        print(f"wrote {rows} comparison row(s) to {args.comparison_csv}")
    if args.json:
        payload = build_json(asr_run, reconstructions, args.offset_ms, spread, comparisons)
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True),
                                   encoding="utf-8")
        print(f"wrote {args.json}")

    if any(not rec.join.clean for rec in reconstructions):
        print("error: the arms did not join cleanly; see the unmatched items above",
              file=sys.stderr)
        return EXIT_UNMATCHED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
