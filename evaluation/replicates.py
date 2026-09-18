#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch-level analysis of a configuration cell that was run more than once.

Reviewers of the previous campaign correctly treated items inside one launch as
input-level variation, not as independent system replicates. GPU cache, process
state and thermal load are launch-level. This module is what makes those
replicates usable:

    1. Per-cell, per-launch summaries of the metrics that will be claimed.
    2. Between-launch range of those summaries: the host-level uncertainty.
    3. ICC(2,1) of item-level scores across launches of the same cell
       (Shrout and Fleiss, 1979; reporting bands from Koo and Li, 2016).
       High ICC at T=0 means the greedy decoder is stable enough that pooling
       items across launches does not mix distinct response distributions.
    4. Sign agreement of launch-level deltas for a contrast, which is the
       claim-bearing check at k=3: a difference that flips sign between
       interleaved rounds is not host-stable, however small its item-level p.

A Wilcoxon test on three launch medians has no useful power and is not
reported as a decision. The launch-level unit is described, not over-tested.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .agreement import icc_two_way, interpret_icc
from .batch import RunIdentity, unique_cells
from .pipeline import EvaluationOutcome
from .stats import sign_agreement

# Metrics whose launch-level summaries carry the paper's claims. Each one is
# already produced by the item-level evaluation; this module only aggregates
# them at the launch, which is the experimental unit the protocol requires.
#
# Status of the aggregation itself: verifiable (mean/median of recorded values).
# Status of the ICC: validated (Shrout and Fleiss, 1979; Koo and Li, 2016).
LAUNCH_METRICS: List[Tuple[str, str, Callable[[Any], Optional[float]]]] = [
    ("adherence_item_strict", "mean",
     lambda r: (float(r.constraints.item_level_strict)
                if r.constraints is not None
                and r.constraints.item_level_strict is not None else None)),
    ("request_coverage", "mean",
     lambda r: r.relevance.get("request_coverage")),
    ("intent_coverage", "mean",
     lambda r: r.relevance.get("intent_coverage")),
    ("answer_presence", "mean",
     lambda r: r.relevance.get("answer_presence")),
    ("lat_ttfa", "median",
     lambda r: r.latency.get("lat_ttfa")),
    ("lat_llm_ttft", "median",
     lambda r: r.latency.get("lat_llm_ttft")),
    ("llm_tokens_per_sec", "median",
     lambda r: r.latency.get("llm_tokens_per_sec")),
]


def analyse_replicates(runs: Sequence[RunIdentity],
                       outcomes: Dict[str, EvaluationOutcome]
                       ) -> Tuple[List[Dict[str, Any]], str]:
    """Summarise every cell that was launched more than once.

    Returns a table of launch-level rows and a text block for the batch report.
    Cells launched once contribute nothing: a single launch has no replicate
    variance to report, which is itself the limitation the previous campaign
    acknowledged.
    """
    cells = unique_cells(runs)
    rows: List[Dict[str, Any]] = []
    replicated = []
    for cell in cells:
        members = [run for run in runs if run.cell_key == cell.cell_key]
        if len(members) < 2:
            continue
        replicated.append((cell, members))
        for member in sorted(members, key=lambda r: r.launch_id):
            rows.append(_launch_row(member, outcomes[member.key], len(members)))
        rows.extend(_cell_icc_rows(cell, members, outcomes))

    return rows, render_replicate_report(replicated, outcomes)


def _launch_row(run: RunIdentity, outcome: EvaluationOutcome,
                n_launches: int) -> Dict[str, Any]:
    summary = outcome.summary
    stages = {entry["stage"]: entry for entry in summary.latency_stages}
    row: Dict[str, Any] = {
        "cell": run.cell,
        "launch_id": run.launch_id,
        "n_launches_of_cell": n_launches,
        "model_tag": run.model_tag,
        "setting": run.setting,
        "recognizer": run.recognizer,
        "n_items": summary.n_items,
        "adherence_item_strict": summary.constraint_item_rate_strict,
        "request_coverage": _metric_mean(outcome, "request_coverage"),
        "intent_coverage": _metric_mean(outcome, "intent_coverage"),
        "answer_presence": _metric_mean(outcome, "answer_presence"),
        "ttfa_median_ms": (stages.get("ttfa") or {}).get("median_ms"),
        "ttft_median_ms": (stages.get("llm_ttft") or {}).get("median_ms"),
        "tokens_per_sec_median": (stages.get("llm_eval") or {}).get("median_ms"),
    }
    # Throughput is not a latency stage name in every run; fall back to items.
    if row["tokens_per_sec_median"] is None:
        row["tokens_per_sec_median"] = _metric_median(outcome, "llm_tokens_per_sec")
    return row


def _metric_mean(outcome: EvaluationOutcome, key: str) -> Optional[float]:
    values = []
    for result in outcome.results:
        value = result.relevance.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _metric_median(outcome: EvaluationOutcome, key: str) -> Optional[float]:
    values = []
    for result in outcome.results:
        value = result.latency.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    values.sort()
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _cell_icc_rows(cell: RunIdentity, members: Sequence[RunIdentity],
                   outcomes: Dict[str, EvaluationOutcome]) -> List[Dict[str, Any]]:
    """One ICC row per metric, treating launches as raters of each item."""
    rows = []
    for name, _, getter in LAUNCH_METRICS:
        table: Dict[str, Dict[str, float]] = defaultdict(dict)
        for member in members:
            launch = member.launch_id or member.timestamp
            for result in outcomes[member.key].results:
                value = getter(result)
                if isinstance(value, (int, float)):
                    table[result.item_id][launch] = float(value)
        icc = icc_two_way(table)
        if icc is None:
            continue
        rows.append({
            "cell": cell.cell,
            "launch_id": "ICC(2,1)",
            "n_launches_of_cell": len(members),
            "model_tag": cell.model_tag,
            "setting": cell.setting,
            "recognizer": cell.recognizer,
            "metric": name,
            "icc_2_1": icc.icc_single,
            "icc_2_k": icc.icc_average,
            "icc_n_items": icc.n_items,
            "icc_n_raters": icc.n_raters,
            "icc_dropped": icc.dropped_items,
            "icc_label": interpret_icc(icc.icc_single),
        })
    return rows


def render_replicate_report(
        replicated: Sequence[Tuple[RunIdentity, List[RunIdentity]]],
        outcomes: Dict[str, EvaluationOutcome]) -> str:
    """Readable block: which cells were repeated, and how far launches moved."""
    if not replicated:
        return ("No configuration cell was launched more than once. Item-level "
                "intervals therefore describe input variation within a single "
                "host session, not launch-to-launch uncertainty.")

    lines = [
        "Independent interleaved launches are the experimental unit for a",
        "configuration contrast (reviewer 2). Items inside one launch show how",
        "the input set varies; they are not system replicates. A claim that a",
        "model is faster or more adherent is host-stable only when every launch",
        "of that cell moves the same way.",
        "",
        f"Cells with more than one launch: {len(replicated)}",
        "",
        f"{'Cell':<52} {'k':>3} {'adherence range':>18} {'TTFA median range':>20}",
        "-" * 96,
    ]
    for cell, members in replicated:
        adherences = []
        ttfas = []
        for member in members:
            outcome = outcomes[member.key]
            if outcome.summary.constraint_item_rate_strict is not None:
                adherences.append(outcome.summary.constraint_item_rate_strict)
            stages = {entry["stage"]: entry
                      for entry in outcome.summary.latency_stages}
            median = (stages.get("ttfa") or {}).get("median_ms")
            if isinstance(median, (int, float)):
                ttfas.append(float(median))
        adh = (_span_pct(adherences) if adherences else "-")
        ttfa = (_span_ms(ttfas) if ttfas else "-")
        lines.append(f"{cell.cell:<52} {len(members):>3} {adh:>18} {ttfa:>20}")

    lines += [
        "",
        "ICC(2,1) of item-level scores across launches of the same cell is in",
        "launch_replicates.csv (Shrout and Fleiss, 1979). Bands follow Koo and",
        "Li (2016): <0.50 poor, 0.50-0.75 moderate, 0.75-0.90 good, >0.90",
        "excellent. A poor ICC at T=0 means two greedy launches of the same",
        "cell still produce different replies, so an adherence gap of a few",
        "percentage points between models cannot be attributed to the model",
        "until it exceeds that launch-level range.",
        "",
        "Method status: ICC validated; launch-range descriptive; sign",
        "agreement of deltas is a qualitative replication check, not a test.",
    ]
    return "\n".join(lines)


def _span_pct(values: Sequence[float]) -> str:
    lo, hi = min(values), max(values)
    return f"{100 * lo:.1f}-{100 * hi:.1f}%"


def _span_ms(values: Sequence[float]) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.0f}-{hi:.0f} ms"


def contrast_sign_agreement(per_launch_deltas: Sequence[float]) -> Dict[str, Any]:
    """Whether independent launches of one contrast agree on the direction.

    Status: descriptive. With k=3 a sign test has no useful power; requiring
    3/3 agreement is a predeclared qualitative gate, not a p-value.
    """
    agreement = sign_agreement(per_launch_deltas)
    nonzero = [v for v in per_launch_deltas
               if isinstance(v, (int, float)) and v != 0]
    return {
        "n_launches": len(per_launch_deltas),
        "n_nonzero": len(nonzero),
        "sign_agreement": agreement,
        "host_stable": agreement == 1.0 if agreement is not None else False,
        "method": "sign agreement across independent launches (descriptive)",
        "status": "descriptive",
    }
