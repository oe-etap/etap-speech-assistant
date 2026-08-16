#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-item timing and resource cost, read from the latency log a run already wrote.

Response quality and response time are not separable questions for a spoken
assistant: a model that answers better after four seconds of silence has not
improved the interaction. The pipeline logs every stage of every item, so the two
can be compared on the same items rather than in separate tables, which is what
this module exists to enable.

Three decisions are worth stating.

Timings are joined to responses by recording, not by position. The log keys items
by the stem of the audio file, the transcript keys them by filename; both are
stable under a reordered or partially failed run, whereas a positional join
silently misaligns everything after the first missing item.

The first item of a run is excluded when the run's own aggregate excluded it. The
first recording pays for cold weights, an empty page cache and an uncompiled
kernel, and pooling that with the rest describes neither. The convention is read
from the run's log_averages.json rather than chosen here, so that these figures
and the run's own summary refer to the same items.

Repeated measurements of one stage within one item are reduced by their median,
following aggregate_logs.py. An item that contained several utterances would
otherwise weigh more than a single-utterance item merely for being longer.

Stage names come from the pipeline and are used verbatim, prefixed with `lat_` in
the metric namespace so that a timing column cannot be confused with a score.
"""

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Sequence

# Timing stages the pipeline records, in the order a request passes through them.
# Listed explicitly so that a stage added to the pipeline appears here as a
# deliberate change rather than as an unexplained new column.
STAGES = [
    "stt",
    "stt_endpoint_delay",
    "llm_prompt_eval",
    "llm_ttft",
    "llm_first_chunk_fill",
    "llm_ttfc",
    "tts_first_chunk",
    "ttfa",
    "llm_eval",
    "tts_total",
    "e2e_response_ready",
]

# Prefix for stage durations in the metric namespace.
STAGE_PREFIX = "lat_"

# Numeric fields carried in the extra_json column, mapped to the metric names
# they are exposed under. These are measurements the engine reports about itself,
# so they are read rather than estimated.
EXTRA_FIELDS: Dict[str, Dict[str, str]] = {
    "llm_prompt_eval": {"prompt_tokens": "llm_prompt_tokens"},
    "llm_eval": {"eval_tokens": "llm_eval_tokens",
                 "tokens_per_sec": "llm_tokens_per_sec"},
    "e2e_response_ready": {"output_duration_ms": "tts_audio_ms",
                           "llm_chunk_count": "llm_chunk_count"},
    "stt": {"stt_rtf": "stt_rtf",
            "input_duration_ms": "input_duration_ms",
            "trailing_silence_ms": "trailing_silence_ms"},
}

# Resource columns summarized at run level. They are near-constant within a run,
# so an item-by-item comparison of them would test nothing; the run-level median
# is what states the memory cost of a model variant.
RESOURCE_COLUMNS = ["cpu_percent", "ram_percent", "rss_mb", "llm_rss_mb",
                    "llm_vram_mb", "llm_model_vram_mb", "gpu_util_percent",
                    "gpu_mem_used_mb"]


@dataclass
class RunLatency:
    """Timings and resource use recovered from one run's latency log."""

    path: Optional[Path] = None
    items: List[str] = field(default_factory=list)
    warmup_items: List[str] = field(default_factory=list)
    per_item: Dict[str, Dict[str, float]] = field(default_factory=dict)
    resource_medians: Dict[str, float] = field(default_factory=dict)
    hardware: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.per_item)

    def for_item(self, key: Optional[str]) -> Dict[str, float]:
        """Timings for one item key, or an empty mapping when it was excluded."""
        if not key:
            return {}
        return self.per_item.get(key, {})

    def stage_medians(self) -> Dict[str, float]:
        """Median of every timing metric over the items that were measured."""
        collected: Dict[str, List[float]] = {}
        for measurements in self.per_item.values():
            for name, value in measurements.items():
                collected.setdefault(name, []).append(value)
        return {name: statistics.median(values)
                for name, values in sorted(collected.items()) if values}


def item_key(name: Optional[str]) -> str:
    """Normalize a recording name to the key the latency log uses.

    The log writes the stem ("00004"), the transcript the filename
    ("00004.wav"). Both reduce to the same key so the two files can be joined.
    """
    if not name:
        return ""
    return Path(str(name).strip()).stem


def discover_latency_log(run_dir: Path) -> Optional[Path]:
    """Find the latency log in a run directory, preferring the newest.

    A rerun in the same directory leaves several logs behind; the newest one
    belongs to the transcripts that sit next to it.
    """
    candidates = sorted(Path(run_dir).glob("latency_log_*.csv"))
    if not candidates:
        candidates = sorted(Path(run_dir).glob("*latency*.csv"))
    return candidates[-1] if candidates else None


def warmup_count(run_dir: Path) -> int:
    """How many leading items the run's own aggregate excluded as warm-up.

    Read from log_averages.json so that these timings and the run's published
    summary describe the same set of items. Absent that file, nothing is
    excluded: dropping items on an assumption would change every median here
    relative to the artefacts the run actually shipped.
    """
    path = Path(run_dir) / "log_averages.json"
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return 0
    value = payload.get("warmup_dropped_per_run")
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0


def load_run_latency(run_dir: Path,
                     warmup_items: Optional[int] = None) -> RunLatency:
    """Load per-item timings for one run directory.

    Returns an empty result rather than raising when no log is present, so that a
    transcript-only run still evaluates on every other tier.
    """
    run_dir = Path(run_dir)
    path = discover_latency_log(run_dir)
    if path is None:
        return RunLatency(warnings=[f"no latency log in {run_dir}"])

    dropped = warmup_count(run_dir) if warmup_items is None else warmup_items
    return load_latency_csv(path, warmup_items=dropped)


def load_latency_csv(path: Path, warmup_items: int = 0) -> RunLatency:
    """Parse one latency log into per-item timings."""
    path = Path(path)
    result = RunLatency(path=path)

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        result.warnings.append(f"could not read {path}: {exc}")
        return result

    ordered: List[str] = []
    for row in rows:
        key = (row.get("item") or "").strip()
        if key and key not in ordered:
            ordered.append(key)

    result.warmup_items = ordered[:warmup_items] if warmup_items > 0 else []
    excluded = set(result.warmup_items)
    result.items = [key for key in ordered if key not in excluded]

    samples: Dict[str, Dict[str, List[float]]] = {}
    resources: Dict[str, List[float]] = {}

    for row in rows:
        key = (row.get("item") or "").strip()
        stage = (row.get("stage") or "").strip()
        if not key or key in excluded:
            continue

        for column in RESOURCE_COLUMNS:
            value = _as_float(row.get(column))
            if value is not None:
                resources.setdefault(column, []).append(value)
        for column in ("gpu_name", "mode", "stt_engine", "tts_engine"):
            value = (row.get(column) or "").strip()
            if value:
                result.hardware.setdefault(column, value)

        if stage not in STAGES:
            continue

        duration = _as_float(row.get("duration_ms"))
        if duration is not None:
            samples.setdefault(key, {}).setdefault(
                f"{STAGE_PREFIX}{stage}", []).append(duration)

        for source, target in EXTRA_FIELDS.get(stage, {}).items():
            value = _as_float(_extra(row).get(source))
            if value is not None:
                samples.setdefault(key, {}).setdefault(target, []).append(value)

    result.per_item = {
        key: {name: statistics.median(values)
              for name, values in sorted(measurements.items()) if values}
        for key, measurements in samples.items()}
    result.resource_medians = {name: statistics.median(values)
                              for name, values in sorted(resources.items())
                              if values}
    return result


def _extra(row: Dict[str, str]) -> Dict[str, Any]:
    """Parse the extra_json column, tolerating an absent or malformed value."""
    raw = (row.get("extra_json") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if number != number else number


def stage_label(metric_key: str) -> str:
    """Human-readable label for a timing metric key."""
    if metric_key.startswith(STAGE_PREFIX):
        return metric_key[len(STAGE_PREFIX):].replace("_", " ") + " (ms)"
    return metric_key.replace("_", " ")


def summarize_stages(latency: RunLatency,
                     stages: Sequence[str] = STAGES) -> List[Dict[str, Any]]:
    """Median, quartiles and tail of each stage, for the run report.

    The 95th percentile is reported next to the median because a spoken assistant
    is judged by the answers that arrive late, not by the typical one (Dean and
    Barroso, 2013).
    """
    rows: List[Dict[str, Any]] = []
    for stage in stages:
        key = f"{STAGE_PREFIX}{stage}"
        values = sorted(measurements[key]
                        for measurements in latency.per_item.values()
                        if key in measurements)
        if not values:
            continue
        rows.append({
            "stage": stage,
            "n": len(values),
            "median_ms": _percentile(values, 50),
            "q1_ms": _percentile(values, 25),
            "q3_ms": _percentile(values, 75),
            "p90_ms": _percentile(values, 90),
            "p95_ms": _percentile(values, 95),
            "max_ms": values[-1],
        })
    return rows


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Order statistic at `percentile`, interpolated between neighbours.

    Computed here rather than with numpy so that a timing table does not depend
    on an array library being importable.
    """
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight)
                 + sorted_values[upper] * weight)


REFERENCE_KEYS = ["tail_at_scale", "paradise"]
