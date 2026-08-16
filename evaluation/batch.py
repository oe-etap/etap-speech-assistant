#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a tree of runs and compare them, in one pass.

A grid of configurations produces one run directory per cell, and the question
being asked of them is never "how did this run score" but "which setting was
better". This module reads the whole tree, scores every run identically, groups
the runs into the contrasts the grid was built to support, and writes one results
directory that holds every table.

    python -m evaluation.batch --root ..\\results\\text-only

Five contrasts are derived from the recorded configuration rather than from the
folder names, so a mislabelled directory cannot silently produce a wrong
comparison:

    parameters     the same model at different decoding settings
    quantization   the same model and size at different weight precisions
    model_size     one lineage and precision at different parameter counts
    cross_model    every variant against the strongest one, at a fixed setting
    recognizer     the same model and setting behind different recognizers

Every contrast holds the recognizer fixed except the one that varies it, because
a run heard by another recognizer answered a different question: the words the
model received are its input, so mixing recognizers into a model contrast would
report an input change as a model effect.

Each contrast is a paired comparison over the shared recordings, so the item
variation cancels and what remains is the effect of the change. A size ladder is
grouped by lineage rather than by exact family, because the sizes on offer are
split across releases; where a ladder crosses one, the contrast says so instead
of presenting the release change as a size change. The baseline of a
comparison is the reference the change is measured against: the highest precision,
the largest model, the greedy decoding setting, the recognizer the rest of the
grid was measured with. A negative delta therefore reads as "this is what the
cheaper configuration costs".

Every run is evaluated once, however many contrasts it appears in. Beyond saving
work, this guarantees that a run's scores are identical in every table it appears
in, which re-evaluating per comparison would not.
"""

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import references, reporting
from .aggregation import SHORT_ANSWER_MAX_WORDS
from .asr import STRATA, corpus_error_rate
from .comparison import (FAMILIES, ComparisonConfig, ComparisonOutcome,
                         compare_evaluated)
from .loaders import discover_run
from .pipeline import EvaluationConfig, EvaluationOutcome, run_evaluation
from .stats import PAIRED_REFERENCE_KEYS, spearman

PathLike = Union[str, Path]

# The contrasts this module knows how to build, in reporting order.
GROUP_KINDS = ["parameters", "quantization", "model_size", "cross_model",
               "recognizer"]

GROUP_QUESTIONS = {
    "parameters": "Does the decoding setting change the responses of one model?",
    "quantization": "What does compressing the weights of one model cost?",
    "model_size": "What does a smaller model of the same lineage cost?",
    "cross_model": "How does each variant compare with the strongest one?",
    "recognizer": "What does hearing the question through another recognizer change?",
}

# Weight formats, mapped to the bits per weight used to order them. Ordering by
# precision is what makes "the strongest variant" a property of the run rather
# than a choice made per table.
_PRECISION_BITS = {
    "fp32": 32, "f32": 32, "fp16": 16, "f16": 16, "bf16": 16,
    "q8_0": 8, "q8": 8, "q6_k": 6, "q5_k_m": 5, "q5_k_s": 5, "q5_1": 5,
    "q5_0": 5, "q4_k_m": 4, "q4_k_s": 4, "q4_1": 4, "q4_0": 4, "q3_k_m": 3,
    "q3_k_l": 3, "q3_k_s": 3, "q2_k": 2,
}

_QUANTIZATION_RE = re.compile(
    r"^(?:fp?(?:16|32)|bf16|q\d+(?:_[0-9a-z]+)*)$", re.IGNORECASE)

_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([bm])$", re.IGNORECASE)


@dataclass(frozen=True)
class RunIdentity:
    """What distinguishes one run from another, read from its configuration.

    Built from `config_used.yaml` rather than from the directory name: the folder
    name is a convenience for a human reader, while the recorded configuration is
    what the run was actually executed with.
    """

    run_dir: Path
    cell: str
    timestamp: str
    order: int
    model_tag: str
    family: str
    size_label: str
    size_b: float
    tuning: str
    quantization: str
    precision_bits: int
    temperature: Optional[float]
    seed: Optional[int]
    num_ctx: Optional[int]
    max_tokens: Optional[int]
    prompt_file: str
    stt_engine: str
    mode: str

    # The recognizer as it was configured, not merely which engine was named: a
    # different acoustic model or compute type is a different recognizer, and it
    # is the recognized words that reach the language model.
    stt_model: str = ""
    stt_device: str = ""
    stt_compute: str = ""

    @property
    def recognizer(self) -> str:
        """The recognizer's identity, used wherever it must be held fixed."""
        parts = [self.stt_engine or "stt", self.stt_model, self.stt_device,
                 self.stt_compute]
        return "-".join(part for part in parts if part)

    @property
    def variant(self) -> str:
        """The model identity, including its weight precision."""
        return self.model_tag

    @property
    def lineage(self) -> str:
        """The family without its version: llama3.2 and llama3.1 share one.

        A size ladder is usually built across releases, because the sizes on
        offer differ between them -- llama3.2 ships 1b and 3b, llama3.1 ships 8b.
        Grouping by lineage keeps that ladder in one table; the release
        difference is then named in the contrast rather than hidden by it.
        """
        match = re.match(r"^([^\d]+)", self.family)
        return (match.group(1).rstrip("-_.") if match else self.family) or self.family

    @property
    def short_model(self) -> str:
        """Model name compact enough for a table column."""
        parts = [f"{self.family}:{self.size_label}" if self.size_label
                 else self.family]
        if self.quantization:
            parts.append(self.quantization)
        return "-".join(parts)

    @property
    def setting(self) -> str:
        """The decoding setting, as it is written in a table."""
        if self.temperature is None:
            return "t?"
        text = f"t{self.temperature:g}"
        return f"{text}/s{self.seed}" if self.seed is not None else text

    @property
    def label(self) -> str:
        """Identifier used in every table, unique within a batch."""
        return f"{self.order:02d} {self.short_model} {self.setting}"

    @property
    def key(self) -> str:
        return str(self.run_dir)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "cell": self.cell,
            "timestamp": self.timestamp,
            "run_dir": str(self.run_dir),
            "model_tag": self.model_tag,
            "family": self.family,
            "size_label": self.size_label,
            "size_b": self.size_b,
            "tuning": self.tuning,
            "quantization": self.quantization,
            "precision_bits": self.precision_bits,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
            "system_prompt_file": self.prompt_file,
            "stt_engine": self.stt_engine,
            "stt_model": self.stt_model,
            "stt_device": self.stt_device,
            "stt_compute": self.stt_compute,
            "recognizer": self.recognizer,
            "mode": self.mode,
        }


def parse_model_tag(tag: str) -> Dict[str, Any]:
    """Split an Ollama tag into family, size, tuning and weight precision.

    "llama3.1:8b-instruct-q4_K_M" yields family llama3.1, size 8b, tuning
    instruct, quantization q4_K_M. An unrecognised tag keeps its whole text as
    the family, so that a run with an unexpected name is grouped on its own
    rather than merged with something else.
    """
    text = str(tag or "").strip()
    family, _, remainder = text.partition(":")
    parts = [p for p in remainder.split("-") if p] if remainder else []

    size_label = ""
    size_b = 0.0
    if parts:
        match = _SIZE_RE.match(parts[0])
        if match:
            size_label = parts.pop(0).lower()
            value = float(match.group(1))
            size_b = value / 1000.0 if match.group(2).lower() == "m" else value

    quantization = ""
    if parts and _QUANTIZATION_RE.match(parts[-1]):
        quantization = parts.pop()

    return {
        "family": family or text,
        "size_label": size_label,
        "size_b": size_b,
        "tuning": "-".join(parts),
        "quantization": quantization,
        "precision_bits": _PRECISION_BITS.get(quantization.lower(), 0),
    }


def discover_runs(root: PathLike) -> List[RunIdentity]:
    """Find every evaluable run under `root`, in directory-name order.

    A run is a directory holding a transcript file; `config_used.yaml` next to it
    supplies the identity. The search is recursive, so both a flat directory of
    runs and the cell/timestamp layout the pipeline writes are accepted.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    directories = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_dir():
            continue
        if any((candidate / name).exists()
               for name in ("transcripts.yaml", "transcripts.jsonl")):
            directories.append(candidate)
    if any((root / name).exists()
           for name in ("transcripts.yaml", "transcripts.jsonl")):
        directories.insert(0, root)

    runs: List[RunIdentity] = []
    for order, directory in enumerate(directories, 1):
        runs.append(_identify(directory, root, order))
    return runs


def _identify(run_dir: Path, root: Path, order: int) -> RunIdentity:
    """Read one run's configuration into a RunIdentity."""
    context = discover_run(run_dir)
    config = context.config
    cell = run_dir.parent.name if run_dir.parent != root else run_dir.name

    leading = re.match(r"^(\d+)", cell)
    parsed = parse_model_tag(config.get("ollama_model") or "")

    return RunIdentity(
        run_dir=run_dir,
        cell=cell,
        timestamp=run_dir.name,
        order=int(leading.group(1)) if leading else order,
        model_tag=str(config.get("ollama_model") or "unknown-model"),
        temperature=_as_float(config.get("llm_temperature")),
        seed=_as_int(config.get("llm_seed")),
        num_ctx=_as_int(config.get("llm_num_ctx")),
        max_tokens=_as_int(config.get("llm_max_tokens")),
        prompt_file=str(config.get("system_prompt_file") or ""),
        stt_engine=str(config.get("stt_engine") or ""),
        mode=str(config.get("mode") or ""),
        **_recognizer_fields(config),
        **parsed)


def _recognizer_fields(config: Dict[str, Any]) -> Dict[str, str]:
    """Read the acoustic model, device and compute type of the recognizer.

    The engine name alone does not identify a recognizer: the same engine at
    another acoustic model or compute type transcribes differently, and the
    pipeline records those under engine-specific keys (`vosk_model`,
    `whisper_model`, `whisper_device`, ...). Vosk is reported on the CPU because
    it offers no device choice, so its runs are not left with a blank device
    beside a recognizer that names one.
    """
    engine = str(config.get("stt_engine") or "").strip().lower()
    if not engine:
        return {"stt_model": "", "stt_device": "", "stt_compute": ""}

    raw_model = str(config.get(f"{engine}_model") or "").strip()
    model = Path(raw_model.replace("\\", "/")).name if raw_model else ""
    for prefix in (f"{engine}-model-", f"{engine}_model_", f"{engine}-"):
        if model.lower().startswith(prefix):
            model = model[len(prefix):]
            break

    device = str(config.get(f"{engine}_device") or "").strip().lower()
    if not device and engine == "vosk":
        device = "cpu"

    return {
        "stt_model": model,
        "stt_device": device,
        "stt_compute": str(config.get(f"{engine}_compute_type") or "").strip().lower(),
    }


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slug(text: str) -> str:
    """Reduce a label to a directory-safe name without losing its meaning."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._-]", "_", text)).strip("_")


@dataclass
class BatchConfig:
    """What to evaluate, how to compare it, and where the results go."""

    root: PathLike
    out_dir: Optional[PathLike] = None

    # Applied to every run, so that no difference between runs comes from the
    # measurement. The input paths are overwritten per run.
    evaluation: Optional[EvaluationConfig] = None

    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0
    include_checks: bool = True

    # Which contrasts to build. Restricting this does not change any score, only
    # which comparison tables are produced.
    group_kinds: List[str] = field(default_factory=lambda: list(GROUP_KINDS))

    progress: Optional[Callable[[str], None]] = None

    def __post_init__(self):
        self.root = Path(self.root)
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
        unknown = [kind for kind in self.group_kinds if kind not in GROUP_KINDS]
        if unknown:
            raise ValueError(f"unknown contrast kind(s): {', '.join(unknown)}")

    def resolved_out_dir(self) -> Path:
        """Where results go: `evaluation_result` beside the tree of runs.

        Keeping the results next to the runs rather than inside them means a
        re-evaluation never writes into the directory it is reading, and the
        whole comparison is one directory that can be archived on its own.
        """
        if self.out_dir is not None:
            return Path(self.out_dir)
        return Path(self.root).parent / "evaluation_result"

    def notify(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


@dataclass
class ContrastGroup:
    """One baseline and the runs to be compared against it."""

    kind: str
    group_id: str
    question: str
    varying: str
    baseline: RunIdentity
    contrasts: List[RunIdentity] = field(default_factory=list)

    @property
    def directory(self) -> str:
        return f"{self.kind}/{_slug(self.group_id)}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "group_id": self.group_id,
            "question": self.question,
            "varying": self.varying,
            "baseline": self.baseline.label,
            "contrasts": [run.label for run in self.contrasts],
            "directory": self.directory,
        }


def build_groups(runs: Sequence[RunIdentity],
                 kinds: Sequence[str] = GROUP_KINDS) -> List[ContrastGroup]:
    """Derive the contrasts a set of runs supports.

    A contrast is only formed when exactly one property differs between the runs
    in it, which is what makes the difference attributable to that property. A
    group with nothing to compare is not emitted.

    Where the batch spans more than one recognizer, the recognizer held fixed is
    named in the group identifier, so that two otherwise identical groups behind
    different recognizers cannot share a name or a directory.
    """
    groups: List[ContrastGroup] = []
    label_stt = len({run.recognizer for run in runs}) > 1
    for kind in kinds:
        if kind == "parameters":
            groups.extend(_parameter_groups(runs, label_stt))
        elif kind == "quantization":
            groups.extend(_quantization_groups(runs, label_stt))
        elif kind == "model_size":
            groups.extend(_size_groups(runs, label_stt))
        elif kind == "cross_model":
            groups.extend(_cross_model_groups(runs, label_stt))
        elif kind == "recognizer":
            groups.extend(_recognizer_groups(runs))
    return groups


def _qualify(group_id: str, run: RunIdentity, label_stt: bool) -> str:
    """Name the recognizer in a group identifier where more than one was used."""
    return f"{group_id}_{run.recognizer}" if label_stt else group_id


def _bucket(runs: Sequence[RunIdentity],
            key: Callable[[RunIdentity], tuple]) -> Dict[tuple, List[RunIdentity]]:
    grouped: Dict[tuple, List[RunIdentity]] = {}
    for run in runs:
        grouped.setdefault(key(run), []).append(run)
    return grouped


def _parameter_groups(runs: Sequence[RunIdentity],
                      label_stt: bool = False) -> List[ContrastGroup]:
    """One group per model variant, contrasting its decoding settings.

    The greedy setting is the baseline: it is the reproducible one, so a
    difference from it is the effect of sampling rather than of two samples.
    """
    groups = []
    buckets = _bucket(runs, lambda r: (r.model_tag, r.recognizer))
    for _, members in sorted(buckets.items()):
        settings = sorted(members, key=lambda r: (r.temperature is None,
                                                  r.temperature or 0.0,
                                                  r.order))
        if len(settings) < 2:
            continue
        baseline, *contrasts = settings
        groups.append(ContrastGroup(
            kind="parameters",
            group_id=_qualify(settings[0].model_tag, baseline, label_stt),
            question=GROUP_QUESTIONS["parameters"],
            varying="decoding settings (temperature, seed)",
            baseline=baseline, contrasts=contrasts))
    return groups


def _quantization_groups(runs: Sequence[RunIdentity],
                         label_stt: bool = False) -> List[ContrastGroup]:
    """One group per model and setting, contrasting weight precisions.

    The highest precision available is the baseline, so the tables read as the
    cost of compression rather than as the benefit of decompression.
    """
    groups = []
    buckets = _bucket(runs, lambda r: (r.family, r.size_label, r.tuning,
                                       r.setting, r.recognizer))
    for (family, size, _, setting, _), members in sorted(buckets.items()):
        precisions = {run.quantization for run in members}
        if len(precisions) < 2:
            continue
        ordered = sorted(members, key=lambda r: (-r.precision_bits, r.order))
        baseline, *contrasts = ordered
        groups.append(ContrastGroup(
            kind="quantization",
            group_id=_qualify(f"{family}_{size}_{setting}", baseline, label_stt),
            question=GROUP_QUESTIONS["quantization"],
            varying="weight precision",
            baseline=baseline, contrasts=contrasts))
    return groups


def _size_groups(runs: Sequence[RunIdentity],
                 label_stt: bool = False) -> List[ContrastGroup]:
    """One group per lineage, precision and setting, contrasting model sizes.

    The largest model is the baseline: it is the quality reference a smaller one
    is being proposed as a cheaper substitute for.

    Held together by lineage rather than by exact family, because the sizes on
    offer are split across releases: llama3.2 ships 1b and 3b, llama3.1 ships 8b,
    so requiring one family would break the ladder in two and leave the largest
    model out of it. Where a ladder does cross a release, that is stated in
    `varying`, since size is then not the only thing that changed.
    """
    groups = []
    buckets = _bucket(runs, lambda r: (r.lineage, r.quantization, r.tuning,
                                       r.setting, r.recognizer))
    for (lineage, quantization, _, setting, _), members in sorted(buckets.items()):
        if len({run.size_label for run in members}) < 2:
            continue
        ordered = sorted(members, key=lambda r: (-r.size_b, r.order))
        baseline, *contrasts = ordered

        releases = sorted({run.family for run in ordered})
        varying = "parameter count"
        if len(releases) > 1:
            varying += f" (and model release: {', '.join(releases)})"

        groups.append(ContrastGroup(
            kind="model_size",
            group_id=_qualify(f"{lineage}_{quantization}_{setting}",
                              baseline, label_stt),
            question=GROUP_QUESTIONS["model_size"],
            varying=varying,
            baseline=baseline, contrasts=contrasts))
    return groups


def _cross_model_groups(runs: Sequence[RunIdentity],
                        label_stt: bool = False) -> List[ContrastGroup]:
    """One group per decoding setting, contrasting every variant with the strongest.

    Held together by the setting, so the comparison is between models rather than
    between a model and a decoding choice. The strongest variant is the baseline,
    which makes every delta the cost of choosing a cheaper model.

    Strongest means most parameters, with weight precision breaking ties within a
    size. Ordering by precision first would make an 8b at 16 bits outrank a 32b at
    4 bits, and the parameter count is what dominates across models: the
    quantization contrast measures what those bits are worth, and it is a fraction
    of the gap between sizes.
    """
    groups = []
    buckets = _bucket(runs, lambda r: (r.setting, r.recognizer))
    for (setting, _), members in sorted(buckets.items()):
        if len({run.model_tag for run in members}) < 2:
            continue
        ordered = sorted(members, key=lambda r: (-r.size_b, -r.precision_bits,
                                                 r.order))
        baseline, *contrasts = ordered
        groups.append(ContrastGroup(
            kind="cross_model",
            group_id=_qualify(f"at_{setting}", baseline, label_stt),
            question=GROUP_QUESTIONS["cross_model"],
            varying="model",
            baseline=baseline, contrasts=contrasts))
    return groups


def _recognizer_groups(runs: Sequence[RunIdentity]) -> List[ContrastGroup]:
    """One group per model and setting, contrasting the recognizers used.

    The recognizer the rest of the grid was measured with is the baseline, so a
    delta reads as what switching recognizer changes relative to the established
    configuration. "The rest of the grid" is decided by how many runs each
    recognizer appears in, which is a property of the batch rather than a choice
    made per table; an equal split falls back to the earlier cell.

    Both sides answered the same recordings, but not the same words: the
    recognized text is the model's input, so a difference here is the joint
    effect of transcription accuracy and of whatever latency the recognizer adds.
    """
    frequency: Dict[str, int] = {}
    for run in runs:
        frequency[run.recognizer] = frequency.get(run.recognizer, 0) + 1

    groups = []
    buckets = _bucket(runs, lambda r: (r.model_tag, r.setting))
    for (model_tag, setting), members in sorted(buckets.items()):
        if len({run.recognizer for run in members}) < 2:
            continue
        ordered = sorted(members, key=lambda r: (-frequency[r.recognizer], r.order))
        baseline, *contrasts = ordered
        groups.append(ContrastGroup(
            kind="recognizer",
            group_id=f"{model_tag}_at_{setting}",
            question=GROUP_QUESTIONS["recognizer"],
            varying="speech recognizer (engine, acoustic model, device)",
            baseline=baseline, contrasts=contrasts))
    return groups


@dataclass
class BatchOutcome:
    """Everything a batch produced: the scored runs and every comparison."""

    root: Path
    runs: List[RunIdentity]
    outcomes: Dict[str, EvaluationOutcome]
    groups: List[ContrastGroup] = field(default_factory=list)
    comparisons: List[Tuple[ContrastGroup, ComparisonOutcome]] = \
        field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    report: str = ""
    alpha: float = 0.05
    # Answer key the runs were scored against, recorded so that a reference-based
    # figure can be traced to the table it came from.
    answer_key: Optional[Path] = None

    def outcome_for(self, run: RunIdentity) -> EvaluationOutcome:
        return self.outcomes[run.key]

    @property
    def corpus(self) -> List[Any]:
        """The item results of the first run, which carry the shared inputs."""
        if not self.runs:
            return []
        return self.outcome_for(self.runs[0]).results

    def recognizer_runs(self) -> Dict[str, List[RunIdentity]]:
        """The runs behind each recognizer, in the order the runs were read."""
        grouped: Dict[str, List[RunIdentity]] = {}
        for run in self.runs:
            grouped.setdefault(run.recognizer, []).append(run)
        return grouped

    def recognizer_corpora(self) -> Dict[str, List[Any]]:
        """One representative run per recognizer, for the transcripts it produced.

        The recording set is shared by every run, but the transcripts are not: two
        recognizers heard the same audio differently, so what the models were
        given has to be reported per recognizer rather than once.
        """
        return {recognizer: self.outcome_for(runs[0]).results
                for recognizer, runs in self.recognizer_runs().items()}

    def leaderboard(self) -> List[Dict[str, Any]]:
        """One row per run: its configuration next to its headline figures."""
        return [_leaderboard_row(run, self.outcome_for(run)) for run in self.runs]

    def strata_table(self) -> List[Dict[str, Any]]:
        """One row per run and input-quality stratum."""
        rows: List[Dict[str, Any]] = []
        for run in self.runs:
            summary = self.outcome_for(run).summary
            for stratum, entry in summary.per_stratum.items():
                rows.append({"run": run.label, "model_tag": run.model_tag,
                             "setting": run.setting,
                             "recognizer": run.recognizer, "stratum": stratum,
                             **entry})
        return rows

    def impact_table(self) -> List[Dict[str, Any]]:
        """Association between how badly an item was heard and how it scored."""
        rows: List[Dict[str, Any]] = []
        for run in self.runs:
            results = self.outcome_for(run).results
            errors = [r.asr.get("stt_wer") for r in results]
            if not any(isinstance(value, (int, float)) for value in errors):
                continue
            row: Dict[str, Any] = {"run": run.label, "model_tag": run.model_tag,
                                   "setting": run.setting,
                                   "recognizer": run.recognizer,
                                   "n_items": len(results)}
            for name, getter in _IMPACT_METRICS.items():
                row[f"rho_{name}"] = _round(spearman(errors,
                                                     [getter(r) for r in results]))
            rows.append(row)
        return rows

    def asr_reference(self) -> List[Dict[str, Any]]:
        """The shared input set: what was asked, what was heard, how far apart.

        Written once per recognizer rather than once per run, because the
        transcript is a property of the recording set and the recognizer that
        decoded it, and is identical across every run sharing that recognizer.
        """
        rows: List[Dict[str, Any]] = []
        for recognizer, results in self.recognizer_corpora().items():
            for result in results:
                if not result.asr:
                    continue
                rows.append({
                    "recognizer": recognizer,
                    "item_id": result.item_id,
                    "filename": result.record.filename,
                    "ori_text": result.record.ori_text,
                    "stt_text": result.record.stt_text,
                    **{key: _round(value) for key, value in result.asr.items()},
                })
        return rows

    def all_items(self) -> List[Dict[str, Any]]:
        """Every item of every run in one long table, for a pivot or a plot."""
        rows: List[Dict[str, Any]] = []
        for run in self.runs:
            outcome = self.outcome_for(run)
            for result in outcome.results:
                rows.append({
                    "run": run.label,
                    "cell": run.cell,
                    "model_tag": run.model_tag,
                    "family": run.family,
                    "size_label": run.size_label,
                    "quantization": run.quantization,
                    "temperature": run.temperature,
                    "seed": run.seed,
                    "recognizer": run.recognizer,
                    **result.flat_row(outcome.rubric),
                })
        return rows

    def comparison_index(self) -> List[Dict[str, Any]]:
        """Every metric of every contrast in one table, keyed by group."""
        rows: List[Dict[str, Any]] = []
        for group, comparison in self.comparisons:
            for pair in comparison.pairs:
                for metric in pair.metrics:
                    rows.append({
                        "group_kind": group.kind,
                        "group_id": group.group_id,
                        "varying": group.varying,
                        "baseline": pair.baseline_label,
                        "contrast": pair.contrast_label,
                        "n_paired": pair.n_paired,
                        **metric.as_dict(),
                    })
        return rows

    def manifest(self) -> Dict[str, Any]:
        """Provenance: which runs were read, with which settings, into what."""
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "root": str(self.root),
            "alpha": self.alpha,
            "answer_key": str(self.answer_key) if self.answer_key else None,
            "n_runs": len(self.runs),
            "runs": [
                {**run.as_dict(),
                 "n_items": self.outcome_for(run).summary.n_items,
                 "generation_settings":
                     self.outcome_for(run).context.generation_settings,
                 "results_dir": f"runs/{_slug(run.cell)}",
                 "latency_log": (str(self.outcome_for(run).latency.path)
                                 if self.outcome_for(run).latency
                                 and self.outcome_for(run).latency.path else None)}
                for run in self.runs],
            "contrasts": [group.as_dict() for group in self.groups],
            "warnings": self.warnings,
            "method_references": [
                ref._asdict() for ref
                in references.resolve(_batch_reference_keys(self))],
        }

    def write(self, out_dir: PathLike) -> Path:
        """Write every artefact of the batch under one directory."""
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)

        for run in self.runs:
            self.outcome_for(run).write(target / "runs" / _slug(run.cell))

        for group, comparison in self.comparisons:
            comparison.write(target / "comparisons" / group.kind
                             / _slug(group.group_id))

        reporting.write_text(target / "summary_report.txt", self.report)
        reporting.write_json(target / "batch_manifest.json", self.manifest())
        _write_csv(target / "leaderboard.csv", self.leaderboard())
        _write_csv(target / "comparison_index.csv", self.comparison_index())
        _write_csv(target / "input_quality_strata.csv", self.strata_table())
        _write_csv(target / "input_quality_impact.csv", self.impact_table())
        _write_csv(target / "input_reference.csv", self.asr_reference())
        _write_csv(target / "all_items.csv", self.all_items())
        return target


# Metrics correlated with the recognizer's error rate. Chosen because each one
# would move for a different reason if the input quality mattered: whether the
# prompt was obeyed, whether the answer addressed the request, how much was said,
# and how long the exchange took.
_IMPACT_METRICS: Dict[str, Callable[[Any], Optional[float]]] = {
    "adherence": lambda r: (float(r.constraints.item_level_strict)
                            if r.constraints is not None else None),
    "request_coverage": lambda r: r.relevance.get("request_coverage"),
    "intent_coverage": lambda r: r.relevance.get("intent_coverage"),
    "word_count": lambda r: (r.constraints.measures.get("word_count")
                             if r.constraints is not None else None),
    "e2e_ms": lambda r: r.latency.get("lat_e2e_response_ready"),
}


def _leaderboard_row(run: RunIdentity,
                     outcome: EvaluationOutcome) -> Dict[str, Any]:
    """Flatten one run into the row a results table reports."""
    summary = outcome.summary
    stages = {entry["stage"]: entry for entry in summary.latency_stages}
    resources = summary.resource_medians

    check_rates = [r.constraints.check_level_strict for r in outcome.results
                   if r.constraints is not None
                   and r.constraints.check_level_strict is not None]

    row: Dict[str, Any] = {
        "run": run.label,
        "cell": run.cell,
        "model_tag": run.model_tag,
        "family": run.family,
        "size_label": run.size_label,
        "size_b": run.size_b,
        "quantization": run.quantization,
        "precision_bits": run.precision_bits,
        "temperature": run.temperature,
        "seed": run.seed,
        "num_ctx": run.num_ctx,
        "max_tokens": run.max_tokens,
        "system_prompt_file": run.prompt_file,
        "stt_engine": run.stt_engine,
        "stt_model": run.stt_model,
        "stt_device": run.stt_device,
        "recognizer": run.recognizer,
        "mode": run.mode,
        "n_items": summary.n_items,
        "n_empty_responses": summary.n_empty_responses,
        "adherence_item_strict": _round(summary.constraint_item_rate_strict),
        "adherence_item_loose": _round(summary.constraint_item_rate_loose),
        "check_pass_rate_strict": _round(statistics.fmean(check_rates)
                                         if check_rates else None),
    }

    # Answer agreement, reported per reference subset rather than pooled: a rate
    # that mixes verified answers with merely plausible ones cannot be read as
    # accuracy. See aggregation._answer_accuracy for what each subset supports.
    accuracy = summary.answer_accuracy
    if accuracy:
        short = accuracy.get("answerable_short_span") or {}
        long_span = accuracy.get("answerable_long_span") or {}
        plausible = accuracy.get("unanswerable_plausible") or {}
        row.update({
            "answer_items_scored": accuracy.get("n_items_with_reference"),
            "n_answerable_short_span": short.get("n_items"),
            "answer_presence_short_span": short.get("answer_presence_rate"),
            "exact_match_short_span": short.get("exact_match_rate"),
            "token_f1_short_span": short.get("token_f1_mean"),
            "answer_presence_long_span": long_span.get("answer_presence_rate"),
            "answer_presence_plausible": plausible.get("answer_presence_rate"),
        })

    for name, column in (("request_coverage", "coverage_stt_median"),
                         ("intent_coverage", "coverage_ori_median"),
                         ("echo_ratio", "echo_median"),
                         ("word_count", "words_median"),
                         ("sentence_count", "sentences_median"),
                         ("opening_words", "opening_words_median"),
                         ("flesch_reading_ease", "reading_ease_median"),
                         ("flesch_kincaid_grade", "grade_level_median"),
                         ("stt_wer", "stt_wer_median_control"),
                         ("llm_tokens_per_sec", "tokens_per_sec_median"),
                         ("llm_prompt_tokens", "prompt_tokens_median"),
                         ("llm_eval_tokens", "generated_tokens_median"),
                         ("tts_audio_ms", "tts_audio_ms_median")):
        row[column] = _round(summary.metric_summaries.get(name, {}).get("median"))

    # Every stage the run recorded, so that the table can be read as a
    # decomposition of what the user waits for rather than as headline figures
    # only. The tail is added where a tail is what matters: the delay a user
    # perceives, and the span of the whole exchange.
    for stage, column in (("llm_prompt_eval", "prompt_eval_ms"),
                          ("llm_ttft", "ttft_ms"),
                          ("llm_first_chunk_fill", "chunk_fill_ms"),
                          ("llm_ttfc", "ttfc_ms"),
                          ("tts_first_chunk", "tts_first_chunk_ms"),
                          ("ttfa", "ttfa_ms"), ("llm_eval", "llm_eval_ms"),
                          ("tts_total", "tts_total_ms"),
                          ("e2e_response_ready", "e2e_ms"),
                          ("stt", "stt_ms_control"),
                          ("stt_endpoint_delay", "endpoint_delay_ms_control")):
        entry = stages.get(stage, {})
        row[f"{column}_median"] = _round(entry.get("median_ms"), 1)
        if stage in ("ttfa", "e2e_response_ready"):
            row[f"{column}_p90"] = _round(entry.get("p90_ms"), 1)
            row[f"{column}_p95"] = _round(entry.get("p95_ms"), 1)

    for name in ("llm_model_vram_mb", "llm_vram_mb", "gpu_mem_used_mb",
                 "llm_rss_mb", "cpu_percent", "ram_percent",
                 "gpu_util_percent"):
        row[name] = _round(resources.get(name), 1)

    row["run_dir"] = str(run.run_dir)
    row["results_dir"] = f"runs/{_slug(run.cell)}"
    return row


def _batch_reference_keys(outcome: "BatchOutcome") -> List[str]:
    """Every method reference the batch's own artefacts rely on."""
    keys = list(PAIRED_REFERENCE_KEYS)
    for run in outcome.runs:
        keys.extend(outcome.outcome_for(run).reference_keys)
    return keys


def _round(value: Any, digits: int = 4) -> Any:
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return None if number != number else round(number, digits)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows as CSV, with the union of their keys as the header.

    The union rather than the first row's keys: a run evaluated with a tier the
    others skipped would otherwise have its columns silently dropped.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_batch(config: BatchConfig) -> BatchOutcome:
    """Evaluate every run under the root and build every supported contrast.

    Raises:
        NotADirectoryError: the root does not exist.
        ValueError: the root holds no run with a transcript file.
    """
    runs = discover_runs(config.root)
    if not runs:
        raise ValueError(f"no run directory with transcripts under {config.root}")

    base = config.evaluation or EvaluationConfig()
    outcomes: Dict[str, EvaluationOutcome] = {}
    for index, run in enumerate(runs, 1):
        config.notify(f"evaluating run {index}/{len(runs)}: {run.label}")
        settings = replace(base, run_dir=run.run_dir, transcripts=None,
                           label=run.label, seed=config.seed)
        outcomes[run.key] = run_evaluation(settings)

    groups = build_groups(runs, config.group_kinds)

    warnings = verify_shared_inputs(runs, outcomes, groups)
    for run in runs:
        warnings.extend(f"{run.label}: {text}"
                        for text in outcomes[run.key].warnings)

    settings = ComparisonConfig(evaluation=base, alpha=config.alpha,
                                n_boot=config.n_boot, seed=config.seed,
                                include_checks=config.include_checks,
                                progress=config.progress)

    comparisons: List[Tuple[ContrastGroup, ComparisonOutcome]] = []
    for index, group in enumerate(groups, 1):
        config.notify(f"comparing group {index}/{len(groups)}: "
                      f"{group.kind}/{group.group_id}")
        comparisons.append((group, compare_evaluated(
            outcomes[group.baseline.key],
            [outcomes[run.key] for run in group.contrasts], settings)))

    outcome = BatchOutcome(root=Path(config.root), runs=runs, outcomes=outcomes,
                           groups=groups, comparisons=comparisons,
                           warnings=warnings, alpha=config.alpha,
                           answer_key=config.evaluation.answer_key
                           if config.evaluation else None)
    outcome.report = render_batch_report(outcome)
    return outcome


def verify_shared_inputs(runs: Sequence[RunIdentity],
                         outcomes: Dict[str, EvaluationOutcome],
                         groups: Optional[Sequence["ContrastGroup"]] = None
                         ) -> List[str]:
    """Check the assumptions the paired comparisons rest on.

    A paired comparison is only about the configuration if everything else was
    held fixed. Rather than assume that, each assumption is checked and any breach
    is reported with the results it affects: the alternative is a table of
    differences that silently includes a change of prompt or of recording set.

    Given the contrasts, each assumption is checked within the runs actually
    compared and reported against that contrast, because a property that varies
    across the batch need not vary inside any one comparison: a batch spanning two
    recognizers is not a breach if every model contrast holds one of them fixed.
    Without them the whole batch is checked as a single set.
    """
    if groups is None:
        return _verify_set(runs, outcomes)

    warnings: List[str] = []
    for group in groups:
        members = [group.baseline, *group.contrasts]
        for text in _verify_set(members, outcomes,
                               recognizer_varies=group.kind == "recognizer"):
            entry = f"{group.kind}/{group.group_id}: {text}"
            if entry not in warnings:
                warnings.append(entry)
    return warnings


def _verify_set(runs: Sequence[RunIdentity],
                outcomes: Dict[str, EvaluationOutcome],
                recognizer_varies: bool = False) -> List[str]:
    """Check one set of runs that is meant to differ in one property only.

    Where the recognizer is the property under test, the transcripts are expected
    to differ and the intended question is checked instead: that is what must
    still be shared for the pairing to compare like with like.
    """
    warnings: List[str] = []
    if len(runs) < 2:
        return warnings

    reference = runs[0]
    reference_results = outcomes[reference.key].results
    reference_items = [r.item_id for r in reference_results]

    def inputs_of(result: Any) -> tuple:
        return ((result.record.ori_text,) if recognizer_varies
                else (result.record.stt_text, result.record.ori_text))

    reference_inputs = {r.item_id: inputs_of(r) for r in reference_results}
    input_kind = "intended question" if recognizer_varies else "input text"

    for run in runs[1:]:
        results = outcomes[run.key].results
        items = [r.item_id for r in results]
        if set(items) != set(reference_items):
            missing = len(set(reference_items) - set(items))
            extra = len(set(items) - set(reference_items))
            warnings.append(
                f"{run.label} does not answer the same items as "
                f"{reference.label} ({missing} missing, {extra} extra): those "
                f"items are excluded from every comparison")

        differing = sum(1 for r in results
                        if r.item_id in reference_inputs
                        and reference_inputs[r.item_id] != inputs_of(r))
        if differing:
            warnings.append(
                f"{run.label} was given a different {input_kind} than "
                f"{reference.label} on {differing} item(s): a difference in the "
                f"responses is then partly a difference in the questions")

    invariants = [
        ("system prompt", {run.label: outcomes[run.key].context.system_prompt
                           for run in runs}),
        ("constraint spec", {run.label: outcomes[run.key].spec.spec_id
                             for run in runs}),
        ("compute mode", {run.label: run.mode for run in runs}),
        ("context window", {run.label: run.num_ctx for run in runs}),
        ("token cap", {run.label: run.max_tokens for run in runs}),
    ]
    if not recognizer_varies:
        invariants.insert(2, ("recognizer",
                              {run.label: run.recognizer for run in runs}))

    for name, values in invariants:
        distinct = {value for value in values.values()}
        if len(distinct) > 1:
            warnings.append(
                f"the runs do not share one {name}: comparisons between them "
                f"also carry that difference")

    warmups = {tuple(outcomes[run.key].latency.warmup_items)
               for run in runs if outcomes[run.key].latency is not None}
    if len(warmups) > 1:
        warnings.append(
            "the runs excluded different warm-up items, so the timing "
            "comparisons are not over an identical item set")
    return warnings


THICK = "=" * 132
THIN = "-" * 132


class _Sections:
    """Numbers sections in the order they are emitted.

    Sections are omitted when the tier feeding them did not run, and fixed
    numbering would then leave gaps that read as missing output.
    """

    def __init__(self):
        self._next = 0

    def title(self, text: str) -> List[str]:
        self._next += 1
        return [f"{self._next}. {text}", THICK]


def render_batch_report(outcome: BatchOutcome) -> str:
    """Render the one file that answers "which configuration was better"."""
    sections = _Sections()
    lines: List[str] = [THICK, "RUN GRID EVALUATION".center(132), THICK]
    lines += _batch_header(outcome)
    lines += _shared_input_section(outcome, sections)
    lines += _quality_leaderboard(outcome, sections)
    lines += _answer_accuracy_section(outcome, sections)
    lines += _runtime_leaderboard(outcome, sections)
    lines += _strata_section(outcome, sections)
    lines += _impact_section(outcome, sections)
    lines += _contrast_section(outcome, sections)
    lines += _files_section(outcome, sections)
    lines += _methods_section(outcome, sections)
    lines += _reading_section(
        sections,
        with_answer_key=any(row.get("answer_items_scored")
                            for row in outcome.leaderboard()))
    return "\n".join(lines)


def _batch_header(outcome: BatchOutcome) -> List[str]:
    first = outcome.outcome_for(outcome.runs[0]) if outcome.runs else None
    items = sorted({outcome.outcome_for(run).summary.n_items
                    for run in outcome.runs})
    lines = [
        f"Generated on           : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Run tree               : {outcome.root}",
        f"Runs evaluated         : {len(outcome.runs)}",
        f"Items per run          : "
        + (", ".join(str(count) for count in items) if items else "-"),
        f"Contrasts built        : {len(outcome.groups)}"
        + (f"  ({', '.join(f'{kind}: {sum(1 for g in outcome.groups if g.kind == kind)}' for kind in GROUP_KINDS if any(g.kind == kind for g in outcome.groups))})"
           if outcome.groups else ""),
        f"Significance level     : {outcome.alpha} (Holm-corrected within each "
        f"metric family)",
    ]
    if outcome.answer_key:
        lines.append(f"Answer key             : {outcome.answer_key}")
    if first is not None:
        lines.append(f"Constraint spec        : {first.spec.spec_id}")
        lines.append(f"System prompt          : "
                     f"{first.context.system_prompt_path or '(not recorded)'}")
        lines.append("")
        lines.append("Tiers executed for every run:")
        for name, state in first.tiers.items():
            lines.append(f"  {name:<34}: {state}")

    if outcome.warnings:
        lines.append("")
        lines.append("WARNINGS about what these comparisons can support:")
        for warning in outcome.warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("")
        lines.append("Assumption checks passed: within every contrast the runs "
                     "share their items, input")
        lines.append("text, prompt, constraint spec, compute mode, context "
                     "window and token cap, and the")
        lines.append("recognizer except where it is the property under test, so "
                     "a difference between them is")
        lines.append("a difference of the property the contrast varies.")

    lines.append(THICK)
    lines.append("")
    return lines


def _shared_input_section(outcome: BatchOutcome,
                          sections: _Sections) -> List[str]:
    """State what each recognizer handed to the models behind it."""
    corpora = {recognizer: [result for result in results if result.asr]
               for recognizer, results in outcome.recognizer_corpora().items()}
    corpora = {name: results for name, results in corpora.items() if results}
    if not corpora:
        return []

    single = len(corpora) == 1
    title = ("THE SHARED INPUT SET (identical in every run)" if single
             else "THE SHARED INPUT SET (same recordings, one transcript per "
                  "recognizer)")
    lines = sections.title(title)

    for recognizer, corpus in corpora.items():
        records = [result.record for result in corpus]
        rate, errors, words = corpus_error_rate(records)
        exact = sum(1 for result in corpus if result.asr.get("stt_exact_match"))
        counts = {name: sum(1 for result in corpus
                            if result.asr.get("stt_stratum") == name)
                  for name in STRATA}
        if not single:
            runs = len(outcome.recognizer_runs().get(recognizer, ()))
            lines += [f"{recognizer}  ({runs} run(s))", THIN]
        lines += [f"Recordings                 : {len(corpus)}",
                  f"Corpus word error rate     : {rate:.3f} "
                  f"({errors} errors over {words} reference words)",
                  f"Recognized exactly         : {exact}/{len(corpus)} "
                  f"({100 * exact / len(corpus):.1f}%)",
                  "Input quality strata       : "
                  + ", ".join(f"{name} {counts[name]}" for name in STRATA),
                  ""]

    if single:
        lines += ["Every run answered these same recordings, so the "
                  "recognizer's errors are held constant and",
                  "the comparisons below are between models."]
    else:
        lines += ["Every run answered the same recordings, but not through the "
                  "same recognizer: model",
                  "contrasts hold one recognizer fixed, and the recognizer "
                  "contrast is where the transcripts",
                  "themselves differ."]
    lines += ["The error rate bounds what any model could have answered: with "
              "a corpus error rate this",
              "high, most items present the model with a question that differs "
              "from the one the speaker",
              "asked, and a low relevance score can be the recognizer's doing "
              "rather than the model's.",
              "Per-item figures are in input_reference.csv.",
              THICK, ""]
    return lines


def _quality_leaderboard(outcome: BatchOutcome,
                         sections: _Sections) -> List[str]:
    rows = outcome.leaderboard()
    if not rows:
        return []

    lines = sections.title("RESPONSE QUALITY BY RUN")
    lines += [f"{'Run':<34}{'n':>5}{'Adher%':>8}{'Chk%':>7}{'Cov(stt)':>10}"
              f"{'Cov(ori)':>10}{'Echo':>8}{'Words':>7}{'Sent':>6}{'FRE':>7}"
              f"{'Grade':>7}",
              THIN]
    for row in rows:
        lines.append(
            f"{str(row['run'])[:33]:<34}"
            f"{row['n_items']:>5}"
            f"{_pct(row['adherence_item_strict']):>8}"
            f"{_pct(row['check_pass_rate_strict']):>7}"
            f"{_fmt(row['coverage_stt_median'], 3):>10}"
            f"{_fmt(row['coverage_ori_median'], 3):>10}"
            f"{_fmt(row['echo_median'], 3):>8}"
            f"{_fmt(row['words_median'], 1):>7}"
            f"{_fmt(row['sentences_median'], 1):>6}"
            f"{_fmt(row['reading_ease_median'], 1):>7}"
            f"{_fmt(row['grade_level_median'], 1):>7}")

    lines += [THIN,
              "Adher% = share of items satisfying every hard prompt constraint. "
              "Chk% = mean share of hard",
              "constraints satisfied per item, which moves even when no item "
              "passes all of them. Cov(stt)",
              "and Cov(ori) are median coverage of the recognized and of the "
              "intended question. Echo is the",
              "share of the answer that repeats the question. The remaining "
              "columns are medians.",
              "",
              "This table ranks nothing on its own: a difference here is not "
              "necessarily larger than the",
              "item-to-item variation. The paired contrasts below test the "
              "differences item by item.",
              THICK, ""]
    return lines


def _answer_accuracy_section(outcome: BatchOutcome,
                             sections: _Sections) -> List[str]:
    """Agreement with the dataset's own reference answers, per run.

    Present only when an answer key was supplied. The subsets are kept apart in
    the table because they do not support the same claim: presence of a short gold
    span in an answerable item is evidence that the answer was right, presence of a
    plausible span in an item the corpus marks unanswerable is not.
    """
    rows = [row for row in outcome.leaderboard()
            if row.get("answer_items_scored")]
    if not rows:
        return []

    lines = sections.title("ANSWER AGREEMENT WITH THE DATASET REFERENCE ANSWERS")
    lines += [f"{'Run':<34}{'n':>5}{'Present%':>10}{'TokenF1':>9}{'EM%':>7}"
              f"{'Plaus%':>9}{'LongSpan%':>11}",
              THIN]
    for row in rows:
        lines.append(
            f"{str(row['run'])[:33]:<34}"
            f"{_fmt(row['n_answerable_short_span'], 0):>5}"
            f"{_pct(row['answer_presence_short_span']):>10}"
            f"{_fmt(row['token_f1_short_span'], 3):>9}"
            f"{_pct(row['exact_match_short_span']):>7}"
            f"{_pct(row['answer_presence_plausible']):>9}"
            f"{_pct(row['answer_presence_long_span']):>11}")

    first = rows[0]
    lines += [THIN,
              f"n = answerable items whose gold answer is a span of at most "
              f"{SHORT_ANSWER_MAX_WORDS} words, the subset on which",
              "agreement is evidence about the answer. Present% = share of those "
              "items whose response",
              "contains the gold span under SQuAD normalization (Rajpurkar et al., "
              "2016; the answer-presence",
              "convention of Chen et al., 2017). TokenF1 and EM% are the SQuAD "
              "measures over the same subset;",
              "EM% is near zero by construction here, because the assistant "
              "answers in a sentence rather than",
              "emitting the span alone, and TokenF1 is depressed by every word of "
              "that sentence.",
              "",
              "Plaus% covers the items the corpus marks unanswerable, where only a "
              "plausible answer exists",
              "(Rajpurkar et al., 2018): agreement there is similarity to what an "
              "annotator would have said and",
              "is not correctness. LongSpan% covers answerable items whose gold "
              "answer is a passage extract",
              "rather than an answer; no response of a few sentences can reproduce "
              "one, so it is reported",
              "apart rather than pooled.",
              "",
              "Present% bounds correctness from above: a response containing the "
              "gold span may still be wrong",
              "elsewhere, and one phrasing the right answer in other words is "
              "counted as a miss. Read it as the",
              f"reference-based ceiling on answer accuracy over "
              f"{first.get('answer_items_scored')} items with a reference, not as "
              f"a graded score.",
              THICK, ""]
    return lines


def _runtime_leaderboard(outcome: BatchOutcome,
                         sections: _Sections) -> List[str]:
    rows = [row for row in outcome.leaderboard()
            if row.get("e2e_ms_median") is not None
            or row.get("ttft_ms_median") is not None]
    if not rows:
        return []

    lines = sections.title("RUNTIME COST BY RUN (medians in milliseconds)")
    lines += [f"{'Run':<34}{'TTFT':>8}{'TTFC':>8}{'TTFA':>8}{'Gen':>8}"
              f"{'E2E':>9}{'E2E p95':>10}{'tok/s':>8}{'Tokens':>8}"
              f"{'VRAM MB':>9}{'RSS MB':>9}",
              THIN]
    for row in rows:
        lines.append(
            f"{str(row['run'])[:33]:<34}"
            f"{_fmt(row['ttft_ms_median'], 0):>8}"
            f"{_fmt(row['ttfc_ms_median'], 0):>8}"
            f"{_fmt(row['ttfa_ms_median'], 0):>8}"
            f"{_fmt(row['llm_eval_ms_median'], 0):>8}"
            f"{_fmt(row['e2e_ms_median'], 0):>9}"
            f"{_fmt(row['e2e_ms_p95'], 0):>10}"
            f"{_fmt(row['tokens_per_sec_median'], 1):>8}"
            f"{_fmt(row['generated_tokens_median'], 0):>8}"
            f"{_fmt(row['llm_model_vram_mb'], 0):>9}"
            f"{_fmt(row['llm_rss_mb'], 0):>9}")

    # The recognition stage is a control only among runs that share a recognizer:
    # another recognizer decodes the same audio in its own time, so pooling them
    # would report a real difference as host drift.
    control_by_stt: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get("stt_ms_control_median")
        if value is not None:
            control_by_stt.setdefault(str(row.get("recognizer") or ""),
                                      []).append(value)
    lines += [THIN,
              "TTFA is the figure a user perceives: from the speaker stopping to "
              "the first audio out. TTFT and",
              "TTFC are measured over the same span, to the model's first token "
              "and to its first speakable",
              "chunk. Gen = generation time. E2E is the wall-clock span of the "
              "whole item, so it includes the",
              "recording being streamed in at speaking pace and is dominated by "
              "it; read TTFA for what the",
              "model change cost the user, and E2E only for the tail. Tokens = "
              "median tokens generated: read",
              "Gen next to it, since a model that says more takes longer for that "
              "reason alone. VRAM MB is the",
              "model's own footprint as reported by the driver; where it is blank, "
              "the run's log did not",
              "attribute VRAM to the model, and the runner's total is in "
              "leaderboard.csv instead."]
    if control_by_stt:
        lines.append("")
        lines.append("Control: the recognition stage is the recording arriving "
                     "at speaking pace rather than")
        lines.append("compute, so among runs sharing a recognizer its spread is "
                     "the drift of the measuring")
        lines.append("host, and a timing difference below that spread carries no "
                     "information about the model.")
        for recognizer, values in control_by_stt.items():
            name = recognizer or "recognizer not recorded"
            lines.append(f"  {name}: {min(values):.0f}-{max(values):.0f} ms "
                         f"over {len(values)} run(s)")
    lines += [THICK, ""]
    return lines


def _strata_section(outcome: BatchOutcome, sections: _Sections) -> List[str]:
    rows = outcome.strata_table()
    if not rows:
        return []

    lines = sections.title("PROMPT ADHERENCE BY INPUT QUALITY")
    lines += [f"{'Run':<34}"
              + "".join(f"{name + ' n':>10}{name + ' adher%':>14}"
                        for name in STRATA),
              THIN]
    by_run: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run"]), {})[str(row["stratum"])] = row

    for run in outcome.runs:
        entries = by_run.get(run.label)
        if not entries:
            continue
        cells = ""
        for name in STRATA:
            entry = entries.get(name, {})
            cells += f"{entry.get('n_items', 0):>10}"
            cells += f"{_pct(entry.get('constraint_item_rate_strict')):>14}"
        lines.append(f"{run.label[:33]:<34}{cells}")

    lines += [THIN,
              "Strata are defined on the word error rate of the item and are the "
              "same in every run: clean =",
              "recognized exactly, mild = up to a third of the reference words "
              "lost or altered, severe =",
              "more than a third. A model whose adherence falls across the "
              "strata is being pushed off the",
              "prompt by the recognizer's errors, which is a property of the "
              "assembled system rather than of",
              "the prompt. Coverage is deliberately not shown here: a correct "
              "answer to a closed question",
              "repeats none of it, so coverage differs between strata for "
              "reasons unrelated to the model.",
              "Full figures per stratum are in input_quality_strata.csv.",
              THICK, ""]
    return lines


def _impact_section(outcome: BatchOutcome, sections: _Sections) -> List[str]:
    rows = outcome.impact_table()
    if not rows:
        return []

    lines = sections.title("HOW THE RECOGNITION ERROR TRACKS THE RESPONSE "
                           "(Spearman rho per run)")
    lines += [f"{'Run':<34}{'adherence':>12}{'cov(stt)':>11}{'cov(ori)':>11}"
              f"{'words':>10}{'e2e ms':>10}",
              THIN]
    for row in rows:
        lines.append(
            f"{str(row['run'])[:33]:<34}"
            f"{_fmt(row.get('rho_adherence'), 2):>12}"
            f"{_fmt(row.get('rho_request_coverage'), 2):>11}"
            f"{_fmt(row.get('rho_intent_coverage'), 2):>11}"
            f"{_fmt(row.get('rho_word_count'), 2):>10}"
            f"{_fmt(row.get('rho_e2e_ms'), 2):>10}")

    lines += [THIN,
              "Rank correlation between an item's word error rate and its score, "
              "within one run. A negative",
              "value on coverage or adherence means the model does worse on the "
              "items it was given a worse",
              "transcript of. This is an association over items, not an "
              "experiment: the items the recognizer",
              "fails on also differ in length and topic, so read it as the size "
              "of a confound to control",
              "for, not as the effect of recognition error.",
              THICK, ""]
    return lines


def _contrast_section(outcome: BatchOutcome, sections: _Sections) -> List[str]:
    if not outcome.comparisons:
        return []

    lines = sections.title("CONTRASTS (paired over the shared recordings)")
    lines += ["Deltas are contrast minus baseline. A star marks a difference "
              "that survives Holm correction",
              "within its metric family. Adher pp is in percentage points of "
              "item-level prompt adherence, and",
              "TTFA is the time from the speaker stopping to the first audio out. "
              "The four columns are a digest",
              "of about forty metrics; the full table of each contrast is the file "
              "named under it.",
              ""]

    for kind in GROUP_KINDS:
        groups = [(g, c) for g, c in outcome.comparisons if g.kind == kind]
        if not groups:
            continue
        lines.append(f"{kind.upper().replace('_', ' ')}  --  "
                     f"{GROUP_QUESTIONS[kind]}")
        lines.append(THIN)
        for group, comparison in groups:
            lines.append(f"  {group.group_id}   (varying: {group.varying})")
            lines.append(f"  baseline: {group.baseline.label}")
            lines.append(f"    {'Contrast':<32}{'n':>5}{'Adher pp':>10}"
                         f"{'Cov(ori)':>10}{'TTFA ms':>10}{'tok/s':>9}"
                         f"   changed by family")
            for pair in comparison.pairs:
                lines.append(
                    f"    {pair.contrast_label[:31]:<32}"
                    f"{pair.n_paired:>5}"
                    f"{_delta(pair, 'constraint_item_pass_strict', 100.0, 1):>10}"
                    f"{_delta(pair, 'intent_coverage', 1.0, 3):>10}"
                    f"{_delta(pair, 'lat_ttfa', 1.0, 0):>10}"
                    f"{_delta(pair, 'llm_tokens_per_sec', 1.0, 1):>9}"
                    f"   {_family_counts(pair)}")
            lines.append(f"    full table: comparisons/{group.kind}/"
                         f"{_slug(group.group_id)}/comparison_report.txt")
            lines.append("")
    lines += [THICK, ""]
    return lines


def _files_section(outcome: BatchOutcome, sections: _Sections) -> List[str]:
    lines = sections.title("WHAT WAS WRITTEN")
    entries = [
        ("summary_report.txt", "this file"),
        ("leaderboard.csv", "one row per run: its configuration and its figures"),
        ("comparison_index.csv",
         "every metric of every contrast, with p-values and verdicts"),
        ("input_reference.csv",
         "the shared input set: intended text, recognized text, error rates"),
        ("input_quality_strata.csv", "each run's figures by input quality"),
        ("input_quality_impact.csv",
         "rank correlation of input error with each response measure"),
        ("all_items.csv",
         f"every item of every run ({len(outcome.runs)} x n) for a pivot or plot"),
        ("batch_manifest.json",
         "which runs were read, with which settings, and where their results are"),
        ("runs/<cell>/",
         "per-run report, per-item CSV, JSON, and a copy of config_used.yaml"),
        ("comparisons/<kind>/<group>/",
         "the full paired comparison table for one contrast"),
    ]
    for name, description in entries:
        lines.append(f"  {name:<28} {description}")
    lines += [THICK, ""]
    return lines


def _methods_section(outcome: BatchOutcome, sections: _Sections) -> List[str]:
    lines = sections.title("METHODS AND SOURCES")
    lines += ["Validation status: verifiable = decided by construction; "
              "validated = validated instrument or",
              "published sampling theory; established = widely used method with "
              "human-correlation evidence;",
              "surrogate = simplification of a published method, reported as "
              "such.",
              THIN]
    lines += references.bibliography_lines(_batch_reference_keys(outcome))
    lines += [THICK, ""]
    return lines


def _reading_section(sections: _Sections,
                     with_answer_key: bool = False) -> List[str]:
    lines = sections.title("WHAT THIS DOES AND DOES NOT ESTABLISH")
    if with_answer_key:
        notes = [
            "Factual correctness is measured only as agreement with the "
            "dataset's reference answer. A",
            "response is credited when it contains the gold span, so a wrong "
            "answer that happens to quote",
            "the span is credited and a right answer in other words is not. "
            "That criterion is a bound on",
            "accuracy, not a graded judgement of it: whether the rest of the "
            "response is true, and whether a",
            "missed item was wrong or merely reworded, needs a judge model "
            "(--judge-model) or a human pass.",
            "",
        ]
    else:
        notes = [
            "Factual correctness is not measured here. No reference answer was "
            "supplied for these items,",
            "so nothing in this report distinguishes a correct answer from a "
            "confident wrong one. Prompt",
            "adherence, coverage, readability and latency are all measurable "
            "without a reference; accuracy",
            "is not. Supply reference answers through --answer-key or --spec, "
            "or a judge model through",
            "--judge-model, before reading any of these figures as answer "
            "quality.",
            "",
        ]
    notes += [
        "Prompt adherence is decidable and needs no calibration: each check "
        "either holds of the response",
        "or does not. It is the strongest evidence in this report, and it "
        "measures obedience to the",
        "prompt rather than usefulness to the user.",
        "",
        "Coverage is a screening signal, not a relevance score. Restating the "
        "question scores well, and a",
        "correct answer to a closed question scores zero, which is why the echo "
        "ratio is reported next to",
        "it and why neither is read on its own.",
        "",
        "A baseline is the reference a change is measured against, not a claim "
        "that it is best: the",
        "highest weight precision, the largest model, the greedy decoding "
        "setting. A negative delta is",
        "therefore what the cheaper configuration costs.",
        "",
        "Significance is not importance. Every contrast reports an effect size "
        "and, where a margin was",
        "declared in advance, an equivalence test, so that \"too small to "
        "matter\" can be distinguished",
        "from \"not detected\". With 120 paired items a difference of a few "
        "milliseconds is detectable",
        "and irrelevant.",
        "",
        "Timings are wall-clock measurements on one shared host. They compare "
        "these runs with each other,",
        "not this hardware with other hardware, and the recognition stage is "
        "printed as the control for",
        "how much the host itself drifted between runs.",
    ]
    lines += ["  " + note if note else "" for note in notes]
    lines += [THICK]
    return lines


def _metric_row(pair: Any, key: str) -> Optional[Any]:
    return next((row for row in pair.metrics if row.metric.key == key), None)


def _delta(pair: Any, key: str, scale: float = 1.0, digits: int = 2) -> str:
    """One contrast's change in one metric, starred when it survives correction."""
    row = _metric_row(pair, key)
    if row is None or row.mean_difference is None:
        return "-"
    marker = "*" if row.significant else ""
    return f"{row.mean_difference * scale:+.{digits}f}{marker}"


def _family_counts(pair: Any) -> str:
    """How many metrics moved in each family, as "adh +2/-1 run 0/-8"."""
    parts = []
    for family in FAMILIES:
        improved = sum(1 for row in pair.metrics
                       if row.metric.family == family and row.verdict == "improved")
        degraded = sum(1 for row in pair.metrics
                       if row.metric.family == family and row.verdict == "degraded")
        if improved or degraded:
            parts.append(f"{family[:4]} +{improved}/-{degraded}")
    return "  ".join(parts) if parts else "nothing moved"


def _pct(value: Any) -> str:
    """Format a rate as a percentage, or '-' when it was not measured."""
    if value is None:
        return "-"
    try:
        return f"{100 * float(value):.1f}"
    except (TypeError, ValueError):
        return "-"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "-" if number != number else f"{number:.{digits}f}"


# ---------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.batch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--root", type=Path, required=True, help=(
        "Directory holding the runs. Searched recursively for directories with "
        "a transcripts.yaml or transcripts.jsonl."))
    parser.add_argument("--out-dir", type=Path, help=(
        "Where to write the results. Defaults to evaluation_result beside the "
        "root directory."))
    parser.add_argument("--group", action="append", default=[],
                        choices=GROUP_KINDS, help=(
        "Contrast to build. Repeatable; defaults to all of them."))

    parser.add_argument("--spec", type=Path, help=(
        "Scenario specification applied to every run, supplying reference "
        "answers and per-item expectations."))
    parser.add_argument("--answer-key", type=Path, help=(
        "Dataset metadata table (CSV) holding the corpus's own reference "
        "answers, applied to every run. Answerable items are scored against "
        "the 'answer' column, items marked is_impossible against "
        "'plausible_answers', and the two are reported separately."))
    parser.add_argument("--constraints", type=Path,
                        help="Constraint specification applied to every run.")
    parser.add_argument("--embedding-model", type=str, help=(
        "sentence-transformers model for embedding similarity, in every run."))
    parser.add_argument("--judge-model", action="append", default=[], help=(
        "Judge model applied to every run. Repeat for a panel. Costly: every "
        "run is graded."))
    parser.add_argument("--selfcheck-samples", type=int, default=0, help=(
        "Self-consistency resamples per item, in every run."))
    parser.add_argument("--no-latency", action="store_true", help=(
        "Do not read the latency logs, scoring responses without their timings."))
    parser.add_argument("--latency-warmup", type=int, help=(
        "Leading items to exclude from the timing aggregates of every run. "
        "Defaults to the convention each run recorded in its own "
        "log_averages.json, so that these figures and the run's published "
        "summary describe the same items. Quality scores are unaffected: a "
        "warm-up item is scored, only its timings are dropped."))

    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: %(default)s).")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap resamples per metric (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for bootstrap resampling (default: %(default)s).")
    parser.add_argument("--no-check-metrics", action="store_true",
                        help="Do not compare the individual constraint checks.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print the summary report to stdout.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .cli import _stderr_progress, _use_utf8_output

    _use_utf8_output()
    args = build_parser().parse_args(argv)

    settings = EvaluationConfig(
        spec=args.spec,
        answer_key=args.answer_key,
        embedding_model=args.embedding_model,
        judge_models=list(args.judge_model),
        selfcheck_samples=args.selfcheck_samples,
        include_latency=not args.no_latency,
        latency_warmup=args.latency_warmup,
        seed=args.seed,
        progress=None if args.quiet else _stderr_progress)
    if args.constraints:
        settings.constraints = args.constraints

    config = BatchConfig(
        root=args.root, out_dir=args.out_dir, evaluation=settings,
        alpha=args.alpha, n_boot=args.n_boot, seed=args.seed,
        include_checks=not args.no_check_metrics,
        group_kinds=list(args.group) if args.group else list(GROUP_KINDS),
        progress=None if args.quiet else _stderr_progress)

    try:
        outcome = run_batch(config)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    out_dir = config.resolved_out_dir()
    outcome.write(out_dir)

    if not args.quiet:
        print(outcome.report)
    for warning in outcome.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"\nArtefacts written to: {out_dir.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())





