#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Did changing a parameter improve the responses? Paired comparison of two runs.

Made for the case where the prompt set is too large to read: both configurations
answer the same inputs, and every metric that can be computed without a human is
compared item by item. Because the inputs are shared, the comparison is paired,
which removes the variation between prompts and leaves the effect of the change.

    python -m evaluation.comparison --baseline runs/q8 --contrast runs/q4

    from evaluation.comparison import ComparisonConfig, compare_runs
    outcome = compare_runs(ComparisonConfig(baseline="runs/q8",
                                            contrast=["runs/q4"]))

Three properties make the verdicts trustworthy at scale:

Direction is declared per metric, not inferred. A metric only counts as improved
or degraded when the direction of "better" is known; word count and sentence
count are reported but never scored, since neither longer nor shorter is
inherently right.

Significance is corrected within each family of metrics. Comparing twenty metrics
at the five percent level produces a false positive about two thirds of the time
if left uncorrected, so Holm's step-down procedure is applied (Holm, 1979) to
each family separately: prompt adherence, response content, runtime cost. The
families are declared in advance, because pooling them would let the number of
pipeline stages that happen to be logged decide how much evidence a claim about
response quality needs.

Size is separated from detectability. With thousands of items a difference of no
consequence still reaches significance, so every row carries an effect size and,
where a margin is declared, an equivalence test that can conclude "the change
made no practical difference" (Lakens, 2017).

Quality and cost are compared on the same items. The timings the pipeline logged
per recording are paired exactly like the scores, so "answers better" and "answers
later" appear in one table rather than in two that cannot be reconciled. The
recognition stages, which cannot depend on the language model, are reported as
controls: a difference there means the runs were measured under different host
conditions and the timing comparison is confounded.

The comparison needs no model and no human labels: it runs on the deterministic
tiers. Judge scores are included when both runs were evaluated with a judge.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePath
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import references
from .pipeline import EvaluationConfig, EvaluationOutcome, run_evaluation
from .aggregation import ItemResult
from .stats import (PAIRED_REFERENCE_KEYS, cliffs_delta, holm_bonferroni,
                    interpret_cliffs_delta, mcnemar, paired_mean_difference_ci,
                    paired_tost, wilcoxon_signed_rank)

PathLike = Union[str, Path]

HIGHER_IS_BETTER = "higher"
LOWER_IS_BETTER = "lower"
DESCRIPTIVE = "descriptive"

# Families of metric, corrected for multiplicity separately. A family is the set
# of tests that answer one question, and it is declared here rather than derived
# from the table: pooling the timings with the adherence checks would let the
# number of stages the pipeline happens to log decide how strong the evidence for
# a quality change has to be. Reported per family, so the correction applied to
# any figure is visible next to it.
FAMILIES = ["adherence", "response", "runtime"]

FAMILY_TITLES = {
    "adherence": "PROMPT ADHERENCE (verifiable, IFEval-style)",
    "response": "RESPONSE CONTENT AND FORM",
    "runtime": "RUNTIME COST (from the latency logs)",
}


@dataclass(frozen=True)
class Metric:
    """One comparable quantity, and what a change in it means."""

    key: str
    label: str
    direction: str
    binary: bool = False
    # Difference small enough not to matter, in the metric's own units. Used for
    # the equivalence test; None leaves that column empty rather than inventing
    # a threshold.
    margin: Optional[float] = None
    family: str = "response"

    @property
    def scored(self) -> bool:
        return self.direction in (HIGHER_IS_BETTER, LOWER_IS_BETTER)


# The comparable surface of an ItemResult. Margins are deliberately modest and
# can be overridden per run; they express what would be too small to act on.
METRICS: List[Metric] = [
    Metric("constraint_item_pass_strict", "prompt adherence (item)",
           HIGHER_IS_BETTER, binary=True, family="adherence"),
    Metric("constraint_item_pass_loose", "prompt adherence, loose",
           HIGHER_IS_BETTER, binary=True, family="adherence"),
    Metric("constraint_check_rate_strict", "share of checks passed",
           HIGHER_IS_BETTER, margin=0.02, family="adherence"),
    Metric("constraint_check_rate_loose", "share of checks passed, loose",
           HIGHER_IS_BETTER, margin=0.02, family="adherence"),

    Metric("request_coverage", "coverage of recognized request",
           HIGHER_IS_BETTER, margin=0.02),
    Metric("intent_coverage", "coverage of intended question",
           HIGHER_IS_BETTER, margin=0.02),
    Metric("echo_ratio", "echo of the question", LOWER_IS_BETTER, margin=0.02),
    Metric("answer_presence", "reference answer present", HIGHER_IS_BETTER,
           binary=True),
    Metric("reference_exact_match", "exact match vs reference", HIGHER_IS_BETTER,
           binary=True),
    Metric("reference_token_f1", "token F1 vs reference", HIGHER_IS_BETTER,
           margin=0.02),
    Metric("reference_rouge_1", "ROUGE-1 vs reference", HIGHER_IS_BETTER,
           margin=0.02),
    Metric("reference_rouge_l", "ROUGE-L vs reference", HIGHER_IS_BETTER,
           margin=0.02),

    Metric("flesch_reading_ease", "reading ease", HIGHER_IS_BETTER, margin=2.0),
    Metric("flesch_kincaid_grade", "grade level", LOWER_IS_BETTER, margin=0.5),

    Metric("word_count", "words", DESCRIPTIVE),
    Metric("sentence_count", "sentences", DESCRIPTIVE),
    Metric("opening_words", "opening sentence length", DESCRIPTIVE),
    Metric("mean_words_per_sentence", "words per sentence", DESCRIPTIVE),
    Metric("estimated_tokens", "estimated tokens", DESCRIPTIVE),
    Metric("estimated_spoken_seconds", "estimated speech seconds", DESCRIPTIVE),

    Metric("selfcheck_mean_inconsistency", "self-inconsistency (mean)",
           LOWER_IS_BETTER, margin=0.05),
    Metric("selfcheck_max_inconsistency", "self-inconsistency (worst sentence)",
           LOWER_IS_BETTER, margin=0.05),
    Metric("factprecision_precision", "atomic factual precision",
           HIGHER_IS_BETTER, margin=0.05),

    Metric("quality_composite", "rubric quality composite", HIGHER_IS_BETTER,
           margin=0.25),
    Metric("safety_minimum", "worst safety dimension", HIGHER_IS_BETTER,
           margin=0.25),

    # Runtime. Margins are in milliseconds and state what a listener would not
    # notice in a spoken exchange; the throughput margin is in tokens a second.
    Metric("lat_llm_prompt_eval", "prompt evaluation (ms)", LOWER_IS_BETTER,
           margin=5.0, family="runtime"),
    Metric("lat_llm_ttft", "time to first token (ms)", LOWER_IS_BETTER,
           margin=50.0, family="runtime"),
    Metric("lat_llm_ttfc", "time to first speakable chunk (ms)",
           LOWER_IS_BETTER, margin=50.0, family="runtime"),
    Metric("lat_llm_eval", "generation time (ms)", LOWER_IS_BETTER,
           margin=50.0, family="runtime"),
    Metric("lat_tts_first_chunk", "first speech chunk synthesis (ms)",
           LOWER_IS_BETTER, margin=50.0, family="runtime"),
    Metric("lat_ttfa", "time to first audio (ms)", LOWER_IS_BETTER,
           margin=100.0, family="runtime"),
    Metric("lat_tts_total", "total synthesis (ms)", LOWER_IS_BETTER,
           margin=50.0, family="runtime"),
    Metric("lat_e2e_response_ready", "end to end, response ready (ms)",
           LOWER_IS_BETTER, margin=100.0, family="runtime"),
    Metric("llm_tokens_per_sec", "generation throughput (tok/s)",
           HIGHER_IS_BETTER, margin=5.0, family="runtime"),

    # Controls and volumes. The recognition stages cannot depend on the language
    # model: a difference in them means the two runs were measured under
    # different host conditions, which is why they are reported and never scored.
    Metric("lat_stt", "recognition, control (ms)", DESCRIPTIVE,
           family="runtime"),
    Metric("lat_stt_endpoint_delay", "endpoint delay, control (ms)",
           DESCRIPTIVE, family="runtime"),
    Metric("lat_llm_first_chunk_fill", "first chunk fill wait (ms)",
           DESCRIPTIVE, family="runtime"),
    Metric("llm_prompt_tokens", "prompt tokens", DESCRIPTIVE, family="runtime"),
    Metric("llm_eval_tokens", "generated tokens", DESCRIPTIVE,
           family="runtime"),
    Metric("tts_audio_ms", "synthesized audio (ms)", DESCRIPTIVE,
           family="runtime"),
]


@dataclass
class ComparisonConfig:
    """What to compare, and how strict to be about calling a difference real.

    The run paths may be left unset when the runs have already been evaluated and
    only the comparison settings are needed; see `compare_evaluated`.
    """

    baseline: Optional[PathLike] = None
    contrast: List[PathLike] = field(default_factory=list)

    # Applied to every run, so both sides are measured identically. Anything the
    # single-run evaluation accepts can be set here except the input paths.
    evaluation: Optional[EvaluationConfig] = None

    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0

    # Only compare per-check pass rates for checks present in both runs.
    include_checks: bool = True

    progress: Optional[Callable[[str], None]] = None

    def __post_init__(self):
        if isinstance(self.contrast, (str, Path)):
            self.contrast = [self.contrast]
        if self.baseline is not None and not self.contrast:
            raise ValueError("provide at least one contrast run")

    def notify(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


@dataclass
class MetricComparison:
    """The paired comparison of one metric between two runs."""

    metric: Metric
    n_pairs: int
    baseline_mean: Optional[float]
    contrast_mean: Optional[float]
    mean_difference: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    improved: int
    degraded: int
    unchanged: int
    effect_size: Optional[float]
    effect_label: str
    p_value: Optional[float]
    test_method: str
    n_effective: int
    p_adjusted: Optional[float] = None
    significant: bool = False
    equivalent: Optional[bool] = None
    verdict: str = "not assessed"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.key,
            "label": self.metric.label,
            "family": self.metric.family,
            "direction": self.metric.direction,
            "n_pairs": self.n_pairs,
            "baseline_mean": _r(self.baseline_mean),
            "contrast_mean": _r(self.contrast_mean),
            "mean_difference": _r(self.mean_difference),
            "ci_low": _r(self.ci_low),
            "ci_high": _r(self.ci_high),
            "improved_items": self.improved,
            "degraded_items": self.degraded,
            "unchanged_items": self.unchanged,
            "effect_size": _r(self.effect_size),
            "effect_label": self.effect_label,
            "p_value": _p(self.p_value),
            "p_adjusted": _p(self.p_adjusted),
            "significant": self.significant,
            "equivalent_within_margin": self.equivalent,
            "equivalence_margin": self.metric.margin,
            "test_method": self.test_method,
            "n_effective": self.n_effective,
            "verdict": self.verdict,
        }


@dataclass
class RunPairComparison:
    """Everything comparing one contrast run against the baseline produced."""

    baseline_label: str
    contrast_label: str
    n_paired: int
    n_baseline_only: int
    n_contrast_only: int
    metrics: List[MetricComparison] = field(default_factory=list)

    @property
    def improvements(self) -> List[MetricComparison]:
        return [m for m in self.metrics if m.verdict == "improved"]

    @property
    def regressions(self) -> List[MetricComparison]:
        return [m for m in self.metrics if m.verdict == "degraded"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_label,
            "contrast": self.contrast_label,
            "n_paired": self.n_paired,
            "n_baseline_only": self.n_baseline_only,
            "n_contrast_only": self.n_contrast_only,
            "metrics": [m.as_dict() for m in self.metrics],
            "improved": [m.metric.key for m in self.improvements],
            "degraded": [m.metric.key for m in self.regressions],
        }


@dataclass
class ComparisonOutcome:
    """The result of comparing one baseline against one or more contrasts."""

    baseline: EvaluationOutcome
    contrasts: List[EvaluationOutcome]
    pairs: List[RunPairComparison]
    alpha: float
    report: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "alpha": self.alpha,
            "baseline": {
                "label": self.baseline.label,
                "run_dir": (str(self.baseline.context.run_dir)
                            if self.baseline.context.run_dir else None),
                "config_used": self.baseline.context.config,
            },
            "contrasts": [
                {"label": outcome.label,
                 "run_dir": (str(outcome.context.run_dir)
                             if outcome.context.run_dir else None),
                 "config_used": outcome.context.config}
                for outcome in self.contrasts],
            "comparisons": [pair.as_dict() for pair in self.pairs],
            "method_references": [
                ref._asdict()
                for ref in references.resolve(PAIRED_REFERENCE_KEYS)],
        }

    def write(self, out_dir: PathLike) -> Path:
        from . import reporting

        target = Path(out_dir)
        reporting.write_text(target / "comparison_report.txt", self.report)
        reporting.write_json(target / "comparison_results.json", self.as_dict())
        _write_metric_csv(target / "comparison_metrics.csv", self.pairs)
        return target


def compare_runs(config: ComparisonConfig) -> ComparisonOutcome:
    """Evaluate every run with identical settings, then compare them pairwise."""
    if config.baseline is None or not config.contrast:
        raise ValueError("provide a baseline run and at least one contrast run")

    base_settings = config.evaluation or EvaluationConfig()

    config.notify(f"evaluating baseline {config.baseline}")
    baseline = run_evaluation(_settings_for(base_settings, config.baseline))

    contrasts: List[EvaluationOutcome] = []
    for run_dir in config.contrast:
        config.notify(f"evaluating contrast {run_dir}")
        contrasts.append(run_evaluation(_settings_for(base_settings, run_dir)))

    return compare_evaluated(baseline, contrasts, config)


def compare_evaluated(baseline: EvaluationOutcome,
                      contrasts: Sequence[EvaluationOutcome],
                      config: Optional[ComparisonConfig] = None
                      ) -> ComparisonOutcome:
    """Compare runs that have already been evaluated.

    Separated from `compare_runs` so that a run appearing in several contrasts is
    evaluated once. Scoring a run again per comparison would be wasted work, and
    with a sampled tier it would also produce a run whose scores differ between
    the tables it appears in.

    The caller is responsible for having evaluated every run with the same
    settings; `compare_runs` guarantees that, this function trusts it.
    """
    settings = config or ComparisonConfig()
    contrasts = list(contrasts)

    pairs = []
    for contrast in contrasts:
        settings.notify(f"comparing {contrast.label} against {baseline.label}")
        pairs.append(_compare_pair(baseline, contrast, settings))

    outcome = ComparisonOutcome(baseline=baseline, contrasts=contrasts,
                                pairs=pairs, alpha=settings.alpha)
    outcome.report = render_comparison_report(outcome)
    return outcome


def _settings_for(template: EvaluationConfig, run_dir: PathLike
                  ) -> EvaluationConfig:
    """Copy the shared settings and point them at one run.

    Both sides must be measured with the same constraint spec, scenario spec and
    judge configuration, or the difference would partly reflect the measurement
    rather than the runs.
    """
    from dataclasses import replace

    return replace(template, run_dir=Path(run_dir), transcripts=None, label=None)


def _compare_pair(baseline: EvaluationOutcome, contrast: EvaluationOutcome,
                  config: ComparisonConfig) -> RunPairComparison:
    paired, base_only, contrast_only = _pair_items(baseline.results,
                                                   contrast.results)

    comparison = RunPairComparison(
        baseline_label=baseline.label, contrast_label=contrast.label,
        n_paired=len(paired), n_baseline_only=base_only,
        n_contrast_only=contrast_only)
    if not paired:
        return comparison

    metrics = list(METRICS)
    if config.include_checks:
        metrics.extend(_check_metrics(baseline, contrast))

    rows: List[MetricComparison] = []
    for metric in metrics:
        row = _compare_metric(metric, paired, config)
        if row is not None:
            rows.append(row)

    _apply_multiplicity_correction(rows, config.alpha)
    comparison.metrics = rows
    return comparison


def _pair_items(baseline: Sequence[ItemResult], contrast: Sequence[ItemResult]
                ) -> Tuple[List[Tuple[ItemResult, ItemResult]], int, int]:
    """Match items across runs by the recording, or by the utterance answered.

    Matching on the item rather than on position means the comparison survives a
    reordered or partially failed run; an item present on one side only is
    counted and excluded rather than silently aligned with a neighbour.

    The recording is the identity where both runs name one, because two runs may
    have heard the same recording differently: a contrast that varies the
    recognizer has the same inputs but not the same transcripts, and matching on
    the transcript would discard exactly those items the recognizers disagree on,
    which are the items such a contrast is about. Where filenames are absent the
    utterance is the only available identity and is used instead.

    A prompt repeated within a run, which is how the variability of a sampling
    configuration is usually measured, is matched occurrence by occurrence: the
    k-th repetition on one side against the k-th on the other. The pairing within
    a repeated prompt is arbitrary, but the repetitions are exchangeable, so it
    does not bias the comparison.
    """
    by_recording = (all(result.record.filename for result in baseline)
                    and all(result.record.filename for result in contrast))

    def key(result: ItemResult) -> str:
        if by_recording:
            return PurePath(result.record.filename.strip()).name.lower()
        text = " ".join(result.record.stt_text.lower().split())
        return text or f"#{result.item_id}"

    contrast_index: Dict[str, List[ItemResult]] = {}
    for result in contrast:
        contrast_index.setdefault(key(result), []).append(result)

    taken: Dict[str, int] = {}
    pairs = []
    for result in baseline:
        item_key = key(result)
        position = taken.get(item_key, 0)
        candidates = contrast_index.get(item_key, ())
        if position < len(candidates):
            pairs.append((result, candidates[position]))
            taken[item_key] = position + 1

    return pairs, len(baseline) - len(pairs), len(contrast) - len(pairs)


def _check_metrics(baseline: EvaluationOutcome,
                   contrast: EvaluationOutcome) -> List[Metric]:
    """One binary metric per constraint check the two runs share."""
    def check_ids(outcome: EvaluationOutcome) -> Dict[str, str]:
        ids = {}
        for result in outcome.results:
            if result.constraints is None:
                continue
            for check in result.constraints.results:
                ids.setdefault(check.check_id, check.check_id)
        return ids

    shared = sorted(set(check_ids(baseline)) & set(check_ids(contrast)))
    return [Metric(f"check::{check_id}", f"check: {check_id}",
                   HIGHER_IS_BETTER, binary=True, family="adherence")
            for check_id in shared]


def _compare_metric(metric: Metric,
                    paired: Sequence[Tuple[ItemResult, ItemResult]],
                    config: ComparisonConfig) -> Optional[MetricComparison]:
    base_values: List[float] = []
    contrast_values: List[float] = []

    for before, after in paired:
        left = _metric_value(metric, before)
        right = _metric_value(metric, after)
        if left is None or right is None:
            continue
        base_values.append(float(left))
        contrast_values.append(float(right))

    if not base_values:
        return None

    differences = [after - before
                   for before, after in zip(base_values, contrast_values)]
    improved, degraded = _count_changes(metric, base_values, contrast_values)
    unchanged = len(differences) - improved - degraded

    ci_low, ci_high = paired_mean_difference_ci(
        differences, n_boot=config.n_boot, alpha=config.alpha, seed=config.seed)

    if metric.binary:
        test = mcnemar([bool(v) for v in base_values],
                       [bool(v) for v in contrast_values])
        effect = None
        effect_label = "n/a"
    else:
        test = wilcoxon_signed_rank(differences)
        effect = cliffs_delta(contrast_values, base_values)
        effect_label = interpret_cliffs_delta(effect)

    equivalence = None
    if metric.margin is not None and not metric.binary:
        result = paired_tost(differences, metric.margin, alpha=config.alpha)
        equivalence = result.equivalent if result else None

    return MetricComparison(
        metric=metric,
        n_pairs=len(differences),
        baseline_mean=sum(base_values) / len(base_values),
        contrast_mean=sum(contrast_values) / len(contrast_values),
        mean_difference=sum(differences) / len(differences),
        ci_low=ci_low, ci_high=ci_high,
        improved=improved, degraded=degraded, unchanged=unchanged,
        effect_size=effect, effect_label=effect_label,
        p_value=test.p_value if test else None,
        test_method=test.method if test else "not tested",
        n_effective=test.n_effective if test else 0,
        equivalent=equivalence)


def _metric_value(metric: Metric, result: ItemResult) -> Optional[float]:
    """Read one metric off an evaluated item, or None when it was not measured."""
    if metric.key.startswith("check::"):
        if result.constraints is None:
            return None
        check_id = metric.key.split("::", 1)[1]
        for check in result.constraints.results:
            if check.check_id == check_id:
                if not check.applicable:
                    return None
                return float(check.passed_strict)
        return None

    if metric.key in ("quality_composite", "safety_minimum"):
        return getattr(result, metric.key)

    if result.constraints is not None:
        if metric.key == "constraint_item_pass_strict":
            return float(result.constraints.item_level_strict)
        if metric.key == "constraint_item_pass_loose":
            return float(result.constraints.item_level_loose)
        if metric.key == "constraint_check_rate_strict":
            return result.constraints.check_level_strict
        if metric.key == "constraint_check_rate_loose":
            return result.constraints.check_level_loose
        if metric.key in result.constraints.measures:
            return _as_float(result.constraints.measures[metric.key])

    if metric.key in result.relevance:
        return _as_float(result.relevance[metric.key])
    if metric.key in result.latency:
        return _as_float(result.latency[metric.key])
    if metric.key in result.asr:
        return _as_float(result.asr[metric.key])
    if metric.key in result.factuality:
        return _as_float(result.factuality[metric.key])
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_changes(metric: Metric, base: Sequence[float],
                   contrast: Sequence[float]) -> Tuple[int, int]:
    """How many items got better and how many got worse.

    A descriptive metric has no better, so both counts stay at zero rather than
    asserting a direction the metric does not have.
    """
    if not metric.scored:
        return 0, 0
    sign = 1.0 if metric.direction == HIGHER_IS_BETTER else -1.0
    improved = sum(1 for b, c in zip(base, contrast) if (c - b) * sign > 0)
    degraded = sum(1 for b, c in zip(base, contrast) if (c - b) * sign < 0)
    return improved, degraded


def _apply_multiplicity_correction(rows: List[MetricComparison],
                                   alpha: float) -> None:
    """Holm-correct within each metric family and set each verdict.

    Correction runs per family rather than over the whole table. A descriptive row
    makes no claim and is excluded, since including it would tighten the
    correction for no reason.
    """
    families: Dict[str, List[MetricComparison]] = {}
    for row in rows:
        if row.metric.scored and row.p_value is not None:
            families.setdefault(row.metric.family, []).append(row)

    for scored in families.values():
        decisions = holm_bonferroni([row.p_value for row in scored], alpha=alpha)
        for row, decision in zip(scored, decisions):
            row.p_adjusted = float(decision["p_holm"])
            row.significant = bool(decision["reject"])

    for row in rows:
        row.verdict = _verdict_for(row)


def _verdict_for(row: MetricComparison) -> str:
    if not row.metric.scored:
        return "descriptive"
    if row.p_value is None:
        return "not tested"
    if row.significant:
        sign = 1.0 if row.metric.direction == HIGHER_IS_BETTER else -1.0
        if row.mean_difference is None or row.mean_difference == 0:
            return "changed"
        return "improved" if row.mean_difference * sign > 0 else "degraded"
    if row.equivalent:
        return "equivalent"
    return "no detected change"


# ---------------------------------------------------------------- reporting

THICK = "=" * 132
THIN = "-" * 132


def render_comparison_report(outcome: ComparisonOutcome) -> str:
    lines: List[str] = [THICK, "PAIRED RUN COMPARISON".center(len(THICK)), THICK]
    lines.append(f"Generated on           : "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Baseline               : {outcome.baseline.label}")
    for contrast in outcome.contrasts:
        lines.append(f"Contrast               : {contrast.label}")
    lines.append(f"Significance level     : {outcome.alpha} "
                 f"(Holm-corrected within each metric family)")
    lines.append("")
    lines.extend(_configuration_diff(outcome))
    lines.append(THICK)

    for index, pair in enumerate(outcome.pairs, 1):
        lines.extend(_render_pair(index, pair))

    lines.append("")
    lines.append("METHODS")
    lines.append(THICK)
    lines.extend(references.bibliography_lines(PAIRED_REFERENCE_KEYS))
    lines.append(THICK)
    lines.append("")
    lines.append("HOW TO READ THIS")
    lines.append(THICK)
    lines.extend(_interpretation_notes())
    lines.append(THICK)
    return "\n".join(lines)


# Entries that name a run's own files rather than how it was configured. Every
# pair of runs differs on them by construction, and listing them buries the one
# setting the comparison is actually about.
BOOKKEEPING_KEYS = frozenset({"config", "out_dir", "output_dir", "run_dir",
                              "log_dir", "latency_csv", "transcripts",
                              "timestamp", "run_id"})


def _configuration_diff(outcome: ComparisonOutcome) -> List[str]:
    """Show which generation settings actually differ between the runs.

    Naming the changed parameter is the point of the exercise: a table of
    differences means nothing if the reader cannot see what was changed.
    """
    lines = ["Generating configuration differences:"]
    base_config = outcome.baseline.context.config or {}
    any_difference = False
    omitted = 0

    for contrast in outcome.contrasts:
        other = contrast.context.config or {}
        keys = sorted(set(base_config) | set(other))
        changed = [(k, base_config.get(k), other.get(k)) for k in keys
                   if base_config.get(k) != other.get(k)]
        omitted += sum(1 for key, _, _ in changed if key in BOOKKEEPING_KEYS)
        changed = [entry for entry in changed
                   if entry[0] not in BOOKKEEPING_KEYS]
        if not changed:
            lines.append(f"  {contrast.label}: no recorded difference "
                         f"in config_used.yaml")
            continue
        any_difference = True
        lines.append(f"  {contrast.label}:")
        for key, before, after in changed:
            lines.append(f"    {key:<20}: {before} -> {after}")

    if omitted:
        lines.append("  (Output paths and run identifiers differ by "
                     "construction and are not listed.)")
    if not any_difference:
        lines.append("  WARNING: the runs record identical settings, so any "
                     "difference below is sampling noise or nondeterminism.")
    return lines


def _render_pair(index: int, pair: RunPairComparison) -> List[str]:
    lines = ["", f"{index}. {pair.contrast_label}  vs  {pair.baseline_label}",
             THICK]
    lines.append(f"Paired items: {pair.n_paired}"
                 + (f"   (unmatched: {pair.n_baseline_only} in baseline, "
                    f"{pair.n_contrast_only} in contrast)"
                    if pair.n_baseline_only or pair.n_contrast_only else ""))
    lines.append("")

    if not pair.metrics:
        lines.append("No comparable metric was measured in both runs.")
        lines.append(THICK)
        return lines

    header = (f"{'Metric':<34}{'base':>11}{'contrast':>11}{'delta':>11}"
              f"{'95% CI':>24}{'better/worse':>14}{'effect':>18}"
              f"{'p(adj)':>9}  verdict")

    grouped: Dict[str, List[MetricComparison]] = {}
    for row in pair.metrics:
        grouped.setdefault(row.metric.family, []).append(row)

    ordered_families = [f for f in FAMILIES if f in grouped]
    ordered_families += sorted(f for f in grouped if f not in FAMILIES)

    for family in ordered_families:
        rows = grouped[family]
        lines.append(FAMILY_TITLES.get(family, family.upper()))
        lines.append(header)
        lines.append(THIN)
        for row in sorted(rows, key=_row_sort_key):
            lines.append(_format_row(row))
        tested = sum(1 for row in rows
                     if row.metric.scored and row.p_value is not None)
        lines.append(f"  Holm correction within this family, over {tested} "
                     f"scored test(s).")
        lines.append("")

    lines.append(_summary_sentence(pair))
    lines.append(THICK)
    return lines


def _row_sort_key(row: MetricComparison) -> tuple:
    """Show what changed first, and the descriptive rows last."""
    order = {"degraded": 0, "improved": 1, "equivalent": 2,
             "no detected change": 3, "changed": 4, "not tested": 5,
             "descriptive": 6}
    return (order.get(row.verdict, 9), row.metric.key)


def _format_row(row: MetricComparison) -> str:
    interval = ("" if row.ci_low is None
                else f"[{_num(row.ci_low, signed=True)}, "
                     f"{_num(row.ci_high, signed=True)}]")
    counts = (f"{row.improved}/{row.degraded}" if row.metric.scored else "")
    p_text = ("" if row.p_adjusted is None
              else ("<0.001" if row.p_adjusted < 0.001 else f"{row.p_adjusted:.3f}"))
    # A binary metric leaves this empty: the change in pass rate shown as delta
    # is already the effect size, and Cliff's delta on two values adds nothing.
    effect = ("" if row.effect_size is None
              else f"{row.effect_size:+.2f} {row.effect_label}")
    return (f"{row.metric.label[:33]:<34}"
            f"{_num(row.baseline_mean):>11}"
            f"{_num(row.contrast_mean):>11}"
            f"{_num(row.mean_difference, signed=True):>11}"
            f"{interval:>24}"
            f"{counts:>14}"
            f"{effect:>18}"
            f"{p_text:>9}  {row.verdict}")


def _num(value: Optional[float], signed: bool = False) -> str:
    """Format a value with a precision suited to its magnitude.

    A millisecond timing printed to three decimals is false precision that also
    collides with the next column, while a pass rate needs all three.
    """
    if value is None:
        return "-"
    magnitude = abs(value)
    digits = 0 if magnitude >= 1000 else (1 if magnitude >= 100 else 3)
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _summary_sentence(pair: RunPairComparison) -> str:
    """One line per family naming what moved, so the table has a readable verdict."""
    if not pair.improvements and not pair.regressions:
        return ("VERDICT: no metric changed detectably after correction for "
                "multiple comparisons.")

    families: Dict[str, Dict[str, List[str]]] = {}
    for row in pair.improvements:
        families.setdefault(row.metric.family, {}).setdefault(
            "improved", []).append(row.metric.label)
    for row in pair.regressions:
        families.setdefault(row.metric.family, {}).setdefault(
            "degraded", []).append(row.metric.label)

    lines = ["VERDICT after Holm correction within each family:"]
    order = {name: index for index, name in enumerate(FAMILIES)}
    for family in sorted(families, key=lambda name: order.get(name, 99)):
        moved = families[family]
        parts = []
        if moved.get("improved"):
            parts.append(f"improved: {', '.join(moved['improved'])}")
        if moved.get("degraded"):
            parts.append(f"DEGRADED: {', '.join(moved['degraded'])}")
        lines.append(f"  {family:<10} " + "; ".join(parts))
    return "\n".join(lines)


def _interpretation_notes() -> List[str]:
    return [
        "  delta is the mean paired difference, contrast minus baseline, in the metric's",
        "  own units. The interval is a percentile bootstrap over item pairs; an interval",
        "  that excludes zero and a significant p-value carry the same information, and",
        "  the interval also states how large the change is.",
        "",
        "  better/worse counts the items that moved in each direction, which distinguishes",
        "  a small shift in every item from a large shift in a few. Read it before the",
        "  p-value: a mean improvement produced by a handful of items is a different",
        "  finding from a uniform one.",
        "",
        "  effect is Cliff's delta with its conventional band. It is the probability that a",
        "  contrast response scores above a baseline one, minus the reverse, so it is",
        "  comparable across metrics measured in different units. It is left empty for a",
        "  pass-or-fail metric, where the change in pass rate shown as delta is itself the",
        "  effect size.",
        "",
        "  p(adj) is Holm-corrected within the metric family it sits in, not across the",
        "  whole table. Descriptive rows, such as word count and the recognition stages, are",
        "  excluded from the correction and never receive a verdict, since neither direction",
        "  is better for them.",
        "",
        "  In the runtime family, recognition and endpoint delay are controls: the same audio",
        "  was decoded by the same recognizer in both runs, so these should not move. If they",
        "  do, the host was not in the same state and the other timings carry that difference",
        "  too. Generation time also tracks how much the model chose to say, so read it next",
        "  to the token count and the throughput rather than on its own.",
        "",
        "  'equivalent' means the whole confidence interval for the difference lies inside",
        "  the declared margin: the change is not merely undetected, it is too small to",
        "  matter. 'no detected change' is weaker and includes an underpowered test.",
        "",
        "  Significance is not importance. With a large prompt set a difference of no",
        "  practical consequence still reaches significance, which is why every row",
        "  reports an effect size and, where a margin was declared, an equivalence test.",
        "",
        "  Constraint metrics are decidable and need no calibration. Judge-derived rows",
        "  inherit the limits of the judge and should not settle a close comparison on",
        "  their own.",
    ]


def _write_metric_csv(path: Path, pairs: Sequence[RunPairComparison]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(baseline=pair.baseline_label, contrast=pair.contrast_label,
                 **metric.as_dict())
            for pair in pairs for metric in pair.metrics]
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _p(value: Optional[float]) -> Optional[float]:
    """Round a p-value to significant digits, not decimal places.

    A decisive test on a large sample produces a p-value far below any decimal
    rounding, and reporting it as exactly zero would overstate what a test can
    show.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == 0.0:
        return 0.0
    from math import floor, log10
    return round(number, -int(floor(log10(abs(number)))) + 2)


# ---------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.comparison",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--baseline", type=Path, required=True,
                        help="Run directory to compare against.")
    parser.add_argument("--contrast", type=Path, action="append", required=True,
                        help=("Run directory produced with the changed setting. "
                              "Repeat to compare several against one baseline."))
    parser.add_argument("--spec", type=Path,
                        help="Scenario specification applied to every run.")
    parser.add_argument("--constraints", type=Path,
                        help="Constraint specification applied to every run.")
    parser.add_argument("--judge-model", action="append", default=[],
                        help=("Judge model, applied to every run. Repeat for a "
                              "panel. Costly: every run is graded."))
    parser.add_argument("--judge-url", type=str,
                        help="Ollama generate endpoint.")
    parser.add_argument("--selfcheck-samples", type=int, default=0,
                        help="Self-consistency resamples per item, in every run.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: %(default)s).")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap resamples per metric (default: %(default)s).")
    parser.add_argument("--no-check-metrics", action="store_true",
                        help="Do not compare the individual constraint checks.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for bootstrap resampling (default: %(default)s).")
    parser.add_argument("--out-dir", type=Path,
                        help="Output directory (default: ./comparison_output).")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print the report to stdout.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .cli import _stderr_progress, _use_utf8_output

    _use_utf8_output()
    args = build_parser().parse_args(argv)

    settings = EvaluationConfig(
        spec=args.spec,
        judge_models=list(args.judge_model),
        selfcheck_samples=args.selfcheck_samples,
        seed=args.seed,
        progress=None if args.quiet else _stderr_progress)
    if args.constraints:
        settings.constraints = args.constraints
    if args.judge_url:
        settings.judge_url = args.judge_url

    config = ComparisonConfig(
        baseline=args.baseline, contrast=list(args.contrast),
        evaluation=settings, alpha=args.alpha, n_boot=args.n_boot,
        seed=args.seed, include_checks=not args.no_check_metrics,
        progress=None if args.quiet else _stderr_progress)

    try:
        outcome = compare_runs(config)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path("comparison_output")
    outcome.write(out_dir)

    if not args.quiet:
        print(outcome.report)
    print(f"\nArtefacts written to: {out_dir.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
