#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programmatic entry point: configure an evaluation, run it, get the results back.

This is the same orchestration the command line performs, expressed as a
function so that a notebook or a batch script can call it without assembling an
argument vector and without parsing a report back into numbers.

    from evaluation import EvaluationConfig, run_evaluation

    outcome = run_evaluation(EvaluationConfig(run_dir="outputs/20260809_164356"))
    print(outcome.summary.acceptance_rate)
    outcome.write("outputs/20260809_164356/evaluation")

Nothing here writes to disk or prints unless asked: `run_evaluation` returns the
outcome, `outcome.write()` persists it, and progress is reported only through the
callback the caller supplies. That separation is what makes the toolkit usable
as a library rather than only as a program.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from . import references, reporting
from .aggregation import (AcceptancePolicy, ItemResult, RunSummary, score_item,
                          summarize_run)
from .agreement import (REFERENCE_KEYS as AGREEMENT_REFS, agreement_report,
                        calibrate_judge, load_human_annotations)
from .asr import (REFERENCE_KEYS as ASR_REFS, corpus_error_rate,
                  evaluate_asr_fidelity)
from .constraints import (REFERENCE_KEYS as CONSTRAINT_REFS, ConstraintSpec,
                          evaluate_constraints)
from .factuality import (REFERENCE_KEYS as FACT_REFS, extract_audit_atoms,
                         run_fact_precision, run_selfcheck)
from .judge import Rubric, build_panel, compare_pairwise
from .latency import (REFERENCE_KEYS as LATENCY_REFS, RunLatency, item_key,
                      load_run_latency, summarize_stages)
from .loaders import (Record, RunContext, build_records, discover_run,
                      load_answer_key, load_scenario_spec, load_transcripts)
from .ollama_client import DEFAULT_URL, OllamaClient
from .relevance import (REFERENCE_KEYS as RELEVANCE_REFS, EmbeddingBackend,
                        build_idf_index, evaluate_relevance)
from .stats import REFERENCE_KEYS as STATS_REFS

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONSTRAINTS = PACKAGE_DIR / "rubrics" / "constraints_short_opener.yaml"
DEFAULT_RUBRIC = PACKAGE_DIR / "rubrics" / "quest_therapy_v1.yaml"

# Dimensions used for pairwise configuration contrasts when none are named.
DEFAULT_PAIRWISE_DIMENSIONS = ["relevance", "spoken_comprehensibility",
                               "factual_accuracy"]

PathLike = Union[str, Path, None]


def _as_path(value: PathLike) -> Optional[Path]:
    """Accept strings as well as Path objects, since callers pass both."""
    return Path(value) if value is not None else None


@dataclass
class EvaluationConfig:
    """Everything an evaluation run needs, with the defaults the CLI uses.

    Only `run_dir` or `transcripts` is required. Every tier beyond the ones that
    need no model stays off until it is configured, so a bare configuration is
    the cheap, offline, always-valid evaluation.
    """

    # Input
    run_dir: PathLike = None
    transcripts: PathLike = None
    spec: PathLike = None
    # Dataset metadata table carrying the corpus's own reference answers. Read
    # instead of, or alongside, a hand-written spec; see loaders.load_answer_key.
    answer_key: PathLike = None
    system_prompt: PathLike = None
    label: Optional[str] = None

    # Tier 0
    constraints: PathLike = DEFAULT_CONSTRAINTS
    max_tokens: Optional[int] = None

    # Tier 1
    embedding_model: Optional[str] = None

    # Runtime cost. Read from the run directory's latency log; disable to score
    # responses without their timings, for a transcript that has none.
    include_latency: bool = True
    # Leading items to exclude as warm-up. None follows the run's own aggregate.
    latency_warmup: Optional[int] = None

    # Tier 2
    selfcheck_samples: int = 0
    selfcheck_model: Optional[str] = None
    selfcheck_temperature: float = 0.8
    fact_precision: bool = False
    max_claims: int = 10

    # Tier 3
    judge_models: List[str] = field(default_factory=list)
    judge_url: str = DEFAULT_URL
    judge_samples: int = 1
    judge_timeout: float = 300.0
    rubric: PathLike = DEFAULT_RUBRIC

    # Configuration contrast
    compare_run_dir: PathLike = None
    pairwise_dimensions: List[str] = field(default_factory=list)

    # Tier 4
    human_annotations: PathLike = None

    # Acceptance policy: predeclare these before measuring.
    min_quality: float = 3.5
    min_safety: float = 4.0
    loose_constraints: bool = False
    constraint_gate: bool = True

    seed: int = 0

    # Called with a one-line status while the model-backed tiers run, which take
    # minutes. Left unset the run is silent, as a library call should be.
    progress: Optional[Callable[[str], None]] = None

    def __post_init__(self):
        for name in ("run_dir", "transcripts", "spec", "answer_key",
                     "system_prompt", "constraints", "rubric", "compare_run_dir",
                     "human_annotations"):
            setattr(self, name, _as_path(getattr(self, name)))
        if isinstance(self.judge_models, str):
            self.judge_models = [self.judge_models]

    def notify(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


@dataclass
class EvaluationOutcome:
    """Everything one evaluation produced, in memory.

    Holds the objects rather than only the rendered report, so that a caller can
    read a single number without re-parsing text, while `write()` still produces
    the same artefacts the command line does.
    """

    label: str
    context: RunContext
    spec: ConstraintSpec
    rubric: Optional[Rubric]
    policy: AcceptancePolicy
    results: List[ItemResult]
    summary: RunSummary
    tiers: Dict[str, str]
    reference_keys: List[str]
    report: str
    agreement: Optional[List[Dict[str, Any]]] = None
    calibration: Optional[List[Dict[str, Any]]] = None
    judge_cost: Optional[List[Dict[str, Any]]] = None
    pairwise: Optional[List[Dict[str, Any]]] = None
    latency: Optional[RunLatency] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """The structured payload written to evaluation_results.json."""
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run": {
                "label": self.label,
                "run_dir": (str(self.context.run_dir)
                            if self.context.run_dir else None),
                "transcripts": (str(self.context.transcripts_path)
                                if self.context.transcripts_path else None),
                "latency_log": (str(self.latency.path)
                                if self.latency and self.latency.path else None),
                "latency_warmup_items": (self.latency.warmup_items
                                         if self.latency else []),
                # The generating configuration travels with the scores: a results
                # table that has been separated from the settings that produced it
                # cannot be attributed to a model or a parameter value.
                "generation_settings": self.context.generation_settings,
                "config_used": self.context.config,
            },
            "constraint_spec": self.spec.spec_id,
            "rubric": self.rubric.rubric_id if self.rubric else None,
            "tiers": self.tiers,
            "acceptance_policy": self.policy.as_dict(),
            "summary": self.summary.as_dict(),
            "items": [result.flat_row(self.rubric) for result in self.results],
            "agreement": self.agreement or [],
            "judge_calibration": self.calibration or [],
            "pairwise": self.pairwise or [],
            "judge_cost": self.judge_cost or [],
            "warnings": self.warnings,
            "method_references": [ref._asdict() for ref
                                  in references.resolve(self.reference_keys)],
        }

    def write(self, out_dir: PathLike, emit_bibtex: bool = False,
              copy_run_artefacts: bool = True) -> Path:
        """Write the report, the per-item CSV and the JSON to `out_dir`.

        The run's own `config_used.yaml` and `system_prompt.txt` are copied in
        verbatim, so that a results directory states which model, decoding
        settings and prompt produced the responses without a reader having to
        find the original run.
        """
        target = Path(out_dir)
        reporting.write_text(target / "evaluation_report.txt", self.report)
        reporting.write_item_csv(target / "evaluation_items.csv",
                                 self.results, self.rubric)
        reporting.write_json(target / "evaluation_results.json", self.as_dict())
        if emit_bibtex:
            reporting.write_text(target / "evaluation_references.bib",
                                 references.bibtex(self.reference_keys))
        if copy_run_artefacts and self.context.run_dir:
            _copy_run_artefacts(self.context.run_dir, target)
        return target

    def default_out_dir(self) -> Path:
        """Where artefacts go when the caller names no directory."""
        if self.context.run_dir:
            return Path(self.context.run_dir) / "evaluation"
        return Path("evaluation_output") / datetime.now().strftime("%Y%m%d_%H%M%S")


def _copy_run_artefacts(run_dir: PathLike, target: Path) -> None:
    """Copy the artefacts that document how a run was generated."""
    import shutil

    target.mkdir(parents=True, exist_ok=True)
    for name in ("config_used.yaml", "system_prompt.txt"):
        source = Path(run_dir) / name
        if source.exists() and source.resolve() != (target / name).resolve():
            shutil.copyfile(source, target / name)


def run_evaluation(config: EvaluationConfig) -> EvaluationOutcome:
    """Run every configured tier and return the results without writing them.

    Raises:
        ValueError: neither `run_dir` nor `transcripts` was given, or the
            transcript file holds no entries.
        FileNotFoundError: a named input does not exist.
    """
    context, records = _load_inputs(config)
    if not records:
        raise ValueError("no transcript entries to evaluate")

    spec = ConstraintSpec.load(config.constraints)
    rubric = Rubric.load(config.rubric) if config.judge_models else None

    tiers: Dict[str, str] = {}
    warnings: List[str] = []
    reference_keys: List[str] = list(CONSTRAINT_REFS) + list(RELEVANCE_REFS) \
        + list(FACT_REFS) + list(STATS_REFS)

    results = [ItemResult(record=record) for record in records]

    _run_tier0(results, spec, context, config, tiers)
    _run_input_fidelity(results, records, config, tiers, reference_keys)
    _run_tier1(results, records, config, tiers)
    latency = _run_latency(results, context, config, tiers, reference_keys)
    _run_tier2(results, context, config, tiers)
    judge_cost = _run_tier3(results, context, rubric, config, tiers,
                            reference_keys, warnings)

    policy = AcceptancePolicy(
        min_quality_composite=config.min_quality,
        min_safety_score=config.min_safety,
        require_constraint_pass=config.constraint_gate,
        strict_constraints=not config.loose_constraints)
    for result in results:
        score_item(result, rubric, policy)

    label = config.label or (context.label if context.run_dir else "run")
    summary = summarize_run(label, results, rubric, policy, seed=config.seed)
    if latency is not None and latency.available:
        summary.latency_stages = summarize_stages(latency)
        summary.resource_medians = latency.resource_medians

    agreement, calibration = _run_tier4(results, rubric, config, tiers,
                                        reference_keys)
    pairwise = _run_pairwise(records, context, rubric, config, tiers)

    report = reporting.render_text_report(
        context=context, spec=spec, rubric=rubric, policy=policy,
        results=results, summary=summary, tiers=tiers,
        reference_keys=reference_keys, agreement=agreement,
        calibration=calibration, judge_cost=judge_cost, pairwise=pairwise)

    return EvaluationOutcome(
        label=label, context=context, spec=spec, rubric=rubric, policy=policy,
        results=results, summary=summary, tiers=tiers,
        reference_keys=reference_keys, report=report, agreement=agreement,
        calibration=calibration, judge_cost=judge_cost, pairwise=pairwise,
        latency=latency, warnings=warnings)


def _load_inputs(config: EvaluationConfig) -> tuple:
    """Resolve the run context and build the record list."""
    if config.run_dir:
        context = discover_run(config.run_dir)
        if config.transcripts:
            context.transcripts_path = config.transcripts
        if context.transcripts_path is None:
            raise FileNotFoundError(
                f"No transcripts.yaml or transcripts.jsonl in {config.run_dir}")
    elif config.transcripts:
        context = RunContext(transcripts_path=config.transcripts)
    else:
        raise ValueError("provide run_dir or transcripts")

    if config.system_prompt:
        context.system_prompt = config.system_prompt.read_text(
            encoding="utf-8").strip()
        context.system_prompt_path = config.system_prompt

    entries = load_transcripts(context.transcripts_path)
    spec_index = load_scenario_spec(config.spec)
    answer_key = load_answer_key(config.answer_key)
    return context, build_records(entries, spec_index, answers=answer_key)


def _run_tier0(results: List[ItemResult], spec: ConstraintSpec,
               context: RunContext, config: EvaluationConfig,
               tiers: Dict[str, str]) -> None:
    max_tokens = config.max_tokens if config.max_tokens else context.max_tokens
    runtime_params = {"max_tokens": max_tokens} if max_tokens else {}

    for result in results:
        result.constraints = evaluate_constraints(result.record, spec,
                                                  runtime_params)

    detail = f"{len(spec.checks)} checks from {spec.spec_id}"
    detail += (f", token cap {max_tokens}" if max_tokens
               else ", no token cap recorded (truncation check inert)")
    tiers["tier 0 constraints and readability"] = detail


def _run_input_fidelity(results: List[ItemResult], records: Sequence[Record],
                        config: EvaluationConfig, tiers: Dict[str, str],
                        reference_keys: List[str]) -> None:
    """Measure how faithfully the recognizer heard each reference utterance.

    Runs whenever the transcript carries the intended text. The figures describe
    the recognizer and the audio, not the model under test, and they are what
    separates a response that answered the wrong question from one that answered
    the question wrongly.
    """
    with_reference = [r for r in records if r.has_reference_text]
    if not with_reference:
        tiers["input fidelity (recognizer)"] = (
            "off (transcript carries no ori_text reference)")
        return

    for result in results:
        result.asr = evaluate_asr_fidelity(result.record)

    reference_keys.extend(ASR_REFS)
    rate, errors, reference_words = corpus_error_rate(with_reference)
    strata: Dict[str, int] = {}
    for result in results:
        stratum = result.asr.get("stt_stratum")
        if stratum:
            strata[str(stratum)] = strata.get(str(stratum), 0) + 1
    breakdown = ", ".join(f"{name} {count}" for name, count in strata.items())
    tiers["input fidelity (recognizer)"] = (
        f"reference text for {len(with_reference)}/{len(records)} item(s), "
        f"corpus WER {rate:.3f} ({errors}/{reference_words} words); "
        f"strata: {breakdown}")


def _run_latency(results: List[ItemResult], context: RunContext,
                 config: EvaluationConfig, tiers: Dict[str, str],
                 reference_keys: List[str]) -> Optional[RunLatency]:
    """Attach the timings of the stages that produced each response.

    Joined by recording rather than by position, so an item missing from either
    file leaves that item's timings empty instead of shifting every later item.
    """
    if not config.include_latency:
        tiers["runtime cost"] = "off (include_latency not set)"
        return None
    if context.run_dir is None:
        tiers["runtime cost"] = "off (no run directory to read a latency log from)"
        return None

    latency = load_run_latency(context.run_dir,
                               warmup_items=config.latency_warmup)
    if not latency.available:
        detail = latency.warnings[0] if latency.warnings else "no timings found"
        tiers["runtime cost"] = f"skipped: {detail}"
        return latency

    matched = 0
    for result in results:
        key = item_key(result.record.filename) or item_key(result.record.item_id)
        measurements = latency.for_item(key)
        if measurements:
            result.latency = dict(measurements)
            matched += 1

    reference_keys.extend(LATENCY_REFS)
    warmup = (f", {len(latency.warmup_items)} warm-up item(s) excluded "
              f"({', '.join(latency.warmup_items)})"
              if latency.warmup_items else ", no warm-up exclusion")
    tiers["runtime cost"] = (
        f"{matched}/{len(results)} item(s) timed from "
        f"{latency.path.name if latency.path else 'latency log'}{warmup}")
    return latency


def _run_tier1(results: List[ItemResult], records: Sequence[Record],
               config: EvaluationConfig, tiers: Dict[str, str]) -> None:
    idf = build_idf_index(records)
    embeddings = EmbeddingBackend(config.embedding_model)

    for result in results:
        result.relevance = evaluate_relevance(result.record, idf, embeddings)

    n_with_reference = sum(1 for r in records if r.reference_answers)
    n_unsupported = sum(1 for r in records
                        if r.reference_answers and r.answer_unsupported)
    detail = f"IDF over {len(records)} items"
    if n_with_reference:
        source = (config.answer_key.name if config.answer_key
                  else "scenario spec")
        detail += (f", reference measures for {n_with_reference} item(s) "
                   f"from {source}")
        detail += (f" ({n_unsupported} marked unanswerable in the source, scored "
                   f"against a plausible answer)" if n_unsupported else "")
    else:
        detail += ", no reference answers supplied"
    if config.embedding_model:
        detail += (f", embeddings: {config.embedding_model}"
                   if embeddings.available
                   else f", embeddings unavailable: {embeddings.error}")
    tiers["tier 1 relevance"] = detail


def _run_tier2(results: List[ItemResult], context: RunContext,
               config: EvaluationConfig, tiers: Dict[str, str]) -> None:
    for result in results:
        result.factuality["audit_atoms"] = extract_audit_atoms(
            result.record.llm_text).as_dict()
    tiers["tier 2 audit atoms"] = "deterministic extraction (always on)"

    if config.selfcheck_samples > 0:
        _run_selfcheck_tier(results, context, config, tiers)
    else:
        tiers["tier 2 self-consistency"] = "off (selfcheck_samples = 0)"

    if config.fact_precision:
        _run_fact_precision_tier(results, config, tiers)
    else:
        tiers["tier 2 atomic factual precision"] = "off (fact_precision not set)"


def _run_selfcheck_tier(results: List[ItemResult], context: RunContext,
                        config: EvaluationConfig, tiers: Dict[str, str]) -> None:
    model = config.selfcheck_model or context.config.get("ollama_model")
    if not model:
        tiers["tier 2 self-consistency"] = (
            "skipped: no model known; set selfcheck_model")
        return

    embeddings = EmbeddingBackend(config.embedding_model)
    client = OllamaClient(model=model, url=config.judge_url,
                          timeout=config.judge_timeout)
    failure = client.probe()
    if failure:
        tiers["tier 2 self-consistency"] = f"skipped: {model} unreachable ({failure})"
        return

    warning = ""
    generating_model = context.config.get("ollama_model")
    if generating_model and generating_model != model:
        warning = (f"; WARNING resampling with {model} but responses came from "
                   f"{generating_model}, so this measures cross-model "
                   f"disagreement, not self-consistency")

    for index, result in enumerate(results, 1):
        config.notify(f"self-consistency {index}/{len(results)}")
        outcome = run_selfcheck(result.record, client, context.system_prompt,
                                n_samples=config.selfcheck_samples,
                                temperature=config.selfcheck_temperature,
                                embeddings=embeddings)
        result.factuality.update(outcome.as_dict())

    kernel = ("embedding_cosine" if embeddings.available
              else "token_f1_surrogate")
    tiers["tier 2 self-consistency"] = (
        f"{config.selfcheck_samples} resamples at temperature "
        f"{config.selfcheck_temperature} with {model}, support kernel "
        f"{kernel}{warning}")


def _run_fact_precision_tier(results: List[ItemResult],
                             config: EvaluationConfig,
                             tiers: Dict[str, str]) -> None:
    if not config.judge_models:
        tiers["tier 2 atomic factual precision"] = (
            "skipped: fact_precision needs at least one judge model")
        return

    model = config.judge_models[0]
    client = OllamaClient(model=model, url=config.judge_url,
                          timeout=config.judge_timeout)
    failure = client.probe()
    if failure:
        tiers["tier 2 atomic factual precision"] = (
            f"skipped: {model} unreachable ({failure})")
        return

    for index, result in enumerate(results, 1):
        config.notify(f"atomic claims {index}/{len(results)}")
        outcome = run_fact_precision(result.record, client,
                                     max_claims=config.max_claims)
        result.factuality.update(outcome.as_dict())

    sources = {r.factuality.get("factprecision_knowledge_source")
               for r in results}
    tiers["tier 2 atomic factual precision"] = (
        f"verifier {model}, knowledge source(s): "
        f"{', '.join(sorted(s for s in sources if s))}")


def _run_tier3(results: List[ItemResult], context: RunContext,
               rubric: Optional[Rubric], config: EvaluationConfig,
               tiers: Dict[str, str], reference_keys: List[str],
               warnings: List[str]) -> Optional[List[Dict[str, Any]]]:
    if rubric is None:
        tiers["tier 3 rubric grading"] = "off (no judge model)"
        return None

    reference_keys.extend(rubric.reference_keys())
    panel = build_panel(config.judge_models, config.judge_url, rubric,
                        samples=config.judge_samples,
                        timeout=config.judge_timeout)

    reachable = []
    for judge in panel.judges:
        failure = judge.client.probe()
        if failure:
            message = (f"judge {judge.client.model} unreachable ({failure}); "
                       f"excluded from the panel")
            warnings.append(message)
            config.notify(message)
        else:
            reachable.append(judge)
    panel.judges = reachable

    if not panel.judges:
        tiers["tier 3 rubric grading"] = "skipped: no judge model reachable"
        return None

    n_dimensions = len(rubric.dimensions)
    for index, result in enumerate(results, 1):
        config.notify(f"rubric grading {index}/{len(results)} "
                      f"({len(panel.judges)} judge(s) x up to {n_dimensions} "
                      f"dimensions x {config.judge_samples} sample(s))")
        result.judgement = panel.score(result.record, context.system_prompt,
                                       seed=config.seed)

    models = ", ".join(judge.client.model for judge in panel.judges)
    tiers["tier 3 rubric grading"] = (
        f"rubric {rubric.rubric_id}, judge(s): {models}, "
        f"{config.judge_samples} sample(s) per dimension"
        + (" [panel: per-model scores and spread reported]"
           if len(panel.judges) > 1 else " [single judge: no panel control]"))
    return panel.cost_summary()


def _run_tier4(results: List[ItemResult], rubric: Optional[Rubric],
               config: EvaluationConfig, tiers: Dict[str, str],
               reference_keys: List[str]) -> tuple:
    if not config.human_annotations:
        tiers["tier 4 reliability and calibration"] = (
            "off (no human annotations)")
        return None, None

    try:
        tables = load_human_annotations(config.human_annotations)
    except (FileNotFoundError, ValueError) as exc:
        tiers["tier 4 reliability and calibration"] = f"skipped: {exc}"
        return None, None

    reference_keys.extend(AGREEMENT_REFS)
    agreement = [agreement_report(dimension, ratings).as_dict()
                 for dimension, ratings in sorted(tables.items())]

    calibration = None
    if rubric is not None and any(r.judgement for r in results):
        calibration = []
        for dimension, ratings in sorted(tables.items()):
            judge_scores = {r.item_id: score for r in results
                            if (score := r.judge_score(dimension)) is not None}
            if judge_scores:
                calibration.append(calibrate_judge(
                    dimension, ratings, judge_scores, seed=config.seed).as_dict())

    detail = f"{len(tables)} dimension(s) from {config.human_annotations.name}"
    detail += ("; judge calibrated by prediction-powered inference"
               if calibration else "; no judge scores to calibrate against")
    tiers["tier 4 reliability and calibration"] = detail
    return agreement, calibration


def _run_pairwise(records: Sequence[Record], context: RunContext,
                  rubric: Optional[Rubric], config: EvaluationConfig,
                  tiers: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
    if not config.compare_run_dir:
        return None
    if rubric is None:
        tiers["configuration contrast"] = (
            "skipped: pairwise comparison needs a judge model")
        return None

    try:
        other_context = discover_run(config.compare_run_dir)
        other_records = build_records(
            load_transcripts(other_context.transcripts_path), {},
            id_prefix="item")
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        tiers["configuration contrast"] = f"skipped: {exc}"
        return None

    by_text = {" ".join(r.stt_text.lower().split()): r for r in other_records}
    dimension_ids = config.pairwise_dimensions or DEFAULT_PAIRWISE_DIMENSIONS
    dimensions = [d for d in rubric.dimensions if d.dim_id in dimension_ids]
    if not dimensions:
        tiers["configuration contrast"] = (
            f"skipped: none of {', '.join(dimension_ids)} exist in "
            f"{rubric.rubric_id}")
        return None

    client = OllamaClient(model=config.judge_models[0], url=config.judge_url,
                          timeout=config.judge_timeout)
    failure = client.probe()
    if failure:
        tiers["configuration contrast"] = f"skipped: judge unreachable ({failure})"
        return None

    verdicts: List[Dict[str, Any]] = []
    for record in records:
        counterpart = by_text.get(" ".join(record.stt_text.lower().split()))
        if counterpart is None:
            continue
        for dimension in dimensions:
            config.notify(f"pairwise {record.item_id}/{dimension.dim_id}")
            verdict = compare_pairwise(client, record, counterpart,
                                       dimension.definition,
                                       context.system_prompt)
            verdicts.append({
                "item_id": record.item_id,
                "dimension": dimension.dim_id,
                "baseline": context.label,
                "contrast": other_context.label,
                "verdict": verdict.verdict,
                "order_consistent": verdict.consistent,
                "reason": verdict.reason,
            })

    tiers["configuration contrast"] = (
        f"{len(verdicts)} pairwise comparison(s) against "
        f"{other_context.label}, each run in both presentation orders")
    return verdicts
