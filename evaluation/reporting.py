#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report rendering for the response evaluation toolkit.

The text report is the primary artefact and is written to be read without the
JSON alongside it. Three properties are deliberate:

  * Every tier states which publication it comes from and what its validation
    status is, so a number cannot be lifted out of the report without its
    provenance.
  * Every failing check reports what was observed and what was required, so a
    verdict can be checked against the response by hand.
  * Interpretation limits are printed with the numbers rather than in separate
    documentation, because a results table that circulates on its own is the one
    that gets over-interpreted.

Formatting follows aggregate_logs.py so that both summaries read the same way in
a plain text editor.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import references
from .aggregation import AcceptancePolicy, ItemResult, RunSummary
from .constraints import ConstraintSpec
from .judge import Rubric
from .loaders import RunContext

THICK = "=" * 100
THIN = "-" * 100


class _Sections:
    """Numbers report sections in the order they are actually emitted.

    Sections are skipped when their tier did not run, so a fixed numbering would
    leave gaps that read as missing output rather than as omitted tiers.
    """

    def __init__(self):
        self._next = 0

    def title(self, text: str) -> List[str]:
        self._next += 1
        return [f"{self._next}. {text}", THICK]


def render_text_report(context: RunContext,
                       spec: ConstraintSpec,
                       rubric: Optional[Rubric],
                       policy: AcceptancePolicy,
                       results: Sequence[ItemResult],
                       summary: RunSummary,
                       tiers: Dict[str, str],
                       reference_keys: Sequence[str],
                       agreement: Optional[List[Dict[str, Any]]] = None,
                       calibration: Optional[List[Dict[str, Any]]] = None,
                       judge_cost: Optional[List[Dict[str, Any]]] = None,
                       pairwise: Optional[List[Dict[str, Any]]] = None) -> str:
    """Render the full human-readable evaluation report."""
    sections = _Sections()
    lines: List[str] = []
    lines += _header(context, spec, rubric, policy, summary, tiers)
    lines += _item_table(results, sections)
    lines += _check_rates(summary, sections)
    lines += _input_quality_section(summary, sections)
    lines += _latency_section(summary, sections)
    lines += _metric_summaries(summary, sections)
    lines += _judge_summaries(summary, rubric, sections)
    lines += _category_table(summary, sections)
    lines += _pairwise_section(pairwise, sections)
    lines += _item_details(results, rubric, sections)
    lines += _agreement_section(agreement, calibration, sections)
    lines += _cost_section(judge_cost, sections)
    lines += _bibliography(reference_keys, sections)
    lines += _caveats(sections)
    return "\n".join(lines)


def _header(context: RunContext, spec: ConstraintSpec, rubric: Optional[Rubric],
            policy: AcceptancePolicy, summary: RunSummary,
            tiers: Dict[str, str]) -> List[str]:
    lines = [THICK,
             "                        LLM RESPONSE EVALUATION REPORT",
             THICK,
             f"Generated on           : {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Run label              : {summary.label}",
             f"Transcripts            : {context.transcripts_path or '(not set)'}",
             f"Run directory          : {context.run_dir or '(not set)'}",
             f"System prompt          : {context.system_prompt_path or '(not recorded)'}",
             f"Constraint spec        : {spec.spec_id} ({spec.source_path})",
             f"Rubric                 : "
             + (f"{rubric.rubric_id} ({rubric.source_path})" if rubric
                else "(judge tier not run)"),
             f"Items evaluated        : {summary.n_items}"
             + (f"  ({summary.n_empty_responses} with empty response)"
                if summary.n_empty_responses else ""),
             ""]

    interesting = ["ollama_model", "llm_temperature", "llm_seed",
                   "llm_max_tokens", "stt_engine", "tts_engine", "mode"]
    recorded = [(k, context.config[k]) for k in interesting if k in context.config]
    if recorded:
        lines.append("Generating configuration (from config_used.yaml):")
        lines += [f"  {key:<20}: {value}" for key, value in recorded]
        lines.append("")

    lines.append("Tiers executed:")
    for name, state in tiers.items():
        lines.append(f"  {name:<34}: {state}")
    lines.append("")

    lines.append("Acceptance policy (predeclare this before measurement):")
    for key, value in policy.as_dict().items():
        lines.append(f"  {key:<28}: {value}")
    lines.append(THICK)
    lines.append("")
    return lines


def _item_table(results: Sequence[ItemResult], sections: _Sections) -> List[str]:
    lines = sections.title("PER-ITEM SCORES")
    lines += [f"{'Item':<10}\t{'Category':<26}\t{'WER':>6}\t{'Adher':>6}"
             f"\t{'Chk%':>6}\t{'Words':>6}\t{'FRE':>7}\t{'Cover':>6}"
             f"\t{'CovOri':>7}\t{'E2E ms':>8}\t{'Incon':>6}"
             f"\t{'Fact':>6}\t{'Qual':>6}\t{'Safe':>6}\t{'Accept':>7}",
             THIN]

    for result in results:
        constraints = result.constraints
        adherence = "-"
        check_rate = "-"
        if constraints is not None:
            adherence = "PASS" if constraints.item_level_strict else "FAIL"
            rate = constraints.check_level_strict
            check_rate = f"{100 * rate:.0f}" if rate is not None else "-"

        measures = constraints.measures if constraints else {}
        lines.append("\t".join([
            f"{result.item_id:<10}",
            f"{result.record.category[:26]:<26}",
            f"{_num(result.asr.get('stt_wer'), 2):>6}",
            f"{adherence:>6}",
            f"{check_rate:>6}",
            f"{_num(measures.get('word_count'), 0):>6}",
            f"{_num(measures.get('flesch_reading_ease'), 1):>7}",
            f"{_num(result.relevance.get('request_coverage'), 2):>6}",
            f"{_num(result.relevance.get('intent_coverage'), 2):>7}",
            f"{_num(result.latency.get('lat_e2e_response_ready'), 0):>8}",
            f"{_num(result.factuality.get('selfcheck_mean_inconsistency'), 2):>6}",
            f"{_num(result.factuality.get('factprecision_precision'), 2):>6}",
            f"{_num(result.quality_composite, 2):>6}",
            f"{_num(result.safety_minimum, 2):>6}",
            f"{_verdict(result.accepted):>7}",
        ]))

    lines += [THIN,
              "WER = word error rate of the recognized utterance against the "
              "reference utterance, a",
              "property of the recognizer rather than of the model. Adher = all "
              "hard prompt constraints",
              "satisfied as generated (item level, strict). Chk% = share of hard "
              "constraints satisfied.",
              "FRE = Flesch Reading Ease. Cover = coverage of the recognized "
              "request, CovOri = coverage of",
              "the reference utterance. E2E ms = wall-clock span of the whole "
              "item, recording included. Incon =",
              "self-consistency inconsistency (higher is worse). Fact = supported "
              "share of atomic claims.",
              "Qual = weighted rubric composite (1-5). Safe = worst safety "
              "dimension (1-5). '-' means the",
              "tier producing that column did not run.",
              THICK, ""]
    return lines


def _check_rates(summary: RunSummary, sections: _Sections) -> List[str]:
    if not summary.per_check_strict:
        return []

    lines = sections.title("CONSTRAINT PASS RATES "
                           "(IFEval-style, Zhou et al., 2023)")
    lines += [f"{'Check id':<34}\t{'Strict %':>10}\t{'Loose %':>10}\t{'Delta':>8}",
              THIN]

    for check_id in sorted(summary.per_check_strict,
                           key=lambda k: summary.per_check_strict[k]):
        strict = summary.per_check_strict[check_id]
        loose = summary.per_check_loose.get(check_id, strict)
        lines.append(f"{check_id:<34}\t{100 * strict:>10.1f}\t{100 * loose:>10.1f}"
                     f"\t{100 * (loose - strict):>8.1f}")

    lines += [THIN]
    if summary.constraint_item_rate_strict is not None:
        lines += [f"{'ITEM LEVEL (all hard checks)':<34}"
                  f"\t{100 * summary.constraint_item_rate_strict:>10.1f}"
                  f"\t{100 * (summary.constraint_item_rate_loose or 0):>10.1f}"
                  f"\t{100 * ((summary.constraint_item_rate_loose or 0) - summary.constraint_item_rate_strict):>8.1f}"]
    lines += ["",
              "Rates are over the items where the check applied; a check with "
              "nothing to verify for an",
              "item is excluded rather than counted as a pass. Delta is the "
              "share of items that violate",
              "the constraint only in formatting: a large delta points at output "
              "formatting, a delta near",
              "zero means the instruction itself was not followed. Report the "
              "item-level figure as the",
              "headline adherence rate.",
              THICK, ""]
    return lines


def _input_quality_section(summary: RunSummary,
                           sections: _Sections) -> List[str]:
    """Report what the recognizer passed to the model, split by input quality."""
    if not summary.per_stratum:
        return []

    wer = summary.metric_summaries.get("stt_wer", {})
    lines = sections.title("INPUT QUALITY (recognizer against the reference "
                           "utterance)")
    if wer:
        lines += [f"Word error rate over items: median {_num(wer.get('median'), 3)}"
                  f", mean {_num(wer.get('mean'), 3)}"
                  f", max {_num(wer.get('max'), 3)}",
                  ""]

    lines += [f"{'Input stratum':<14}\t{'n':>4}\t{'WER':>7}\t{'Adher %':>9}"
              f"\t{'Cover(stt)':>11}\t{'Cover(ori)':>11}\t{'Words':>7}"
              f"\t{'E2E ms':>9}",
              THIN]
    for stratum, entry in summary.per_stratum.items():
        adherence = entry.get("constraint_item_rate_strict")
        lines.append("\t".join([
            f"{stratum:<14}",
            f"{entry['n_items']:>4}",
            f"{_num(entry.get('stt_wer_median'), 3):>7}",
            f"{(f'{100 * adherence:.1f}' if adherence is not None else '-'):>9}",
            f"{_num(entry.get('request_coverage_median'), 3):>11}",
            f"{_num(entry.get('intent_coverage_median'), 3):>11}",
            f"{_num(entry.get('word_count_median'), 1):>7}",
            f"{_num(entry.get('lat_e2e_response_ready_median'), 0):>9}",
        ]))

    lines += [THIN,
              "Strata are defined on the word error rate of the item: clean = "
              "transcribed exactly,",
              "severe = more than a third of the reference words lost or "
              "altered. These figures are a",
              "property of the recognizer and the audio, identical in every run "
              "over the same recordings,",
              "so they bound what the language model could have answered rather "
              "than describing it.",
              "",
              "Cover(stt) is coverage of the recognized request, which is all the "
              "model received.",
              "Cover(ori) is coverage of what the speaker actually asked. The gap "
              "between them is the",
              "cost the recognizer imposed on the interaction, and it belongs to "
              "the recognizer.",
              "",
              "Compare a column down the strata of one run with caution, and "
              "across runs freely. A correct",
              "answer to a closed question repeats none of it -- \"In what years "
              "was the Great Famine?\"",
              "answered \"1918-1921.\" scores zero coverage -- so a stratum "
              "holding more such questions",
              "shows lower coverage for reasons that have nothing to do with the "
              "model. Prompt adherence",
              "carries no such artefact and is the column to read down this "
              "table.",
              THICK, ""]
    return lines


def _latency_section(summary: RunSummary, sections: _Sections) -> List[str]:
    """Report the timing of each pipeline stage, median first with the tail."""
    if not summary.latency_stages:
        return []

    lines = sections.title("RUNTIME COST PER STAGE (from the run's latency log)")
    lines += [f"{'Stage':<24}\t{'n':>4}\t{'Median':>9}\t{'Q1':>9}\t{'Q3':>9}"
              f"\t{'P90':>9}\t{'P95':>9}\t{'Max':>9}",
              THIN]
    for entry in summary.latency_stages:
        lines.append("\t".join([
            f"{entry['stage'][:24]:<24}",
            f"{entry['n']:>4}",
            f"{_num(entry.get('median_ms'), 0):>9}",
            f"{_num(entry.get('q1_ms'), 0):>9}",
            f"{_num(entry.get('q3_ms'), 0):>9}",
            f"{_num(entry.get('p90_ms'), 0):>9}",
            f"{_num(entry.get('p95_ms'), 0):>9}",
            f"{_num(entry.get('max_ms'), 0):>9}",
        ]))
    lines += [THIN]

    for name, label in (("llm_tokens_per_sec", "generation throughput (tok/s)"),
                        ("llm_eval_tokens", "generated tokens"),
                        ("llm_prompt_tokens", "prompt tokens"),
                        ("tts_audio_ms", "synthesized audio (ms)")):
        stats = summary.metric_summaries.get(name)
        if stats:
            lines.append(f"{label:<32}: median {_num(stats.get('median'), 1)}"
                         f"   mean {_num(stats.get('mean'), 1)}")

    if summary.resource_medians:
        lines.append("")
        lines.append("Resource use (run medians):")
        for name, value in summary.resource_medians.items():
            lines.append(f"  {name:<20}: {_num(value, 1)}")

    lines += ["",
              "All values are milliseconds unless stated. The median is the "
              "headline figure and the 95th",
              "percentile the one a user notices, since an assistant is judged "
              "by the answers that arrive",
              "late (Dean and Barroso, 2013). Timings are wall-clock on a shared "
              "host: they compare runs",
              "measured under the same conditions and are not portable hardware "
              "benchmarks.",
              "",
              "The stages overlap and are not all measured from the same instant, "
              "so they do not add up.",
              "`stt` is the recording arriving at speaking pace, which on file "
              "input is the length of the",
              "audio rather than a cost of recognition. `ttfa` runs from the "
              "speaker stopping to the first",
              "audio out and is the delay a user actually waits through. "
              "`e2e_response_ready` spans the whole",
              "item including that recording, so it is dominated by how long the "
              "person spoke and moves",
              "little between models.",
              THICK, ""]
    return lines


def _metric_summaries(summary: RunSummary, sections: _Sections) -> List[str]:
    if not summary.metric_summaries:
        return []

    lines = sections.title("DESCRIPTIVE MEASURES")
    lines += [f"{'Metric':<32}\t{'n':>4}\t{'Mean':>9}\t{'SD':>9}\t{'Median':>9}"
              f"\t{'Min':>9}\t{'Max':>9}\t{'95% CI':>21}",
              THIN]

    for name, stats in summary.metric_summaries.items():
        ci = "-"
        if stats.get("ci95_low") is not None:
            ci = f"[{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]"
        lines.append("\t".join([
            f"{name:<32}",
            f"{stats['n']:>4}",
            f"{_num(stats.get('mean'), 3):>9}",
            f"{_num(stats.get('sd'), 3):>9}",
            f"{_num(stats.get('median'), 3):>9}",
            f"{_num(stats.get('min'), 3):>9}",
            f"{_num(stats.get('max'), 3):>9}",
            f"{ci:>21}",
        ]))

    lines += [THIN,
              "Intervals are percentile bootstrap intervals for the mean "
              "(Efron and Tibshirani, 1993),",
              "omitted below three observations.",
              THICK, ""]
    return lines


def _judge_summaries(summary: RunSummary, rubric: Optional[Rubric],
                     sections: _Sections) -> List[str]:
    if not summary.judge_summaries or rubric is None:
        return []

    domains = {d.dim_id: d.domain for d in rubric.dimensions}
    safety = {d.dim_id for d in rubric.safety_dimensions}

    lines = sections.title("RUBRIC DIMENSION SCORES (1-5, higher is better)")
    lines += [f"{'Dimension':<30}\t{'Domain':<26}\t{'n':>4}\t{'Mean':>7}"
              f"\t{'SD':>7}\t{'Min':>5}\t{'95% CI':>18}\t{'Panel':>6}",
              THIN]

    for dim_id, stats in summary.judge_summaries.items():
        ci = "-"
        if stats.get("ci95_low") is not None:
            ci = f"[{stats['ci95_low']:.2f}, {stats['ci95_high']:.2f}]"
        marker = " (S)" if dim_id in safety else ""
        lines.append("\t".join([
            f"{(dim_id + marker)[:30]:<30}",
            f"{domains.get(dim_id, '')[:26]:<26}",
            f"{stats['n']:>4}",
            f"{_num(stats.get('mean'), 2):>7}",
            f"{_num(stats.get('sd'), 2):>7}",
            f"{_num(stats.get('min'), 1):>5}",
            f"{ci:>18}",
            f"{_num(summary.panel_spread.get(dim_id), 2):>6}",
        ]))

    lines += [THIN,
              "(S) marks a safety-critical dimension. These are excluded from the "
              "quality composite and",
              "gated by their minimum instead. Panel = largest disagreement "
              "between panel members on any",
              "item; a wide spread means the dimension is not measured reliably "
              "enough to carry a claim.",
              THICK, ""]

    if summary.safety_failures:
        lines += ["SAFETY GATE FAILURES: " + ", ".join(summary.safety_failures),
                  "These items must be reviewed by a human expert before any "
                  "aggregate is reported.",
                  ""]
    return lines


def _category_table(summary: RunSummary, sections: _Sections) -> List[str]:
    if len(summary.per_category) < 2:
        return []

    lines = sections.title("BREAKDOWN BY SCENARIO CATEGORY")
    lines += [f"{'Category':<30}\t{'n':>4}\t{'Adher %':>9}\t{'Quality':>9}"
              f"\t{'Worst safe':>11}\t{'Accept %':>9}",
              THIN]

    for category, entry in summary.per_category.items():
        adherence = entry.get("constraint_item_rate_strict")
        acceptance = entry.get("acceptance_rate")
        lines.append("\t".join([
            f"{category[:30]:<30}",
            f"{entry['n_items']:>4}",
            f"{(f'{100 * adherence:.1f}' if adherence is not None else '-'):>9}",
            f"{_num(entry.get('quality_composite_mean'), 2):>9}",
            f"{_num(entry.get('safety_minimum_worst'), 2):>11}",
            f"{(f'{100 * acceptance:.1f}' if acceptance is not None else '-'):>9}",
        ]))
    lines += [THICK, ""]
    return lines


def _pairwise_section(pairwise: Optional[List[Dict[str, Any]]],
                      sections: _Sections) -> List[str]:
    if not pairwise:
        return []

    lines = sections.title("CONFIGURATION CONTRAST "
                           "(order-controlled pairwise judging)")
    lines += [f"{'Dimension':<30}\t{'baseline wins':>14}\t{'contrast wins':>14}"
              f"\t{'ties':>6}\t{'order-consistent':>17}",
              THIN]

    by_dimension: Dict[str, List[Dict[str, Any]]] = {}
    for verdict in pairwise:
        by_dimension.setdefault(verdict["dimension"], []).append(verdict)

    for dimension, group in sorted(by_dimension.items()):
        wins_baseline = sum(1 for v in group if v["verdict"] == "A")
        wins_contrast = sum(1 for v in group if v["verdict"] == "B")
        ties = sum(1 for v in group if v["verdict"] == "tie")
        consistent = sum(1 for v in group if v["order_consistent"])
        lines.append(f"{dimension[:30]:<30}\t{wins_baseline:>14}"
                     f"\t{wins_contrast:>14}\t{ties:>6}"
                     f"\t{consistent:>10}/{len(group):<6}")

    labels = {(v["baseline"], v["contrast"]) for v in pairwise}
    lines += [THIN]
    for baseline, contrast in sorted(labels):
        lines.append(f"baseline = {baseline}    contrast = {contrast}")
    lines += ["",
              "Every pair is judged twice with the responses swapped. A pair "
              "whose verdict flips with the",
              "order is counted as a tie, since a verdict that depends on "
              "presentation order is position",
              "bias (Zheng et al., 2023) rather than a preference. A low "
              "order-consistent count invalidates",
              "the comparison for that dimension.",
              THICK, ""]
    return lines


def _item_details(results: Sequence[ItemResult], rubric: Optional[Rubric],
                  sections: _Sections) -> List[str]:
    lines = sections.title("PER-ITEM DETAIL AND AUDIT TRAIL")

    for result in results:
        lines.append(f"[{result.item_id}] category={result.record.category}"
                     f"  accepted={_verdict(result.accepted)}")
        lines.append(f"  user     : {_wrap(result.record.stt_text)}")
        lines.append(f"  response : {_wrap(result.record.llm_text)}")

        if result.constraints is not None:
            failed = [c for c in result.constraints.hard if not c.passed_strict]
            if failed:
                lines.append("  constraint violations:")
                for check in failed:
                    loose = " (passes after formatting fix)" if check.passed_loose else ""
                    lines.append(f"    - {check.check_id}: observed "
                                 f"{check.observed} | required {check.expected}{loose}")
            soft_failed = [c for c in result.constraints.applicable
                           if c.severity == "soft" and not c.passed_strict]
            for check in soft_failed:
                lines.append(f"    ~ advisory {check.check_id}: {check.observed}")

        atoms = result.factuality.get("audit_atoms")
        if atoms and atoms.get("atom_count"):
            parts = []
            for key in ("years", "quantities", "name_spans"):
                if atoms.get(key):
                    parts.append(f"{key}: {', '.join(atoms[key])}")
            lines.append("  factual commitments to verify -> " + "; ".join(parts))

        flagged = result.factuality.get("selfcheck_flagged_sentences")
        if flagged:
            lines.append("  low self-consistency sentences:")
            for sentence in flagged:
                lines.append(f"    - {_wrap(sentence, indent=6)}")

        unsupported = result.factuality.get("factprecision_unsupported_claims")
        if unsupported:
            lines.append("  claims labelled unsupported:")
            for claim in unsupported:
                lines.append(f"    - {_wrap(claim, indent=6)}")

        if result.judgement is not None and rubric is not None:
            low = []
            for dimension in rubric.dimensions:
                score = result.judgement.panel_mean(dimension.dim_id)
                if score is not None and score <= 3.0:
                    reasons = result.judgement.reasons(dimension.dim_id)
                    reason = next(iter(reasons.values()), "")
                    low.append(f"    - {dimension.dim_id}={score:.1f}"
                               + (f": {reason}" if reason else ""))
            if low:
                lines.append("  rubric dimensions scored 3 or below:")
                lines += low

        if result.acceptance_reasons:
            lines.append("  not accepted because: "
                         + "; ".join(result.acceptance_reasons))
        lines.append(THIN)

    lines.append("")
    return lines


def _agreement_section(agreement: Optional[List[Dict[str, Any]]],
                       calibration: Optional[List[Dict[str, Any]]],
                       sections: _Sections) -> List[str]:
    if not agreement and not calibration:
        return []

    lines = sections.title("HUMAN RELIABILITY AND JUDGE CALIBRATION")

    if agreement:
        lines += [f"{'Dimension':<28}\t{'items':>6}\t{'raters':>6}\t{'%agree':>7}"
                  f"\t{'alpha':>7}\t{'AC1':>7}\t{'kappa':>7}\t{'ICC(2,1)':>9}"
                  f"\t{'ICC label':>11}",
                  THIN]
        for entry in agreement:
            kappa = entry.get("cohen_kappa")
            if kappa is None:
                kappa = entry.get("fleiss_kappa")
            lines.append("\t".join([
                f"{entry['dimension'][:28]:<28}",
                f"{entry['n_items_multi_rated']:>6}",
                f"{entry['n_raters']:>6}",
                f"{_num(entry.get('percent_agreement'), 3):>7}",
                f"{_num(entry.get('krippendorff_alpha_ordinal'), 3):>7}",
                f"{_num(entry.get('gwet_ac1'), 3):>7}",
                f"{_num(kappa, 3):>7}",
                f"{_num(entry.get('icc_2_1'), 3):>9}",
                f"{str(entry.get('icc_2_1_interpretation', '-')):>11}",
            ]))
        lines += [THIN,
                  "alpha is Krippendorff's alpha on the ordinal metric. On a "
                  "category where nearly every",
                  "item is a pass, read AC1 rather than kappa: kappa collapses "
                  "under that skew "
                  "(Feinstein",
                  "and Cicchetti, 1990). ICC labels follow Koo and Li (2016).",
                  ""]

    if calibration:
        lines += ["Judge calibration against human ratings:", THIN,
                  f"{'Dimension':<24}\t{'n lab':>6}\t{'n unlab':>8}\t{'human':>7}"
                  f"\t{'judge':>7}\t{'bias':>7}\t{'MAE':>6}\t{'rho':>6}"
                  f"\t{'PPI estimate (95% CI)':>28}",
                  THIN]
        for entry in calibration:
            ppi = "-"
            if entry.get("ppi_estimate") is not None:
                ppi = (f"{entry['ppi_estimate']:.3f} "
                       f"[{entry['ppi_ci95_low']:.3f}, {entry['ppi_ci95_high']:.3f}]")
            lines.append("\t".join([
                f"{entry['dimension'][:24]:<24}",
                f"{entry['n_labelled']:>6}",
                f"{entry['n_unlabelled']:>8}",
                f"{_num(entry.get('human_mean'), 2):>7}",
                f"{_num(entry.get('judge_mean_all_items'), 2):>7}",
                f"{_num(entry.get('judge_bias_vs_human'), 2):>7}",
                f"{_num(entry.get('judge_mae_vs_human'), 2):>6}",
                f"{_num(entry.get('judge_human_spearman'), 2):>6}",
                f"{ppi:>28}",
            ]))
            if entry.get("note"):
                lines.append(f"  note ({entry['dimension']}): {entry['note']}")
        lines += [THIN,
                  "PPI is the prediction-powered estimate of the mean human "
                  "rating over all items",
                  "(Angelopoulos et al., 2023). Its interval stays valid however "
                  "biased the judge is;",
                  "report it in preference to the raw judge mean. rho is the "
                  "Spearman correlation",
                  "between judge and human on the labelled subsample.",
                  ""]

    lines += [THICK, ""]
    return lines


def _cost_section(judge_cost: Optional[List[Dict[str, Any]]],
                  sections: _Sections) -> List[str]:
    if not judge_cost:
        return []
    lines = sections.title("EVALUATION COST")
    lines += [f"{'Model':<32}\t{'calls':>7}\t{'failures':>9}\t{'tokens':>9}"
              f"\t{'seconds':>9}",
              THIN]
    for entry in judge_cost:
        lines.append(f"{str(entry['model'])[:32]:<32}\t{entry['calls']:>7}"
                     f"\t{entry['failures']:>9}\t{entry['generated_tokens']:>9}"
                     f"\t{entry['wall_seconds']:>9.1f}")
    lines += [THIN,
              "A non-zero failure count means some items were scored from fewer "
              "samples than requested.",
              THICK, ""]
    return lines


def _bibliography(reference_keys: Sequence[str],
                  sections: _Sections) -> List[str]:
    lines = sections.title("METHODS AND SOURCES")
    lines += ["Validation status: verifiable = decided by construction; "
              "validated = validated",
              "instrument or published sampling theory; established = widely used "
              "method with",
              "human-correlation evidence but no instrument status; surrogate = "
              "simplification of a",
              "published method, reported as such.",
              THIN]
    lines += references.bibliography_lines(reference_keys)
    lines += [THICK, ""]
    return lines


def _caveats(sections: _Sections) -> List[str]:
    lines = sections.title("INTERPRETATION LIMITS")
    notes = [
        "Constraint results are decidable and need no calibration. Everything "
        "else in this report is",
        "an estimate and inherits the limits of the method that produced it.",
        "",
        "Judge scores are a screening measure, not a validated instrument. "
        "Zheng et al. (2023) document",
        "position, verbosity and self-preference bias in LLM judges, and a small "
        "local judge is weaker",
        "than the models those results were obtained with. Do not report a judge "
        "mean as a quality",
        "estimate until it has been calibrated against human ratings on a "
        "subsample of this dataset.",
        "",
        "Self-consistency scores use a token-overlap support kernel by default, "
        "not the BERTScore, NLI",
        "or QA kernels of the original method, so absolute values are not "
        "comparable with published",
        "SelfCheckGPT figures. They rank items within this run.",
        "",
        "Atomic factual precision computed without a reference answer relies on "
        "the judge's own",
        "knowledge. Read the knowledge_source field before reporting it.",
        "",
        "Word error rate is not a quality score for the assembled system. It "
        "states how much of the",
        "user's utterance survived recognition, and a system can answer well "
        "from a badly recognized",
        "question or badly from a perfect one (Wang et al., 2003). Read it as "
        "the bound on what the",
        "model could have known, and read the strata table before any pooled "
        "relevance figure.",
        "",
        "Timings are wall-clock measurements taken on a shared host while the "
        "pipeline ran. They are",
        "comparable between runs measured on the same host under the same "
        "preset, and they are not",
        "hardware benchmarks. A stage that should not depend on the language "
        "model, such as recognition,",
        "is the control: if it differs between two runs, the host was not in the "
        "same state and the",
        "timing comparison is confounded.",
        "",
        "Readability formulas were validated for written text. They describe "
        "lexical and syntactic",
        "complexity here, not intelligibility of the synthesized speech, which "
        "requires a listening",
        "test (for example ITU-T P.85 or P.808).",
        "",
        "Scenario coverage bounds what any of this supports. Figures computed on "
        "open-domain items",
        "describe open-domain behaviour and carry no implication for "
        "therapy-training scenarios.",
    ]
    lines += ["  " + note if note else "" for note in notes]
    lines += [THICK]
    return lines


# --------------------------------------------------------------------------
# File writers
# --------------------------------------------------------------------------

def write_item_csv(path: Path, results: Sequence[ItemResult],
                   rubric: Optional[Rubric]) -> None:
    """Write one row per item with every measurement, for downstream analysis."""
    rows = [result.flat_row(rubric) for result in results]
    if not rows:
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write the full machine-readable result set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def write_text(path: Path, text: str) -> None:
    """Write a text artefact, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _num(value: Any, digits: int = 2) -> str:
    """Format a number for a fixed-width column, or '-' when it is absent."""
    if value is None or isinstance(value, bool):
        return "-" if value is None else str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "-"
    return f"{number:.{digits}f}"


def _verdict(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "NO"


def _wrap(text: str, width: int = 88, indent: int = 13) -> str:
    """Fold long text into an indented block so the report stays readable."""
    flat = " ".join((text or "").split())
    if len(flat) <= width:
        return flat

    words = flat.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    pad = " " * indent
    return ("\n" + pad).join(lines)
