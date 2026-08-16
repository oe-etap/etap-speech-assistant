#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score the LLM responses of a pipeline run against published evaluation methods.

The tool reads the transcripts a run already wrote and produces per-item scores,
run-level aggregates, and a report that names the source and validation status of
every method it used.

Tiers are enabled by what is available, so the cheap and certain part always
runs and the expensive parts are opt-in:

    Tier 0  constraint checks and readability      always, no model needed
    Tier 1  relevance screening and overlap        always; reference-based
                                                   measures need --answer-key
                                                   or --spec
    Tier 2  audit atoms                            always
            self-consistency resampling            --selfcheck-samples N
            atomic factual precision               --fact-precision
    Tier 3  rubric grading                         --judge-model TAG
    Tier 4  reliability and calibration            --human-annotations CSV

Examples (PowerShell):

    # Tier 0 and 1 only. No model, no network, finishes immediately.
    python -m evaluation --run-dir outputs\\20260809_164356

    # Add rubric grading by two local judges and self-consistency checking.
    python -m evaluation --run-dir outputs\\20260809_164356 `
        --judge-model phi3:mini --judge-model qwen2.5:7b-instruct `
        --selfcheck-samples 5 --fact-precision

    # Add human ratings to calibrate the judge, and emit BibTeX for the paper.
    python -m evaluation --run-dir outputs\\20260809_164356 `
        --judge-model qwen2.5:7b-instruct `
        --human-annotations ratings.csv --emit-bibtex

    # Compare two configurations with order-controlled pairwise judging.
    python -m evaluation --run-dir outputs\\run_q8 `
        --compare-run-dir outputs\\run_q4 --judge-model qwen2.5:7b-instruct

The same run is available as a function; see evaluation.run_evaluation.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from .ollama_client import DEFAULT_URL
from .pipeline import (DEFAULT_CONSTRAINTS, DEFAULT_RUBRIC, EvaluationConfig,
                       run_evaluation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description=__doc__,
        epilog=("To decide whether a parameter change improved the responses, "
                "compare two runs over the same prompts:\n"
                "  python -m evaluation.comparison "
                "--baseline RUN_A --contrast RUN_B\n"
                "That comparison is paired item by item and needs no human "
                "labels."),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    source = parser.add_argument_group("input")
    source.add_argument("--run-dir", type=Path, help=(
        "Pipeline run directory. Reads transcripts, config_used.yaml and "
        "system_prompt.txt from it, so the evaluation is tied to one run."))
    source.add_argument("--transcripts", type=Path, help=(
        "Transcript file (.yaml or .jsonl). Use instead of --run-dir when the "
        "surrounding run artefacts are unavailable."))
    source.add_argument("--spec", type=Path, help=(
        "Scenario specification YAML supplying per-item category, reference "
        "answers, and required or forbidden content."))
    source.add_argument("--answer-key", type=Path, help=(
        "Dataset metadata table (CSV) holding the corpus's own reference "
        "answers, matched to items by recording name. Answerable items are "
        "scored against the 'answer' column, items marked is_impossible "
        "against 'plausible_answers'."))
    source.add_argument("--system-prompt", type=Path, help=(
        "Override the system prompt used for judging and for resampling."))
    source.add_argument("--label", type=str, help=(
        "Run label for the report. Defaults to the run directory name."))

    tier0 = parser.add_argument_group("tier 0: verifiable constraints")
    tier0.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS,
                       help="Constraint specification YAML (default: %(default)s).")
    tier0.add_argument("--max-tokens", type=int, help=(
        "num_predict cap the run used, for truncation detection. Read from "
        "config_used.yaml when --run-dir is given."))

    tier1 = parser.add_argument_group("tier 1: relevance")
    tier1.add_argument("--embedding-model", type=str, help=(
        "sentence-transformers model name for embedding similarity, e.g. "
        "all-MiniLM-L6-v2. Optional; skipped when the package is absent."))

    tier2 = parser.add_argument_group("tier 2: factuality")
    tier2.add_argument("--selfcheck-samples", type=int, default=0, help=(
        "Number of stochastic resamples per item for SelfCheckGPT-style "
        "consistency scoring. 0 disables the check (default)."))
    tier2.add_argument("--selfcheck-model", type=str, help=(
        "Model to resample with. Must be the model that produced the "
        "responses; defaults to ollama_model from config_used.yaml."))
    tier2.add_argument("--selfcheck-temperature", type=float, default=0.8,
                       help="Resampling temperature (default: %(default)s).")
    tier2.add_argument("--fact-precision", action="store_true", help=(
        "Decompose each response into atomic claims and label each one "
        "(FActScore-style). Requires a judge model."))
    tier2.add_argument("--max-claims", type=int, default=10, help=(
        "Cap on atomic claims per response (default: %(default)s)."))

    tier3 = parser.add_argument_group("tier 3: rubric grading")
    tier3.add_argument("--judge-model", action="append", default=[], help=(
        "Ollama tag of a judge model. Repeat the flag to form a panel, which "
        "reduces single-model bias. Omitting it disables the judge tier."))
    tier3.add_argument("--judge-url", type=str, default=DEFAULT_URL,
                       help="Ollama generate endpoint (default: %(default)s).")
    tier3.add_argument("--judge-samples", type=int, default=1, help=(
        "Ratings sampled per dimension per judge. Values above 1 approximate "
        "G-Eval's expected score (default: %(default)s)."))
    tier3.add_argument("--judge-timeout", type=float, default=300.0,
                       help="Per-request timeout in seconds (default: %(default)s).")
    tier3.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC,
                       help="Rubric YAML (default: %(default)s).")

    contrast = parser.add_argument_group("configuration contrast")
    contrast.add_argument("--compare-run-dir", type=Path, help=(
        "Second run directory. Its responses are compared with the primary run "
        "pairwise, in both presentation orders, to control position bias."))
    contrast.add_argument("--pairwise-dimension", action="append", default=[],
                          help=("Rubric dimension id to compare on. Repeatable; "
                                "defaults to relevance, spoken_comprehensibility "
                                "and factual_accuracy."))

    tier4 = parser.add_argument_group("tier 4: reliability and calibration")
    tier4.add_argument("--human-annotations", type=Path, help=(
        "Long-format CSV of expert ratings with columns "
        "item_id,rater_id,dimension,score."))

    policy = parser.add_argument_group("acceptance policy (predeclare these)")
    policy.add_argument("--min-quality", type=float, default=3.5,
                        help="Minimum rubric composite (default: %(default)s).")
    policy.add_argument("--min-safety", type=float, default=4.0, help=(
        "Minimum score on the worst safety dimension (default: %(default)s)."))
    policy.add_argument("--loose-constraints", action="store_true", help=(
        "Gate acceptance on the loose rather than the strict constraint "
        "verdict, ignoring formatting-only violations."))
    policy.add_argument("--no-constraint-gate", action="store_true",
                        help="Do not require constraint adherence for acceptance.")

    output = parser.add_argument_group("output")
    output.add_argument("--out-dir", type=Path, help=(
        "Output directory. Defaults to <run-dir>/evaluation, or "
        "./evaluation_output/<timestamp> without a run directory."))
    output.add_argument("--emit-bibtex", action="store_true", help=(
        "Write BibTeX entries for the methods used, ready to append to a .bib."))
    output.add_argument("--seed", type=int, default=0,
                        help="Seed for bootstrap resampling (default: %(default)s).")
    output.add_argument("--quiet", action="store_true",
                        help="Do not print the report to stdout.")
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    """Translate parsed arguments into the configuration the library takes."""
    return EvaluationConfig(
        run_dir=args.run_dir,
        transcripts=args.transcripts,
        spec=args.spec,
        answer_key=args.answer_key,
        system_prompt=args.system_prompt,
        label=args.label,
        constraints=args.constraints,
        max_tokens=args.max_tokens,
        embedding_model=args.embedding_model,
        selfcheck_samples=args.selfcheck_samples,
        selfcheck_model=args.selfcheck_model,
        selfcheck_temperature=args.selfcheck_temperature,
        fact_precision=args.fact_precision,
        max_claims=args.max_claims,
        judge_models=list(args.judge_model),
        judge_url=args.judge_url,
        judge_samples=args.judge_samples,
        judge_timeout=args.judge_timeout,
        rubric=args.rubric,
        compare_run_dir=args.compare_run_dir,
        pairwise_dimensions=list(args.pairwise_dimension),
        human_annotations=args.human_annotations,
        min_quality=args.min_quality,
        min_safety=args.min_safety,
        loose_constraints=args.loose_constraints,
        constraint_gate=not args.no_constraint_gate,
        seed=args.seed,
        progress=None if args.quiet else _stderr_progress)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _use_utf8_output()
    args = build_parser().parse_args(argv)

    try:
        outcome = run_evaluation(config_from_args(args))
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for warning in outcome.warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    out_dir = Path(args.out_dir) if args.out_dir else outcome.default_out_dir()
    outcome.write(out_dir, emit_bibtex=args.emit_bibtex)

    if not args.quiet:
        print(outcome.report)
    print(f"\nArtefacts written to: {out_dir.resolve()}", file=sys.stderr)
    return 0


def _stderr_progress(message: str) -> None:
    """Report progress on stderr, since the model tiers take minutes."""
    print(f"  ... {message}", file=sys.stderr, flush=True)


def _use_utf8_output() -> None:
    """Print UTF-8 regardless of the console code page.

    Model output contains curly quotes and dashes. On a legacy Windows code page
    those either raise or arrive as mojibake, which would make a report quoting
    the response unreadable in the very cases worth inspecting. The report files
    are written as UTF-8 either way.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
