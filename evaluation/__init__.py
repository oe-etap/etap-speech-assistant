#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response evaluation toolkit for the on-host ASR-LLM-TTS pipeline.

The package scores (user utterance, assistant response) pairs produced by a
pipeline run, using published evaluation methods rather than an ad-hoc rubric.
It is organized in tiers of decreasing certainty and increasing cost:

    constraints  Tier 0  Verifiable prompt-adherence checks and readability.
                         Decidable by construction; no calibration needed.
    asr                  Recognizer fidelity against the reference utterance:
                         what the model was given, as opposed to what was asked.
    latency              Per-item timing and resource cost, read from the run's
                         own latency log, so quality and cost share a table.
    relevance    Tier 1  Relevance screening and reference-based overlap.
    factuality   Tier 2  Hallucination evidence: deterministic audit atoms,
                         self-consistency resampling, atomic claim verification.
    judge        Tier 3  Rubric grading by one or more local models.
    agreement    Tier 4  Human inter-rater reliability, and calibration of the
                         judge against human ratings.

Tier 0 runs with no model and no network. Tiers 2 and 3 need a local Ollama
server; nothing is sent off the host. Tier 4 needs human annotations.

The package is self-contained and can be called two ways.

From a shell:

    python -m evaluation --run-dir outputs/20260809_164356

From Python:

    from evaluation import EvaluationConfig, run_evaluation

    outcome = run_evaluation(EvaluationConfig(run_dir="outputs/20260809_164356"))
    print(outcome.summary.acceptance_rate)
    outcome.write("outputs/20260809_164356/evaluation")

`run_evaluation` neither writes files nor prints; the caller decides both. The
command line is a thin wrapper over the same call.

To decide whether a parameter change improved the responses, compare two runs
over the same prompts instead of reading two reports side by side:

    python -m evaluation.comparison --baseline runs/before --contrast runs/after

    from evaluation import ComparisonConfig, compare_runs

    outcome = compare_runs(ComparisonConfig(baseline="runs/before",
                                            contrast=["runs/after"]))

The comparison is paired item by item and needs no human labels, which is what
makes it usable on a prompt set too large to read.

A grid of configurations is evaluated and compared in one pass, which is the form
a parameter sweep actually arrives in:

    python -m evaluation.batch --root results/text-only

    from evaluation import BatchConfig, run_batch

    outcome = run_batch(BatchConfig(root="results/text-only"))
    outcome.write("results/evaluation_result")

Every run is scored once and then enters every contrast it belongs to, so a run's
figures are the same in each table it appears in.
"""

from .pipeline import EvaluationConfig, EvaluationOutcome, run_evaluation

# Imported on first use rather than here. Importing either module at package
# import time places it in sys.modules before `python -m evaluation.comparison`
# executes it as __main__, so it runs twice and two sets of its dataclasses end up
# in play. Deferring keeps `from evaluation import compare_runs` working without
# that.
_LAZY_COMPARISON = frozenset({"ComparisonConfig", "ComparisonOutcome",
                              "compare_runs", "compare_evaluated"})

_LAZY_BATCH = frozenset({"BatchConfig", "BatchOutcome", "ContrastGroup",
                         "RunIdentity", "build_groups", "discover_runs",
                         "run_batch"})


def __getattr__(name: str):
    if name in _LAZY_COMPARISON:
        from . import comparison
        return getattr(comparison, name)
    if name in _LAZY_BATCH:
        from . import batch
        return getattr(batch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvaluationConfig",
    "EvaluationOutcome",
    "run_evaluation",
    "ComparisonConfig",
    "ComparisonOutcome",
    "compare_runs",
    "compare_evaluated",
    "BatchConfig",
    "BatchOutcome",
    "ContrastGroup",
    "RunIdentity",
    "build_groups",
    "discover_runs",
    "run_batch",
    "aggregation",
    "agreement",
    "asr",
    "batch",
    "cli",
    "comparison",
    "constraints",
    "factuality",
    "judge",
    "latency",
    "loaders",
    "ollama_client",
    "pipeline",
    "references",
    "relevance",
    "reporting",
    "stats",
    "textutils",
]

__version__ = "1.0.0"
