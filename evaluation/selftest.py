#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-test for the evaluation toolkit.

Run with:  python -m evaluation.selftest

The reliability statistics and the prediction-powered estimator are the parts of
this package that fail silently: a wrong coefficient still returns a plausible
number between zero and one, and nothing downstream notices. They are therefore
checked against worked examples with published values, or against cases whose
answer is derivable by hand, rather than against whatever the implementation
happened to produce first.

The model-backed tiers are exercised with a stub client so that parsing,
aggregation and the position-bias control are testable without a running Ollama
server. No test here needs a network connection.
"""

import math
import sys
from typing import Dict, List, Optional

import requests

from . import textutils as tu
from .agreement import (cohen_kappa, gwet_ac1, icc_two_way, krippendorff_alpha,
                        percent_agreement, prediction_powered_mean)
from .constraints import ConstraintSpec, evaluate_constraints
from .factuality import extract_audit_atoms, run_selfcheck
from .judge import Rubric, _parse_score, compare_pairwise
from .loaders import Record, build_records
from .ollama_client import (GenerationResult, OllamaClient, extract_json_object,
                           reset_probe_cache)
from .relevance import request_coverage, rouge_l, token_f1
from .stats import (cliffs_delta, holm_bonferroni, mcnemar,
                    paired_mean_difference_ci, paired_tost, spearman,
                    tost_equivalence, wilcoxon_signed_rank)

_FAILURES: List[str] = []
_CHECKS = 0


def check(condition: bool, description: str) -> None:
    """Record the outcome of one assertion without aborting the run.

    Collecting failures rather than raising means one broken statistic does not
    hide the state of the others.
    """
    global _CHECKS
    _CHECKS += 1
    if not condition:
        _FAILURES.append(description)


def close(actual: Optional[float], expected: float, tolerance: float,
          description: str) -> None:
    """Check a numeric result against a reference value."""
    if actual is None:
        _FAILURES.append(f"{description}: got None, expected {expected}")
        global _CHECKS
        _CHECKS += 1
        return
    check(abs(actual - expected) <= tolerance,
          f"{description}: got {actual!r}, expected {expected} "
          f"(tolerance {tolerance})")


# --------------------------------------------------------------------------
# Text primitives
# --------------------------------------------------------------------------

def test_sentence_splitting() -> None:
    # A single-letter initial must not end a sentence, or every response naming a
    # person is credited with an extra sentence and a lower mean sentence length.
    sentences = tu.split_sentences(
        "Absolutely, I suggest '2001: A Space Odyssey' by Arthur C. Clarke. "
        "This novel is long.")
    check(len(sentences) == 2,
          f"initials must not split a sentence: got {len(sentences)}: {sentences}")

    check(len(tu.split_sentences("Pi is about 3.14 in value. Yes.")) == 2,
          "a decimal point must not split a sentence")

    check(len(tu.split_sentences("Visit Dr. Smith today. Then rest.")) == 2,
          "a known abbreviation must not split a sentence")

    # Semicolons are clause boundaries, not sentence boundaries: a prompt asking
    # for a sentence ending in a full stop is about . ! ? only.
    check(len(tu.split_sentences(
        "Machine learning focuses on algorithms; AI is broader.")) == 1,
        "a semicolon must not split a sentence")

    check(tu.word_count("Canberra, Australia's political and administrative "
                        "center since 1948.") == 8,
          "possessives and years each count as one word")


def test_readability() -> None:
    # Flesch Reading Ease is defined on the ratios of words to sentences and
    # syllables to words, so a short plain sentence must land high and a long
    # polysyllabic one low. Absolute values depend on the syllable heuristic;
    # the ordering must not.
    easy = tu.flesch_reading_ease("The cat sat on the mat. The dog ran.")
    hard = tu.flesch_reading_ease(
        "Notwithstanding the aforementioned considerations, the "
        "implementation demonstrates substantial methodological "
        "inconsistencies requiring comprehensive reevaluation.")
    check(easy > hard, f"reading ease must order plain above complex text: "
                       f"{easy:.1f} vs {hard:.1f}")
    check(math.isnan(tu.flesch_reading_ease("")),
          "readability of an empty response is undefined, not zero")


# --------------------------------------------------------------------------
# Tier 0
# --------------------------------------------------------------------------

def test_constraints() -> None:
    spec = ConstraintSpec.load(
        __file__.replace("selftest.py", "rubrics/constraints_short_opener.yaml"))

    compliant = Record(item_id="ok", stt_text="what is the capital of france",
                       llm_text="Paris. It has been the capital since 1944, and "
                                "it is the largest city in France.")
    outcome = evaluate_constraints(compliant, spec, {"max_tokens": 150})
    check(outcome.item_level_strict,
          "a compliant response must pass every hard constraint, failed: "
          f"{outcome.failures()}")

    # Long opener, too many detail sentences, and an unterminated ending at a
    # token cap the response plausibly reached: four distinct violations in one
    # response, which is what the per-check breakdown has to separate.
    violating = Record(
        item_id="bad", stt_text="tell me about london",
        llm_text="Eiffel Tower, but it is fictional in London; start with Big "
                 "Ben instead. It is a landmark. It is in Paris. Although it is "
                 "not in London,")
    outcome = evaluate_constraints(violating, spec, {"max_tokens": 40})
    failures = set(outcome.failures())
    for expected in ("opener_at_most_five_words", "at_most_two_detail_sentences",
                     "total_sentences_at_most_three",
                     "response_is_complete", "not_truncated_at_token_cap"):
        check(expected in failures,
              f"expected {expected} to fail; failures were {sorted(failures)}")

    # The prompt says "if that fully answers the user, stop there", so a bare
    # opener is the intended best case and must not be scored as a violation.
    # This is the regression that a detail-sentence minimum of one introduces:
    # it fails almost every item of a run whose answers are short by design.
    bare = Record(item_id="bare", stt_text="what fraction remained",
                  llm_text="Sixty percent.")
    outcome = evaluate_constraints(bare, spec, {"max_tokens": 150})
    check(outcome.item_level_strict,
          "a bare opener that answers the question must pass, failed: "
          f"{outcome.failures()}")

    # The language check needs enough tokens for a proportion to mean anything.
    # "Sixty percent." is English and contains no function word, so scoring it
    # would measure the marker list rather than the reply.
    language = [r for r in outcome.results if r.check_id == "answers_in_english"]
    check(language and not language[0].applicable,
          "the language check must be inapplicable on a two-word reply")

    # Above the guard it applies again, and still catches a real switch.
    swapped = Record(item_id="de", stt_text="what is the capital of france",
                     llm_text="Paris ist seit dem Jahr neunzehnhundert"
                              " vierundvierzig die Hauptstadt.")
    outcome = evaluate_constraints(swapped, spec, {"max_tokens": 150})
    check("answers_in_english" in outcome.failures(),
          "a wholesale language switch above the guard must still fail")

    # A simulated turn label is the failure that reaches the user as speech.
    dialogue = Record(item_id="turn", stt_text="hello",
                      llm_text="Hello there.\nUser: and then?\nAssistant: yes.")
    outcome = evaluate_constraints(dialogue, spec, {})
    check("no_simulated_dialogue" in outcome.failures(),
          "a turn label must fail the simulated-dialogue check")

    # Conditional checks must be inapplicable, not passing, when the scenario
    # spec supplies nothing for them.
    inert = [r for r in outcome.results if r.check_type == "must_include"]
    check(inert and not inert[0].applicable,
          "must_include with no requirement must be marked not applicable")

    # Bold markers glued to the opener stop it from terminating a sentence, so
    # the whole response reads as one long opening sentence. This is a formatting
    # violation, not a length violation, and the strict/loose pair exists to tell
    # the two apart: strict must fail, loose must pass once the markup is removed.
    wrapped = Record(item_id="md", stt_text="what is the capital of france",
                     llm_text="**Paris.** It is the largest city in France.")
    outcome = evaluate_constraints(wrapped, spec, {})
    opener = next(r for r in outcome.results
                  if r.check_id == "opener_at_most_five_words")
    check(not opener.passed_strict,
          "markup that suppresses the sentence boundary must fail strictly")
    check(opener.passed_loose,
          "the same response must pass loosely once the markup is stripped")


def test_audit_atoms() -> None:
    atoms = extract_audit_atoms(
        "Canberra, Australia's political centre since 1948. Gustave Eiffel "
        "designed it in 12 years.")
    check("1948" in atoms.years, f"years not extracted: {atoms.years}")
    check(any("Gustave Eiffel" in span for span in atoms.name_spans),
          f"name span not extracted: {atoms.name_spans}")
    check(atoms.count > 0, "audit atoms must be found in a factual response")


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------

def test_relevance() -> None:
    close(token_f1("Paris", "Paris")["f1"], 1.0, 1e-9,
          "token F1 of identical answers")
    close(token_f1("Paris", "London")["f1"], 0.0, 1e-9,
          "token F1 of disjoint answers")
    # SQuAD normalization strips articles, so these must be identical.
    close(token_f1("the capital is Paris", "capital is Paris")["f1"], 1.0, 1e-9,
          "token F1 must ignore articles")
    close(rouge_l("a b c d", "a b c d"), 1.0, 1e-9, "ROUGE-L of identical text")

    # The possessive in the response must match the base form in the request,
    # otherwise coverage understates how much of the request was addressed.
    coverage = request_coverage(
        "what is the capital of australia",
        "Canberra, Australia's political and administrative center.")
    check(coverage is not None and coverage > 0.0,
          f"possessive forms must count towards request coverage: {coverage}")

    off_topic = request_coverage("what is the capital of australia",
                                "I enjoy baking bread on weekends.")
    check(off_topic is not None and off_topic < 0.2,
          f"an off-topic response must score low coverage: {off_topic}")


# --------------------------------------------------------------------------
# Tier 3 parsing
# --------------------------------------------------------------------------

def test_json_recovery() -> None:
    cases = [
        ('{"score": 4, "reason": "clear"}', 4),
        ('Here is my rating:\n```json\n{"score": 2, "reason": "vague"}\n```', 2),
        ("Sure! {'score': 5, 'reason': 'excellent'} Hope that helps.", 5),
        ('{"score": 3, "reason": "ok",}', 3),
        ('{"score": "4"}', 4),
        ('{"score": 4.0}', 4),
        ("I would say 3 out of 5.", 3),
        ('{"reason": "text with } brace", "score": 1}', 1),
    ]
    for raw, expected in cases:
        score, _ = _parse_score(raw, 1, 5)
        check(score == expected,
              f"score recovery from {raw!r}: got {score}, expected {expected}")

    check(extract_json_object("no json at all") is None,
          "text without JSON must yield None rather than a partial object")
    # An out-of-range integer must not be accepted as a rating.
    score, _ = _parse_score('{"score": 9}', 1, 5)
    check(score is None, f"out-of-range rating must be rejected, got {score}")


def test_rubric_loads() -> None:
    rubric = Rubric.load(
        __file__.replace("selftest.py", "rubrics/quest_therapy_v1.yaml"))
    check(len(rubric.dimensions) >= 10,
          f"rubric should define the full dimension set, got "
          f"{len(rubric.dimensions)}")
    check(len(rubric.safety_dimensions) >= 3,
          "rubric must mark safety-critical dimensions for the gate")
    for dimension in rubric.dimensions:
        check(set(dimension.anchors) == {1, 2, 3, 4, 5},
              f"dimension {dimension.dim_id} must anchor every scale point, "
              f"has {sorted(dimension.anchors)}")
        check(bool(dimension.definition),
              f"dimension {dimension.dim_id} needs a definition")

    # Therapy-specific dimensions must not be scored on open-domain items, or the
    # run reports empathy figures for questions that express nothing.
    smoke = Record(item_id="s", stt_text="hi", llm_text="Hello.",
                   category="open_domain_smoke_test")
    empathy = next(d for d in rubric.dimensions
                   if d.dim_id == "empathic_nonjudgmental")
    check(not empathy.applicable_to(smoke),
          "empathy must be skipped for open-domain smoke-test items")


# --------------------------------------------------------------------------
# Reliability statistics against worked examples
# --------------------------------------------------------------------------

def _two_rater_table(counts: Dict[tuple, int]) -> Dict[str, Dict[str, float]]:
    """Build a ratings table from a 2x2 contingency count dictionary."""
    table: Dict[str, Dict[str, float]] = {}
    index = 0
    for (value_a, value_b), n in counts.items():
        for _ in range(n):
            index += 1
            table[f"i{index:03d}"] = {"A": float(value_a), "B": float(value_b)}
    return table


def test_cohen_kappa_and_ac1() -> None:
    # Textbook 2x2 table: 20 both-yes, 15 both-no, 5 and 10 disagreements.
    # Observed agreement 0.70; expected 0.50; kappa therefore exactly 0.40.
    table = _two_rater_table({(1, 1): 20, (1, 0): 5, (0, 1): 10, (0, 0): 15})
    close(percent_agreement(table), 0.70, 1e-9, "observed agreement")
    close(cohen_kappa(table), 0.40, 1e-9, "Cohen's kappa on the 2x2 example")

    # Gwet's AC1 on the same table. Chance agreement is
    # (1/(q-1)) * sum_k pi_k (1 - pi_k) with pi_yes = 0.55, giving 0.495, so
    # AC1 = (0.70 - 0.495) / (1 - 0.495).
    close(gwet_ac1(table), (0.70 - 0.495) / (1 - 0.495), 1e-9,
          "Gwet's AC1 on the 2x2 example")

    # The prevalence paradox: with almost every item a pass, kappa collapses
    # while AC1 stays high. This is the case that makes AC1 the statistic to
    # read on safety categories, so the implementations must reproduce it.
    skewed = _two_rater_table({(1, 1): 95, (1, 0): 2, (0, 1): 2, (0, 0): 1})
    kappa = cohen_kappa(skewed)
    ac1 = gwet_ac1(skewed)
    check(kappa is not None and ac1 is not None and ac1 > kappa + 0.3,
          f"AC1 must exceed kappa markedly under high prevalence: "
          f"kappa={kappa}, AC1={ac1}")


def test_krippendorff_alpha() -> None:
    # Three observers, twelve units, missing cells, ratings on 1-5. Expected
    # values are derived by hand for this exact matrix rather than quoted, so the
    # test verifies the implementation independently of any transcription.
    #
    # Ten units carry at least two ratings, giving n = 28 pairable values with
    # marginal counts n_1..n_5 = 5, 10, 8, 3, 2. Only u02, u06 and u08 contain
    # disagreement.
    #
    # Nominal:  D_o = 7/28,   D_e = (28^2 - sum n_k^2)/(28*27) = 582/756,
    #           alpha = 1 - (7/28)/(582/756)      = 0.675258
    # Ordinal:  D_o = 684/28, D_e = 94640/756,
    #           alpha = 1 - (684/28)/(94640/756)  = 0.804861
    data = {
        "u01": {"A": 1, "B": 1},
        "u02": {"A": 2, "B": 2, "C": 3},
        "u03": {"A": 3, "B": 3, "C": 3},
        "u04": {"A": 3, "B": 3, "C": 3},
        "u05": {"A": 2, "B": 2, "C": 2},
        "u06": {"A": 1, "B": 2, "C": 3},
        "u07": {"A": 4, "B": 4, "C": 4},
        "u08": {"A": 1, "B": 1, "C": 2},
        "u09": {"A": 2, "B": 2, "C": 2},
        "u10": {"B": 5, "C": 5},
        "u12": {"B": 3},
    }
    close(krippendorff_alpha(data, "nominal"), 1 - (7 / 28) / (582 / 756), 1e-9,
          "Krippendorff's alpha, nominal metric")
    close(krippendorff_alpha(data, "ordinal"),
          1 - (684 / 28) / (94640 / 756), 1e-9,
          "Krippendorff's alpha, ordinal metric")

    # The ordinal metric must exceed the nominal one on this matrix: every
    # disagreement here is between adjacent or near-adjacent categories, which
    # the ordinal metric penalises less than an outright category mismatch.
    check(krippendorff_alpha(data, "ordinal")
          > krippendorff_alpha(data, "nominal"),
          "ordinal alpha must exceed nominal alpha for near-miss disagreement")

    perfect = {"a": {"r1": 3, "r2": 3}, "b": {"r1": 5, "r2": 5},
               "c": {"r1": 1, "r2": 1}}
    close(krippendorff_alpha(perfect, "ordinal"), 1.0, 1e-9,
          "alpha under perfect agreement")

    # Systematic disagreement must give a negative coefficient, not zero.
    inverted = {"a": {"r1": 1, "r2": 5}, "b": {"r1": 5, "r2": 1},
                "c": {"r1": 1, "r2": 5}, "d": {"r1": 5, "r2": 1}}
    alpha = krippendorff_alpha(inverted, "nominal")
    check(alpha is not None and alpha < 0,
          f"systematic disagreement must give negative alpha, got {alpha}")


def test_icc() -> None:
    # Shrout and Fleiss (1979) worked example: six targets, four judges.
    # Published ICC(2,1) = 0.290.
    rows = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
            [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    table = {f"t{i}": {f"j{j}": float(value) for j, value in enumerate(row, 1)}
             for i, row in enumerate(rows, 1)}

    result = icc_two_way(table)
    check(result is not None, "ICC must be computable on a complete matrix")
    if result is not None:
        close(result.icc_single, 0.290, 0.01, "ICC(2,1) on the Shrout-Fleiss data")
        check(result.icc_average is not None
              and result.icc_average > result.icc_single,
              "ICC(2,k) must exceed ICC(2,1): averaging raters raises reliability")
        check(result.n_items == 6 and result.n_raters == 4,
              "ICC must report the sample it used")

    # Incomplete rows must be dropped and counted, not silently imputed.
    table["t7"] = {"j1": 5.0}
    partial = icc_two_way(table)
    check(partial is not None and partial.dropped_items == 1,
          "an incompletely rated item must be reported as dropped")


def test_prediction_powered_inference() -> None:
    # A judge with a constant bias of +1. The rectifier must remove exactly that
    # bias, so the estimate equals the mean human rating over the unlabelled
    # items, not the judge's mean. This is the property the whole tier rests on.
    human_labelled = [3.0, 4.0, 2.0, 5.0, 3.0, 4.0]
    judge_labelled = [value + 1.0 for value in human_labelled]
    judge_unlabelled = [4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]
    implied_human_unlabelled = [value - 1.0 for value in judge_unlabelled]

    outcome = prediction_powered_mean(human_labelled, judge_labelled,
                                      judge_unlabelled)
    check(outcome is not None, "PPI must be computable with both subsets present")
    if outcome is not None:
        estimate, (low, high) = outcome
        expected = sum(implied_human_unlabelled) / len(implied_human_unlabelled)
        close(estimate, expected, 1e-9,
              "PPI must correct a constant judge bias exactly")
        check(low < estimate < high, "PPI interval must contain its estimate")
        judge_mean = sum(judge_unlabelled) / len(judge_unlabelled)
        check(not (low <= judge_mean <= high),
              "PPI interval must exclude the biased judge mean in this "
              f"construction: {low:.3f}-{high:.3f} vs {judge_mean:.3f}")

    check(prediction_powered_mean([3.0], [4.0], [4.0, 5.0]) is None,
          "PPI must decline to report on too few labelled items")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def test_statistics() -> None:
    close(cliffs_delta([5, 5, 5], [1, 1, 1]), 1.0, 1e-9,
          "Cliff's delta under complete dominance")
    close(cliffs_delta([1, 1, 1], [5, 5, 5]), -1.0, 1e-9,
          "Cliff's delta under complete reverse dominance")
    close(cliffs_delta([1, 2, 3], [1, 2, 3]), 0.0, 1e-9,
          "Cliff's delta for identical samples")

    # Two samples with the same mean must be judged equivalent within a wide
    # margin, and not equivalent within a margin far tighter than the noise.
    left = [4.0, 4.1, 3.9, 4.0, 4.2, 3.8, 4.0, 4.1]
    right = [4.0, 3.9, 4.1, 4.0, 3.8, 4.2, 4.1, 3.9]
    wide = tost_equivalence(left, right, margin=0.5)
    check(wide is not None and wide.equivalent,
          "identical distributions must be equivalent within a wide margin")

    far = [1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.1, 0.9]
    narrow = tost_equivalence(left, far, margin=0.2)
    check(narrow is not None and not narrow.equivalent,
          "clearly different distributions must not be declared equivalent")

    adjusted = holm_bonferroni([0.001, 0.02, 0.04, 0.5])
    check(adjusted[0]["p_holm"] <= adjusted[1]["p_holm"] <= adjusted[2]["p_holm"],
          "Holm adjustment must be monotone in the ordered p-values")
    close(adjusted[0]["p_holm"], 0.004, 1e-9,
          "smallest p-value scaled by the number of tests")

    close(spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]), 1.0, 1e-9,
          "Spearman correlation of a monotone relation")
    close(spearman([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]), -1.0, 1e-9,
          "Spearman correlation of an inverse relation")


# --------------------------------------------------------------------------
# Model-backed tiers, exercised with a stub client
# --------------------------------------------------------------------------

class _StubClient(OllamaClient):
    """An OllamaClient that replays canned responses instead of calling a server."""

    def __init__(self, responses: List[str]):
        super().__init__(model="stub", url="http://stub/api/generate")
        self._responses = responses
        self._index = 0
        self.prompts: List[str] = []

    def generate(self, prompt, temperature=None, seed=None, num_predict=None,
                 stop=None, timeout=None):
        self.prompts.append(prompt)
        text = self._responses[self._index % len(self._responses)]
        self._index += 1
        return GenerationResult(text=text, ok=True, eval_count=len(text.split()))


def test_selfcheck_scoring() -> None:
    record = Record(item_id="x", stt_text="what is the capital of australia",
                    llm_text="Canberra is the capital of Australia.")

    # Every resample repeats the response, so the sentence is fully supported and
    # the inconsistency must be near zero.
    consistent = _StubClient(["Canberra is the capital of Australia."] * 4)
    outcome = run_selfcheck(record, consistent, "system", n_samples=4)
    check(outcome.mean_inconsistency is not None
          and outcome.mean_inconsistency < 0.1,
          f"a fully supported sentence must score near zero: "
          f"{outcome.mean_inconsistency}")

    # No resample supports the claim, so inconsistency must be high.
    contradicting = _StubClient(["Sydney hosts the government offices.",
                                 "Melbourne holds the parliament today.",
                                 "Perth serves that role instead.",
                                 "Brisbane fills the position now."])
    outcome = run_selfcheck(record, contradicting, "system", n_samples=4)
    check(outcome.mean_inconsistency is not None
          and outcome.mean_inconsistency > 0.5,
          f"an unsupported sentence must score high: "
          f"{outcome.mean_inconsistency}")
    check(outcome.kernel == "token_f1_surrogate",
          "the support kernel must be named in the result, since the default "
          "is a surrogate for the published one")

    # The resampling prompt must reproduce the pipeline's framing, otherwise the
    # samples are not drawn from the distribution the response came from.
    check(any('The user said: "' in prompt for prompt in contradicting.prompts),
          "resampling must use the pipeline's prompt framing")


def test_pairwise_position_bias() -> None:
    baseline = Record(item_id="p", stt_text="which city", llm_text="Answer one.")
    contrast = Record(item_id="p", stt_text="which city", llm_text="Answer two.")

    # A judge that always names the first option shows pure position bias. Both
    # orders then nominate different responses, so the verdict must be recorded
    # as inconsistent and resolved to a tie rather than a win.
    biased = _StubClient(['{"winner": "A", "reason": "first"}'])
    verdict = compare_pairwise(biased, baseline, contrast, "overall quality")
    check(not verdict.consistent,
          "a judge that always picks position A must be flagged inconsistent")
    check(verdict.verdict == "tie",
          f"an order-inconsistent comparison must resolve to a tie, got "
          f"{verdict.verdict}")

    # A judge that prefers the same text in both orders is order-consistent, and
    # the swapped verdict must be remapped back to the baseline's label.
    stable = _StubClient(['{"winner": "A", "reason": "better"}',
                          '{"winner": "B", "reason": "better"}'])
    verdict = compare_pairwise(stable, baseline, contrast, "overall quality")
    check(verdict.consistent and verdict.verdict == "A",
          f"a stable preference for the baseline must survive the swap, got "
          f"{verdict.verdict} (consistent={verdict.consistent})")


def test_paired_statistics() -> None:
    # McNemar looks only at the items whose verdict changed. Here eight failures
    # became passes and one pass became a failure; the twenty items that did not
    # change carry no information about the change and must not dilute it.
    baseline = [False] * 9 + [True] * 21
    contrast = [True] * 8 + [False] + [True] * 20 + [False]
    result = mcnemar(baseline, contrast)
    check(result is not None and result.n_effective == 9,
          f"McNemar must count only discordant pairs, got "
          f"{result.n_effective if result else None}")
    # Exact two-sided binomial for 8 versus 1 out of 9: 2 * 10/512.
    check(result is not None and abs(result.p_value - 2 * 10 / 512) < 1e-12,
          f"McNemar exact p must be 2*10/512, got "
          f"{result.p_value if result else None}")

    unchanged = mcnemar([True, False, True], [True, False, True])
    check(unchanged is not None and unchanged.p_value == 1.0
          and unchanged.n_effective == 0,
          "an unchanged binary outcome must give p = 1 and no effective pairs")

    # Wilcoxon on a textbook-shaped sample: differences +1..+5 and one -1.
    # Ranks of |d|: the two 1s share rank 1.5, then 3, 4, 5, 6. W- is the single
    # negative difference at rank 1.5, so the statistic is 1.5.
    differences = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0]
    signed = wilcoxon_signed_rank(differences)
    check(signed is not None and abs(signed.statistic - 1.5) < 1e-9,
          f"Wilcoxon statistic must be the smaller signed-rank sum 1.5, got "
          f"{signed.statistic if signed else None}")
    check(signed is not None and signed.n_effective == 6,
          "every non-zero difference must count as an informative pair")

    zeros = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    check(zeros is not None and zeros.p_value == 1.0 and zeros.n_effective == 0,
          "identical runs must produce no evidence of a difference")

    # A consistent shift must be detected, and its direction preserved.
    improving = wilcoxon_signed_rank([0.4] * 30)
    check(improving is not None and improving.p_value < 0.001,
          f"a consistent shift over 30 pairs must be significant, got "
          f"{improving.p_value if improving else None}")

    # The paired interval must exclude zero for that shift, and contain it for
    # noise that averages out.
    low, high = paired_mean_difference_ci([0.4] * 30, n_boot=500, seed=0)
    check(low is not None and low > 0,
          f"a uniform improvement must give an interval above zero, got {low}")
    noise = [0.5, -0.5] * 20
    low, high = paired_mean_difference_ci(noise, n_boot=500, seed=0)
    check(low is not None and low < 0 < high,
          f"symmetric noise must give an interval spanning zero, got "
          f"[{low}, {high}]")

    # Equivalence: a tiny but perfectly consistent difference is significant and
    # yet equivalent within a margin that says it does not matter. This is the
    # case a significance test alone reports misleadingly on a large sample.
    tiny = [0.01] * 200
    detected = wilcoxon_signed_rank(tiny)
    equivalence = paired_tost(tiny, margin=0.1)
    check(detected is not None and detected.p_value < 0.05,
          "a consistent tiny difference is still detectable")
    check(equivalence is not None and equivalence.equivalent,
          "a tiny difference must be judged equivalent within a 0.1 margin")

    # Cliff's delta must agree between the pairwise and the rank-based paths,
    # since the rank path exists only to keep large runs computable.
    from .stats import _rank
    treatment = [5.0, 3.0, 4.0, 4.0, 2.0]
    control = [3.0, 3.0, 1.0, 4.0]
    direct = cliffs_delta(treatment, control)
    ranks = _rank(treatment + control)
    u = sum(ranks[:len(treatment)]) - len(treatment) * (len(treatment) + 1) / 2
    by_rank = 2 * u / (len(treatment) * len(control)) - 1
    check(abs(direct - by_rank) < 1e-12,
          f"the two Cliff's delta paths must agree: {direct} vs {by_rank}")


class _RefusingSession:
    """A requests session stand-in that always refuses, and counts attempts."""

    def __init__(self, error: Exception):
        self._error = error
        self.attempts = 0

    def post(self, *args, **kwargs):
        self.attempts += 1
        raise self._error


def test_transport_retry_policy() -> None:
    # A refused connection means nothing is listening, so retrying only adds
    # delay before the tier reports that it cannot run. A read timeout may pass,
    # so it keeps its retries.
    refused = _RefusingSession(requests.exceptions.ConnectionError("refused"))
    client = OllamaClient(model="m", url="http://stub/api/generate", retries=2)
    client._session = refused
    result = client.generate("hi")
    check(not result.ok, "a refused connection must be reported as a failure")
    check(refused.attempts == 1,
          f"a refused connection must not be retried, got {refused.attempts} "
          f"attempts")

    timing_out = _RefusingSession(requests.exceptions.ReadTimeout("slow"))
    client = OllamaClient(model="m", url="http://stub/api/generate", retries=2)
    client._session = timing_out
    client.generate("hi")
    check(timing_out.attempts == 3,
          f"a timeout must use every retry, got {timing_out.attempts} attempts")

    # Several tiers probe the same server and model; the verdict is shared so
    # that an unreachable server is waited for once, not once per tier.
    reset_probe_cache()
    counting = _RefusingSession(requests.exceptions.ConnectionError("refused"))
    client = OllamaClient(model="m", url="http://stub/api/generate", retries=2)
    client._session = counting
    first, second = client.probe(), client.probe()
    check(first is not None and first == second,
          "a repeated probe must return the same verdict")
    check(counting.attempts == 1,
          f"a repeated probe must not re-contact the server, got "
          f"{counting.attempts} attempts")
    reset_probe_cache()


def test_record_building() -> None:
    entries = [{"stt_text": "hello there", "llm_text": "Hi."},
               {"stt_text": "second one", "llm_text": ""}]
    spec = {"hello there": {"category": "safety_critical_crisis",
                            "must_include": ["professional"],
                            "reference_answers": ["A greeting."]}}
    records = build_records(entries, spec)

    check(records[0].category == "safety_critical_crisis",
          "the scenario spec must attach by verbatim input text")
    check(records[0].safety_critical,
          "a crisis category must default to safety-critical")
    check(records[0].must_include == ["professional"],
          "required content must reach the record")
    check(records[1].is_empty_response,
          "an empty response must be detectable")
    check(records[1].item_id == "item002",
          f"positional ids must follow file order, got {records[1].item_id}")


def test_answer_key() -> None:
    """The dataset's answer columns, and what agreement with them may claim."""
    import tempfile
    from pathlib import Path as _Path

    from .aggregation import ItemResult, summarize_run
    from .loaders import load_answer_key
    from .relevance import answer_presence, evaluate_relevance, exact_match

    table = (
        "id,filename,question,answer,is_impossible,plausible_answers\n"
        # Answerable: the reference is the gold span in `answer`.
        "12,00012.wav,What did her mother own?,salon,False,\n"
        # Unanswerable: the reference is the plausible span, and the empty
        # `answer` must not be preferred over it.
        "30,00030.wav,What did he never develop?,,True,theology\n"
        # A quoted span, which a spreadsheet export wraps in quote characters
        # that are not part of the answer.
        "88,00088.wav,How many were there?,\"\"\"40,000\"\"\",True,\n"
        # A passage that bled into the answer column: the answer is what follows
        # the closing quote, and scoring the passage instead would be a defect
        # of the table reported as a property of the response.
        "298,00298.wav,Why gardens?,\"a long passage of prose.\"\",for medical "
        "use\",False,\n"
        # No answer in either column: the item must not be scored at all.
        "554,00554.wav,What happened?,,False,\n")

    with tempfile.TemporaryDirectory() as directory:
        path = _Path(directory) / "metadata.csv"
        path.write_text(table, encoding="utf-8")
        key = load_answer_key(path)

    check(key["00012"]["reference_answers"] == ["salon"],
          f"an answerable item must take the answer column, got {key['00012']}")
    check(key["00012"]["answer_unsupported"] is False,
          "an answerable item must not be marked unsupported")
    check(key["00030"]["reference_answers"] == ["theology"]
          and key["00030"]["answer_unsupported"] is True,
          f"an unanswerable item must take the plausible answer and be marked, "
          f"got {key['00030']}")
    check(key["00088"]["reference_answers"] == ["40,000"],
          f"export quotes must be stripped, got {key['00088']}")
    check(key["00298"]["reference_answers"] == ["for medical use"],
          f"a bled passage must yield the answer after it, got {key['00298']}")
    check("00554" not in key,
          "an item with no answer in either column must not be keyed")

    # The key attaches by recording, by id and by question text, because a
    # recognizer that misheard the question must still be scored against the
    # answer of the item it was given.
    check(key.get("00012.wav") is key.get("00012")
          is key.get("what did her mother own?"),
          "the key must attach by recording, by id and by question")

    entries = [{"filename": "00012.wav", "stt_text": "what could be once his "
                "mother own", "ori_text": "What did her mother own?",
                "llm_text": "Her mother owned a hair salon in Houston."},
               {"filename": "00030.wav", "stt_text": "what was one subject he "
                "never developed", "ori_text": "What did he never develop?",
                "llm_text": "He never developed a theory of optics."}]
    records = build_records(entries, {}, answers=key)

    check(records[0].reference_answers == ["salon"]
          and not records[0].answer_unsupported,
          f"the answer key must reach the record, got {records[0]}")
    check(records[1].answer_unsupported,
          "an unanswerable item must stay marked on the record")

    # A hand-written spec is the more specific statement and must win.
    spec = {"00012": {"reference_answers": ["a hair salon"]}}
    check(build_records(entries, spec, answers=key)[0].reference_answers
          == ["a hair salon"],
          "an explicit spec must override the corpus answer key")

    # Presence credits the answer inside a sentence; exact match does not, which
    # is why presence is the measure read for a conversational response.
    close(answer_presence("Her mother owned a hair salon.", "salon"), 1.0, 1e-9,
          "a gold span inside a sentence must count as present")
    close(answer_presence("Her mother owned a shop.", "salon"), 0.0, 1e-9,
          "a missing gold span must not count as present")
    close(answer_presence("She owned a salon and a hair studio", "hair salon"),
          0.0, 1e-9, "a span must match in order and uninterrupted")
    close(exact_match("Her mother owned a hair salon.", "salon"), 0.0, 1e-9,
          "a sentence must not exact-match a bare span")
    close(exact_match("A salon.", "salon"), 1.0, 1e-9,
          "normalization must ignore punctuation and articles")

    metrics = evaluate_relevance(records[0])
    close(metrics["answer_presence"], 1.0, 1e-9,
          "the correct answer must be scored as present")
    close(metrics["reference_answer_words"], 1.0, 1e-9,
          "the gold span length must be carried per item")
    check("answer_presence" not in evaluate_relevance(
        Record(item_id="x", stt_text="a", llm_text="b")),
        "an item without a reference must have no agreement figures")

    # The subsets are summarized apart: a rate that pools a verified answer with
    # a merely plausible one cannot be read as accuracy.
    results = [ItemResult(record=r) for r in records]
    for result in results:
        result.relevance = evaluate_relevance(result.record)
    accuracy = summarize_run("run", results).answer_accuracy

    check(accuracy["answerable_short_span"]["n_items"] == 1
          and accuracy["unanswerable_plausible"]["n_items"] == 1,
          f"answerable and unanswerable items must be counted apart, got "
          f"{accuracy}")
    close(accuracy["answerable_short_span"]["answer_presence_rate"], 1.0, 1e-9,
          "the answerable subset must report its own rate")
    close(accuracy["unanswerable_plausible"]["answer_presence_rate"], 0.0, 1e-9,
          "the unanswerable subset must report its own rate")


def test_word_error_rate() -> None:
    from .asr import (align, character_error_rate, content_recall,
                      evaluate_asr_fidelity, number_readings, stratum_of,
                      tokens, word_error_rate)

    # One substituted word out of six reference words.
    result = word_error_rate("the cat sat on the mat", "the cat sat on a mat")
    close(result.error_rate, 1 / 6, 1e-9, "one substitution in six words")
    check(result.substitutions == 1 and result.deletions == 0
          and result.insertions == 0,
          f"the edit must be counted as a substitution, got {result}")

    # A dropped word and an added word are different failures and are counted
    # separately: a deletion leaves the question incomplete, an insertion does not.
    dropped = word_error_rate("the cat sat on the mat", "the cat sat the mat")
    check(dropped.deletions == 1 and dropped.substitutions == 0,
          f"a missing word must count as a deletion, got {dropped}")
    added = word_error_rate("the cat sat", "the big cat sat")
    check(added.insertions == 1,
          f"an extra word must count as an insertion, got {added}")

    # Punctuation and case belong to the written reference, not to the audio.
    exact = word_error_rate("What is the campus TV station?",
                            "what is the campus tv station")
    check(exact.errors == 0,
          f"punctuation and case must not count as errors, got {exact}")

    # A recognizer that never writes digits is not wrong for saying the number
    # out loud, so the reference is credited with its best spoken reading.
    check("nineteen ninety" in number_readings("1990"),
          f"a four-digit year must offer its year reading, got "
          f"{number_readings('1990')}")
    year = word_error_rate("she was born in 1990", "she was born in nineteen ninety")
    check(year.errors == 0,
          f"a spoken year must match its numeral, got {year}")
    cardinal = word_error_rate("after 2004", "after two thousand four")
    check(cardinal.errors == 0,
          f"a spoken cardinal must match its numeral, got {cardinal}")

    # The character rate must be gentler than the word rate on a near miss,
    # which is the reason both are reported.
    check(character_error_rate("colour", "color") < 1.0,
          "a one-letter miss must cost less than a whole word")
    close(word_error_rate("colour", "color").error_rate, 1.0, 1e-9,
          "a misspelled word is still a wrong word")

    # Topic words are what decide whether the question survived.
    close(content_recall("what is the campus tv station",
                         "what does the campus tv station"), 1.0, 1e-9,
          "recall must ignore the lost function word")
    check(content_recall("who founded the university",
                         "what time is it") < 0.5,
          "recall must fall when the topic words are gone")

    check((stratum_of(0.0), stratum_of(0.2), stratum_of(0.5))
          == ("clean", "mild", "severe"),
          "strata must follow the declared error-rate boundaries")

    folded = tokens("Beyoncé's mother")
    check(folded == ["beyoncés", "mother"],
          f"tokenization must fold apostrophes, got {folded}")

    # Without a reference utterance there is nothing to measure, and an assumed
    # zero error rate would be reported as a measurement.
    check(evaluate_asr_fidelity(Record(item_id="i", stt_text="a", llm_text="b"))
          == {},
          "no reference text must yield no fidelity figures")
    measured = evaluate_asr_fidelity(Record(item_id="i", stt_text="the cat",
                                            llm_text="b", ori_text="the cat"))
    check(measured.get("stt_exact_match") is True
          and measured.get("stt_stratum") == "clean",
          f"an exactly recognized item must be clean, got {measured}")

    # The alignment must be symmetric in cost: reversing the arguments turns
    # deletions into insertions and leaves the edit distance unchanged.
    forward = align(["a", "b", "c"], ["a", "c"])
    backward = align(["a", "c"], ["a", "b", "c"])
    check(forward.errors == backward.errors == 1,
          "the edit distance must not depend on the argument order")


def test_latency_loading() -> None:
    import csv
    import json
    import tempfile
    from pathlib import Path as _Path

    from .latency import item_key, load_run_latency, summarize_stages

    rows = [
        # item, stage, duration, extra
        ("00001", "stt", 100, {}),
        ("00001", "llm_ttft", 200, {}),
        ("00001", "llm_eval", 300, {"eval_tokens": 10, "tokens_per_sec": 33.3}),
        ("00002", "stt", 400, {}),
        # Two measurements of one stage within one item: the median stands in for
        # the item so that a multi-utterance item does not weigh more.
        ("00002", "llm_ttft", 500, {}),
        ("00002", "llm_ttft", 700, {}),
        ("00003", "stt", 900, {}),
        ("00003", "llm_ttft", 1000, {}),
    ]

    with tempfile.TemporaryDirectory() as directory:
        run_dir = _Path(directory)
        with open(run_dir / "latency_log_20260101_000000.csv", "w",
                  encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item", "stage", "duration_ms", "cpu_percent",
                             "llm_rss_mb", "extra_json"])
            for item, stage, duration, extra in rows:
                writer.writerow([item, stage, duration, 10, 500,
                                 json.dumps(extra)])

        # Read without the warm-up convention first.
        plain = load_run_latency(run_dir, warmup_items=0)
        check(plain.available and len(plain.per_item) == 3,
              f"every item must be timed, got {len(plain.per_item)}")
        close(plain.per_item["00002"]["lat_llm_ttft"], 600.0, 1e-9,
              "repeated measurements of one stage must reduce to their median")
        close(plain.per_item["00001"]["llm_tokens_per_sec"], 33.3, 1e-9,
              "the engine's own throughput figure must be read from extra_json")
        close(plain.resource_medians.get("cpu_percent"), 10.0, 1e-9,
              "resource columns must be summarized at run level")

        # The warm-up convention is read from the run's own aggregate, so these
        # timings and the run's published summary describe the same items.
        (run_dir / "log_averages.json").write_text(
            json.dumps({"warmup_dropped_per_run": 1}), encoding="utf-8")
        trimmed = load_run_latency(run_dir)
        check(trimmed.warmup_items == ["00001"],
              f"the first item must be excluded as warm-up, got "
              f"{trimmed.warmup_items}")
        check("00001" not in trimmed.per_item and len(trimmed.per_item) == 2,
              "an excluded warm-up item must carry no timings")

        stages = {entry["stage"]: entry for entry in summarize_stages(trimmed)}
        close(stages["stt"]["median_ms"], 650.0, 1e-9,
              "the stage median must be over the retained items only")
        check(stages["llm_ttft"]["p95_ms"] >= stages["llm_ttft"]["median_ms"],
              "the tail must not fall below the median")

    # The transcript names a file, the log names a stem: both must reduce to the
    # same key or the join silently produces no timings at all.
    check(item_key("00004.wav") == item_key("00004") == "00004",
          "the recording key must be the file stem")
    check(load_run_latency(_Path("does-not-exist")).available is False,
          "a missing latency log must be reported as unavailable, not raised")


def test_intent_coverage() -> None:
    from .relevance import build_idf_index, evaluate_relevance

    # The recognizer turned "monasteries" into "monasteries" but lost the topic
    # of the question; coverage of what was heard and of what was meant then
    # differ, and that difference is the recognizer's contribution.
    record = Record(item_id="i", filename="00001.wav",
                    ori_text="why did monasteries have gardens",
                    stt_text="why did monster his have gardens",
                    llm_text="Monasteries had gardens to grow herbs.")
    idf = build_idf_index([record])
    metrics = evaluate_relevance(record, idf)

    check("intent_coverage" in metrics and "coverage_intent_gap" in metrics,
          f"a reference utterance must add intent coverage, got {sorted(metrics)}")
    check(metrics["intent_coverage"] > metrics["request_coverage"],
          f"answering the intended question must score above the misheard one: "
          f"{metrics['intent_coverage']} vs {metrics['request_coverage']}")
    close(metrics["coverage_intent_gap"],
          metrics["intent_coverage"] - metrics["request_coverage"], 1e-12,
          "the gap must be the difference of the two coverages")

    # Without a reference utterance the intent measures must be absent rather
    # than silently computed against the recognized text.
    plain = evaluate_relevance(Record(item_id="i", stt_text="why gardens",
                                      llm_text="For herbs."), idf)
    check("intent_coverage" not in plain,
          "no reference utterance must mean no intent coverage")


def test_metric_families_are_corrected_separately() -> None:
    from .comparison import (DESCRIPTIVE, HIGHER_IS_BETTER, Metric,
                             MetricComparison, _apply_multiplicity_correction)

    def row(key: str, family: str, p_value: float) -> MetricComparison:
        return MetricComparison(
            metric=Metric(key, key, HIGHER_IS_BETTER, family=family),
            n_pairs=120, baseline_mean=0.0, contrast_mean=0.1,
            mean_difference=0.1, ci_low=0.05, ci_high=0.15,
            improved=60, degraded=10, unchanged=50, effect_size=0.2,
            effect_label="small", p_value=p_value, test_method="stub",
            n_effective=70)

    # The same p-value in two families, to isolate what the family does: four
    # response metrics at p = 0.02 and one runtime metric at p = 0.02.
    rows = [row(f"a{index}", "response", 0.020) for index in range(1, 5)]
    rows.append(row("b1", "runtime", 0.020))
    _apply_multiplicity_correction(rows, alpha=0.05)
    by_key = {r.metric.key: r for r in rows}

    # Four tests in the response family: the smallest is scaled by four, which
    # takes it past the threshold, so none of them is claimed.
    close(by_key["a1"].p_adjusted, 0.08, 1e-12,
          "Holm must scale by the size of the metric's own family")
    check(not any(by_key[f"a{index}"].significant for index in range(1, 5)),
          "one of four related tests at p = 0.02 must not be claimed alone")

    # The runtime metric has the same p-value and is the only test in its family,
    # so it keeps its decision: it is not penalised for the number of quality
    # metrics that happened to be measured beside it, which is the whole point of
    # declaring families in advance.
    close(by_key["b1"].p_adjusted, 0.020, 1e-12,
          "a lone test in its family must not be scaled up")
    check(by_key["b1"].significant,
          "a family of one must keep its uncorrected decision")

    # Under one pooled correction the runtime metric would have been scaled by
    # five and lost, which is the outcome the per-family design exists to avoid.
    pooled = [row(f"a{index}", "response", 0.020) for index in range(1, 5)]
    pooled.append(row("b1", "response", 0.020))
    _apply_multiplicity_correction(pooled, alpha=0.05)
    check(not pooled[-1].significant,
          "pooling the families must be what suppresses the lone test")

    # A descriptive row states no direction and must stay out of the family.
    descriptive = [MetricComparison(
        metric=Metric("words", "words", DESCRIPTIVE, family="runtime"),
        n_pairs=10, baseline_mean=1.0, contrast_mean=2.0, mean_difference=1.0,
        ci_low=0.5, ci_high=1.5, improved=0, degraded=0, unchanged=10,
        effect_size=None, effect_label="n/a", p_value=0.001,
        test_method="stub", n_effective=10)]
    _apply_multiplicity_correction(descriptive, alpha=0.05)
    check(descriptive[0].p_adjusted is None
          and descriptive[0].verdict == "descriptive",
          "a descriptive metric must not receive a verdict")


def test_batch_grouping() -> None:
    from .batch import RunIdentity, build_groups, parse_model_tag

    parsed = parse_model_tag("llama3.1:8b-instruct-q4_K_M")
    check(parsed == {"family": "llama3.1", "size_label": "8b", "size_b": 8.0,
                     "tuning": "instruct", "quantization": "q4_K_M",
                     "precision_bits": 4},
          f"an Ollama tag must split into its parts, got {parsed}")
    check(parse_model_tag("qwen2.5:1.5b-instruct-q4_K_M")["size_b"] == 1.5,
          "a fractional parameter count must parse")
    check(parse_model_tag("llama3.1:8b-instruct-fp16")["precision_bits"] == 16,
          "an unquantized variant must rank above a quantized one")
    # An unrecognised tag must group on its own rather than merge with something.
    check(parse_model_tag("custom-model")["family"] == "custom-model",
          "an unparseable tag must keep its whole text as the family")

    def identity(order: int, tag: str, temperature: float,
                 seed: Optional[int], stt_engine: str = "vosk",
                 stt_model: str = "small-en-us-0.15",
                 stt_device: str = "cpu") -> RunIdentity:
        from pathlib import Path as _Path
        return RunIdentity(
            run_dir=_Path(f"runs/{order:02d}"), cell=f"{order:02d}-cell",
            timestamp="t", order=order, model_tag=tag,
            temperature=temperature, seed=seed, num_ctx=1024, max_tokens=150,
            prompt_file="p.txt", stt_engine=stt_engine, mode="cpu",
            stt_model=stt_model, stt_device=stt_device,
            **parse_model_tag(tag))

    runs = [
        identity(1, "llama3.2:1b-instruct-q4_K_M", 0.0, None),
        identity(2, "llama3.2:1b-instruct-q4_K_M", 0.7, 42),
        identity(3, "llama3.2:3b-instruct-q4_K_M", 0.0, None),
        identity(4, "llama3.2:3b-instruct-q4_K_M", 0.7, 42),
        identity(5, "llama3.1:8b-instruct-fp16", 0.0, None),
        identity(6, "llama3.1:8b-instruct-q8_0", 0.0, None),
    ]
    groups = {(g.kind, g.group_id): g for g in build_groups(runs)}

    parameters = [g for (kind, _), g in groups.items() if kind == "parameters"]
    check(len(parameters) == 2,
          f"only the models run at two settings can form a parameter contrast, "
          f"got {len(parameters)}")
    # Greedy decoding is the baseline: it is the reproducible setting, so the
    # difference from it is the effect of sampling.
    check(all(g.baseline.temperature == 0.0 for g in parameters),
          "the greedy setting must be the baseline of a parameter contrast")

    size = groups[("model_size", "llama_q4_K_M_t0")]
    check(size.baseline.size_b == 3.0 and size.contrasts[0].size_b == 1.0,
          "the larger model must be the baseline of a size contrast")
    check(size.varying == "parameter count",
          f"a ladder within one release must claim only size, got "
          f"{size.varying!r}")

    # The sizes on offer are split across releases -- llama3.2 ships 1b and 3b,
    # llama3.1 ships 8b -- so a size ladder that stopped at the family boundary
    # would leave the largest model out of the comparison entirely.
    ladder = build_groups(runs + [identity(7, "llama3.1:8b-instruct-q4_K_M",
                                           0.0, None)])
    spanning = {g.group_id: g for g in ladder if g.kind == "model_size"}
    check("llama_q4_K_M_t0" in spanning
          and spanning["llama_q4_K_M_t0"].baseline.size_b == 8.0,
          f"the 8b must join the 4-bit llama ladder as its baseline, got "
          f"{sorted(spanning)}")
    check("llama3.1" in spanning["llama_q4_K_M_t0"].varying
          and "llama3.2" in spanning["llama_q4_K_M_t0"].varying,
          f"a ladder crossing a release must name both releases, got "
          f"{spanning['llama_q4_K_M_t0'].varying!r}")

    quantization = groups[("quantization", "llama3.1_8b_t0")]
    check(quantization.baseline.precision_bits == 16,
          "the highest precision must be the baseline of a quantization contrast")

    cross = groups[("cross_model", "at_t0")]
    check(cross.baseline.model_tag == "llama3.1:8b-instruct-fp16",
          f"precision must break the tie between two 8b variants, got "
          f"{cross.baseline.model_tag}")
    check(len(cross.contrasts) == 3,
          f"every other variant at that setting must be a contrast, got "
          f"{len(cross.contrasts)}")

    # Parameter count outranks precision across models: a 32b at 4 bits is the
    # stronger reference than an 8b at 16, and ordering by bits first would put
    # the smaller model in the baseline.
    with_large = build_groups(runs + [identity(8, "qwen2.5:32b-instruct-q4_K_M",
                                               0.0, None)])
    largest = next(g for g in with_large
                   if g.kind == "cross_model" and g.group_id == "at_t0")
    check(largest.baseline.size_b == 32.0,
          f"the largest model must be the cross-model baseline, got "
          f"{largest.baseline.model_tag}")

    # A run with nothing to compare against must not produce an empty group.
    alone = build_groups([identity(1, "llama3.2:1b-instruct-q4_K_M", 0.0, None)])
    check(alone == [], f"a single run supports no contrast, got {alone}")


def test_recognizer_grouping() -> None:
    """A second recognizer must form its own contrast, not join the model ones."""
    from .batch import (RunIdentity, _recognizer_fields, build_groups,
                        parse_model_tag)

    fields = _recognizer_fields({"stt_engine": "vosk",
                                 "vosk_model": "./vosk/vosk-model-small-en-us-0.15"})
    check(fields == {"stt_model": "small-en-us-0.15", "stt_device": "cpu",
                     "stt_compute": ""},
          f"the Vosk acoustic model must be read from its path, got {fields}")
    fields = _recognizer_fields({"stt_engine": "whisper", "whisper_model": "small",
                                 "whisper_device": "cuda",
                                 "whisper_compute_type": "int8"})
    check(fields == {"stt_model": "small", "stt_device": "cuda",
                     "stt_compute": "int8"},
          f"the Whisper model, device and compute type must be read, got {fields}")

    def identity(order: int, tag: str, temperature: float,
                 stt_engine: str = "vosk", stt_model: str = "small-en-us-0.15",
                 stt_device: str = "cpu", stt_compute: str = "") -> RunIdentity:
        from pathlib import Path as _Path
        return RunIdentity(
            run_dir=_Path(f"runs/{order:02d}"), cell=f"{order:02d}-cell",
            timestamp="t", order=order, model_tag=tag, temperature=temperature,
            seed=None, num_ctx=1024, max_tokens=150, prompt_file="p.txt",
            stt_engine=stt_engine, mode="cpu", stt_model=stt_model,
            stt_device=stt_device, stt_compute=stt_compute,
            **parse_model_tag(tag))

    small = "llama3.2:1b-instruct-q4_K_M"
    large = "llama3.1:8b-instruct-q4_K_M"
    runs = [identity(1, small, 0.0), identity(2, large, 0.0),
            identity(3, small, 0.7),
            identity(4, small, 0.0, "whisper", "small", "cuda", "int8"),
            identity(5, large, 0.0, "whisper", "small", "cuda", "int8")]
    groups = build_groups(runs)

    # The words the recognizer produced are the model's input, so a model
    # contrast that mixed recognizers would report an input change as a model
    # effect. Every group except the recognizer contrast must hold it fixed.
    for group in groups:
        members = [group.baseline, *group.contrasts]
        recognizers = {run.recognizer for run in members}
        check(len(recognizers) == 1 or group.kind == "recognizer",
              f"{group.kind}/{group.group_id} mixes recognizers: "
              f"{sorted(recognizers)}")

    cross = [g for g in groups if g.kind == "cross_model"]
    check(len(cross) == 2,
          f"each recognizer must get its own cross-model group per setting, got "
          f"{[g.group_id for g in cross]}")
    check(all(g.baseline.recognizer in g.group_id for g in cross),
          f"a group id must name the recognizer it holds fixed, got "
          f"{[g.group_id for g in cross]}")

    recognizer_groups = {g.group_id: g for g in groups if g.kind == "recognizer"}
    check(len(recognizer_groups) == 2,
          f"every model measured behind both recognizers must form a contrast, "
          f"got {sorted(recognizer_groups)}")
    contrast = recognizer_groups[f"{small}_at_t0"]
    # The recognizer the rest of the grid was measured with is the baseline, so a
    # delta reads as what switching away from the established setup changes.
    check(contrast.baseline.stt_engine == "vosk"
          and contrast.contrasts[0].stt_engine == "whisper",
          f"the more frequently used recognizer must be the baseline, got "
          f"{contrast.baseline.recognizer}")
    check(contrast.baseline.temperature == contrast.contrasts[0].temperature
          and contrast.baseline.model_tag == contrast.contrasts[0].model_tag,
          "a recognizer contrast must hold the model and the setting fixed")

    # With one recognizer the group ids stay as they were, so a grid measured
    # behind a single recognizer is unaffected by this dimension.
    single = build_groups([run for run in runs if run.stt_engine == "vosk"])
    check(any(g.group_id == "at_t0" for g in single if g.kind == "cross_model"),
          f"a single-recognizer batch must keep its plain group ids, got "
          f"{[g.group_id for g in single if g.kind == 'cross_model']}")
    check(not any(g.kind == "recognizer" for g in single),
          "one recognizer supports no recognizer contrast")


def test_recognizer_pairing_and_checks() -> None:
    """Cross-recognizer contrasts must pair on the recording, not the transcript."""
    from types import SimpleNamespace

    from .aggregation import ItemResult
    from .batch import RunIdentity, ContrastGroup, parse_model_tag, \
        verify_shared_inputs
    from .comparison import _pair_items

    def items(texts: Dict[str, str]) -> List[ItemResult]:
        return [ItemResult(record=Record(item_id=name.split(".")[0],
                                         filename=name, ori_text="what is this",
                                         stt_text=text, llm_text="A reply."))
                for name, text in texts.items()]

    heard = items({"00001.wav": "what is this", "00002.wav": "how many are there"})
    misheard = items({"00001.wav": "what is his",
                      "00002.wav": "how many are their"})

    paired, base_only, contrast_only = _pair_items(heard, misheard)
    check(len(paired) == 2 and not base_only and not contrast_only,
          f"items must pair on the recording when the transcripts differ, got "
          f"{len(paired)} pair(s)")
    check(all(a.record.filename == b.record.filename for a, b in paired),
          "pairing on the recording must align the same file on both sides")

    # Without filenames the utterance is the only identity left, and an item the
    # other side does not have must be excluded rather than aligned with a
    # neighbour.
    bare = [ItemResult(record=Record(item_id="a", stt_text="one word",
                                     llm_text="A reply.")),
            ItemResult(record=Record(item_id="b", stt_text="another",
                                     llm_text="A reply."))]
    other = [ItemResult(record=Record(item_id="a", stt_text="one word",
                                      llm_text="A reply."))]
    paired, base_only, contrast_only = _pair_items(bare, other)
    check(len(paired) == 1 and base_only == 1 and contrast_only == 0,
          f"an unmatched item must be counted and excluded, got {len(paired)} "
          f"pair(s), {base_only} baseline-only")

    def run(order: int, engine: str) -> RunIdentity:
        from pathlib import Path as _Path
        tag = "llama3.2:1b-instruct-q4_K_M"
        return RunIdentity(
            run_dir=_Path(f"runs/{order}"), cell=f"{order:02d}-cell",
            timestamp="t", order=order, model_tag=tag, temperature=0.0,
            seed=None, num_ctx=1024, max_tokens=150, prompt_file="p.txt",
            stt_engine=engine, mode="cpu", stt_model="m", stt_device="cpu",
            **parse_model_tag(tag))

    def outcome(results: List[ItemResult]) -> SimpleNamespace:
        return SimpleNamespace(results=results, latency=None,
                               context=SimpleNamespace(system_prompt="p"),
                               spec=SimpleNamespace(spec_id="s"))

    vosk, whisper = run(1, "vosk"), run(2, "whisper")
    outcomes = {vosk.key: outcome(heard), whisper.key: outcome(misheard)}

    # Checked as one set, two recognizers are a breach worth reporting.
    pooled = verify_shared_inputs([vosk, whisper], outcomes)
    check(any("recognizer" in text for text in pooled),
          f"a pooled check must report the differing recognizer, got {pooled}")

    # Checked against the contrast that varies the recognizer, the same runs are
    # exactly what that contrast is for, so neither the recognizer nor the
    # transcripts may be reported as a breach.
    group = ContrastGroup(kind="recognizer", group_id="g", question="q",
                          varying="speech recognizer", baseline=vosk,
                          contrasts=[whisper])
    scoped = verify_shared_inputs([vosk, whisper], outcomes, [group])
    check(not scoped,
          f"a recognizer contrast must not warn about its own variable, got "
          f"{scoped}")

    # The intended question must still be shared: that is what makes the pairing
    # a comparison of like with like.
    diverged = items({"00001.wav": "what is this", "00002.wav": "how many"})
    diverged[0].record.ori_text = "a different question"
    outcomes[whisper.key] = outcome(diverged)
    scoped = verify_shared_inputs([vosk, whisper], outcomes, [group])
    check(any("intended question" in text for text in scoped),
          f"a changed reference utterance must be reported, got {scoped}")


def main() -> int:
    tests = [
        test_sentence_splitting,
        test_readability,
        test_constraints,
        test_audit_atoms,
        test_relevance,
        test_json_recovery,
        test_rubric_loads,
        test_cohen_kappa_and_ac1,
        test_krippendorff_alpha,
        test_icc,
        test_prediction_powered_inference,
        test_statistics,
        test_paired_statistics,
        test_selfcheck_scoring,
        test_pairwise_position_bias,
        test_transport_retry_policy,
        test_record_building,
        test_answer_key,
        test_word_error_rate,
        test_latency_loading,
        test_intent_coverage,
        test_metric_families_are_corrected_separately,
        test_batch_grouping,
        test_recognizer_grouping,
        test_recognizer_pairing_and_checks,
    ]

    for test in tests:
        try:
            test()
            print(f"  ran {test.__name__}")
        except Exception as exc:
            _FAILURES.append(f"{test.__name__} raised "
                             f"{exc.__class__.__name__}: {exc}")

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} of {_CHECKS} checks")
        for failure in _FAILURES:
            print(f"  - {failure}")
        return 1

    print(f"OK: {_CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
