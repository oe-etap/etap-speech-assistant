#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 0: verifiable constraint checking and speech-suitability measures.

Following the verifiable-instruction principle of IFEval (Zhou et al., 2023),
every check in this module is decided by a program rather than estimated by a
model or a rater. Nothing here needs calibration: if the prompt asks for an
opening sentence of at most five words, the number of words in the opening
sentence settles the question. That is what makes this tier the part of the
protocol with no measurement error to defend.

As in IFEval, each check is reported twice. The *strict* verdict tests the
response as generated. The *loose* verdict retests it after formatting-only
transformations (stripped markdown, dropped wrapper lines, removed "Answer:"
preambles), so that a response which obeys the instruction but wraps it in
boilerplate is not scored as a content failure. Both figures are reported,
because the gap between them is itself informative: it separates instruction
failures from formatting noise.

Two aggregate levels are reported, again mirroring IFEval:
  * check level - the share of individual constraints satisfied;
  * item level  - whether *all* hard constraints of an item are satisfied.
The item-level figure is the stricter and the one to headline, since a response
that violates any single hard constraint did not follow the prompt.

The readability measures (Flesch Reading Ease, Flesch-Kincaid Grade Level) are
validated instruments for written text. They are reported here as descriptive
speech-suitability indicators, not as a quality score: they were not developed
or normed for synthesized spoken output, so a reading-ease figure supports a
statement about lexical and syntactic complexity, not about intelligibility.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

from . import textutils as tu
from .loaders import Record, load_yaml


# Formatting-only rewrites used for the loose verdict. Applied in the order
# listed and also in combination, exactly as the strict/loose pair in IFEval is
# meant to work: the response passes loosely if any variant passes.
def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)


def _strip_preamble(text: str) -> str:
    return re.sub(r"^\s*(?:answer|response|assistant|sure|okay)\s*[:\-]\s*",
                  "", text, flags=re.IGNORECASE)


def _drop_first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[1:]).strip() if len(lines) > 1 else text


def _drop_last_line(text: str) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[:-1]).strip() if len(lines) > 1 else text


_LOOSE_TRANSFORMS: List[Callable[[str], str]] = [
    lambda t: t,
    _strip_markdown,
    _strip_preamble,
    _drop_first_line,
    _drop_last_line,
    lambda t: _strip_preamble(_strip_markdown(t)),
    lambda t: _drop_first_line(_strip_markdown(t)),
    lambda t: _drop_last_line(_drop_first_line(_strip_markdown(t))),
]


@dataclass
class CheckResult:
    """Outcome of one constraint on one response."""

    check_id: str
    check_type: str
    severity: str            # "hard" (prompt violation) or "soft" (advisory)
    passed_strict: bool
    passed_loose: bool
    observed: str            # What was measured, for auditability
    expected: str            # What the constraint required
    description: str = ""
    applicable: bool = True  # False when the item's category exempts the check


@dataclass
class ConstraintOutcome:
    """All Tier 0 results for one response, plus derived descriptive measures."""

    item_id: str
    results: List[CheckResult] = field(default_factory=list)
    measures: Dict[str, Any] = field(default_factory=dict)

    @property
    def applicable(self) -> List[CheckResult]:
        return [r for r in self.results if r.applicable]

    @property
    def hard(self) -> List[CheckResult]:
        return [r for r in self.applicable if r.severity == "hard"]

    @property
    def check_level_strict(self) -> Optional[float]:
        """Share of applicable hard constraints satisfied as generated."""
        checks = self.hard
        if not checks:
            return None
        return sum(1 for r in checks if r.passed_strict) / len(checks)

    @property
    def check_level_loose(self) -> Optional[float]:
        """Share of applicable hard constraints satisfied after formatting fixes."""
        checks = self.hard
        if not checks:
            return None
        return sum(1 for r in checks if r.passed_loose) / len(checks)

    @property
    def item_level_strict(self) -> bool:
        """True only when every applicable hard constraint is satisfied."""
        return all(r.passed_strict for r in self.hard)

    @property
    def item_level_loose(self) -> bool:
        return all(r.passed_loose for r in self.hard)

    def failures(self, loose: bool = False) -> List[str]:
        """Ids of the hard constraints this response violated."""
        return [r.check_id for r in self.hard
                if not (r.passed_loose if loose else r.passed_strict)]


# --------------------------------------------------------------------------
# Individual check implementations.
#
# Each returns (passed, observed_text). They receive the candidate response,
# the record under evaluation, and the check's YAML parameters.
# --------------------------------------------------------------------------

def _check_non_empty(text: str, record: Record, params: Dict[str, Any]):
    return bool(text.strip()), f"{tu.word_count(text)} words"


def _check_opening_max_words(text: str, record: Record, params: Dict[str, Any]):
    limit = int(params.get("max_words", 5))
    sentences = tu.split_sentences(text)
    if not sentences:
        return False, "no sentence found"
    count = tu.word_count(sentences[0])
    return count <= limit, f"{count} words: {_excerpt(sentences[0])}"


def _check_opening_terminator(text: str, record: Record, params: Dict[str, Any]):
    allowed = str(params.get("terminators", "."))
    sentences = tu.split_sentences(text)
    if not sentences:
        return False, "no sentence found"
    stripped = sentences[0].rstrip("\"')]}…")
    last = stripped[-1] if stripped else ""
    return last in allowed, f"ends with {last!r}"


def _check_detail_sentences(text: str, record: Record, params: Dict[str, Any]):
    low = int(params.get("min", 1))
    high = int(params.get("max", 2))
    n_detail = max(0, len(tu.split_sentences(text)) - 1)
    return low <= n_detail <= high, f"{n_detail} detail sentences"


def _check_total_sentences(text: str, record: Record, params: Dict[str, Any]):
    low = int(params.get("min", 1))
    high = int(params.get("max", 4))
    n = len(tu.split_sentences(text))
    return low <= n <= high, f"{n} sentences"


def _check_language(text: str, record: Record, params: Dict[str, Any]):
    threshold = float(params.get("min_marker_ratio", 0.08))
    ratio = tu.english_marker_ratio(text)
    return ratio >= threshold, f"english marker ratio {ratio:.3f}"


def _check_forbidden_patterns(text: str, record: Record, params: Dict[str, Any]):
    patterns = params.get("patterns") or []
    hits = [p for p in patterns
            if re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)]
    return not hits, ("none" if not hits else "matched: " + ", ".join(hits))


def _check_no_question(text: str, record: Record, params: Dict[str, Any]):
    # Quoted or parenthesised question marks are counted too: the prompt forbids
    # asking the user anything, and a question inside quotes is still spoken.
    count = text.count("?")
    return count == 0, f"{count} question marks"


def _check_response_complete(text: str, record: Record, params: Dict[str, Any]):
    allowed = str(params.get("terminators", ".!?"))
    stripped = text.strip().rstrip("\"')]}…")
    if not stripped:
        return False, "empty"
    return stripped[-1] in allowed, f"ends with {stripped[-1]!r}"


def _check_not_truncated_at_cap(text: str, record: Record, params: Dict[str, Any]):
    """Flag a response that both ran out of budget and stopped mid-sentence.

    Either signal alone is weak: a long answer may legitimately end on the cap
    with a complete sentence, and a short answer may end oddly for other
    reasons. Together they identify the failure that actually reaches the user,
    where the synthesized speech cuts off mid-utterance.
    """
    cap = params.get("max_tokens")
    if not cap:
        return True, "no token cap recorded"
    ratio_threshold = float(params.get("cap_ratio", 0.9))
    estimated = tu.estimate_tokens(text)
    near_cap = estimated >= ratio_threshold * float(cap)
    stripped = text.strip().rstrip("\"')]}…")
    incomplete = bool(stripped) and stripped[-1] not in ".!?"
    truncated = near_cap and incomplete
    return (not truncated,
            f"~{estimated} est. tokens vs cap {cap}, "
            f"{'incomplete' if incomplete else 'complete'} ending")


def _check_max_words(text: str, record: Record, params: Dict[str, Any]):
    limit = int(params.get("max_words", 80))
    count = tu.word_count(text)
    return count <= limit, f"{count} words"


def _check_max_spoken_seconds(text: str, record: Record, params: Dict[str, Any]):
    limit = float(params.get("max_seconds", 25.0))
    rate = float(params.get("words_per_second", 2.5))
    seconds = tu.spoken_seconds(text, rate)
    return seconds <= limit, f"~{seconds:.1f} s at {rate} words/s"


def _check_must_include(text: str, record: Record, params: Dict[str, Any]):
    """Scenario-supplied content requirement, e.g. an expected escalation phrase."""
    patterns = record.must_include
    if not patterns:
        return True, "no requirement specified"
    missing = [p for p in patterns if not re.search(p, text, re.IGNORECASE)]
    return not missing, ("all present" if not missing
                         else "missing: " + ", ".join(missing))


def _check_must_forbid(text: str, record: Record, params: Dict[str, Any]):
    """Scenario-supplied prohibition, e.g. a diagnostic label or a dosage."""
    patterns = record.must_forbid
    if not patterns:
        return True, "no prohibition specified"
    hits = [p for p in patterns if re.search(p, text, re.IGNORECASE)]
    return not hits, ("none" if not hits else "matched: " + ", ".join(hits))


# Parameters supplied at evaluation time rather than by the spec, mapped to the
# check types that consume them. Merging a runtime value into every check would
# make it appear in the audit trail of checks that never read it.
_RUNTIME_PARAM_CONSUMERS: Dict[str, set] = {
    "max_tokens": {"not_truncated_at_cap"},
}

# Checks that measure a scenario-supplied expectation. With nothing specified for
# an item there is no expectation to satisfy, so they are marked not applicable
# rather than passing: a pass rate that counts inert checks overstates how much
# was actually verified.
_CONDITIONAL_CHECK_TYPES = {
    "must_include": lambda record: bool(record.must_include),
    "must_forbid": lambda record: bool(record.must_forbid),
}


CHECK_TYPES: Dict[str, Callable] = {
    "non_empty": _check_non_empty,
    "opening_sentence_max_words": _check_opening_max_words,
    "opening_sentence_terminator": _check_opening_terminator,
    "detail_sentence_count": _check_detail_sentences,
    "total_sentence_count": _check_total_sentences,
    "language_marker_ratio": _check_language,
    "forbidden_patterns": _check_forbidden_patterns,
    "no_question": _check_no_question,
    "response_complete": _check_response_complete,
    "not_truncated_at_cap": _check_not_truncated_at_cap,
    "max_total_words": _check_max_words,
    "max_spoken_seconds": _check_max_spoken_seconds,
    "must_include": _check_must_include,
    "must_forbid": _check_must_forbid,
}


@dataclass
class ConstraintSpec:
    """A named set of verifiable constraints, loaded from YAML."""

    spec_id: str
    description: str
    checks: List[Dict[str, Any]] = field(default_factory=list)
    source_path: Optional[Path] = None

    @classmethod
    def load(cls, path: Path) -> "ConstraintSpec":
        data = load_yaml(Path(path))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        checks = data.get("checks") or []
        if not isinstance(checks, list):
            raise ValueError(f"{path}: 'checks' must be a list")

        unknown = sorted({c.get("type") for c in checks
                          if c.get("type") not in CHECK_TYPES})
        if unknown:
            raise ValueError(
                f"{path}: unknown check type(s): {', '.join(map(str, unknown))}. "
                f"Known types: {', '.join(sorted(CHECK_TYPES))}")

        return cls(spec_id=str(data.get("id") or Path(path).stem),
                   description=str(data.get("description") or ""),
                   checks=checks,
                   source_path=Path(path))


def evaluate_constraints(record: Record,
                         spec: ConstraintSpec,
                         runtime_params: Optional[Dict[str, Any]] = None
                         ) -> ConstraintOutcome:
    """Run every applicable check in `spec` against one record.

    Args:
        record: The (utterance, response) pair under evaluation.
        spec: The constraint set, normally derived from the system prompt.
        runtime_params: Values known only at evaluation time and merged into
            every check's parameters, notably `max_tokens` recovered from the
            run's `config_used.yaml`.

    Returns:
        A ConstraintOutcome holding one CheckResult per check plus the
        descriptive speech-suitability measures for the response.
    """
    runtime_params = runtime_params or {}
    text = record.llm_text
    outcome = ConstraintOutcome(item_id=record.item_id)

    for entry in spec.checks:
        check_type = entry["type"]
        check_id = str(entry.get("id") or check_type)
        severity = str(entry.get("severity", "hard"))
        spec_params = dict(entry.get("params") or {})
        params = dict(spec_params)
        for name, value in runtime_params.items():
            if (check_type in _RUNTIME_PARAM_CONSUMERS.get(name, set())
                    and name not in params):
                params[name] = value

        func = CHECK_TYPES[check_type]
        reason = _inapplicable_reason(entry, record, check_type)

        if reason:
            outcome.results.append(CheckResult(
                check_id=check_id, check_type=check_type, severity=severity,
                passed_strict=True, passed_loose=True,
                observed=reason, expected="-",
                description=str(entry.get("description") or ""),
                applicable=False))
            continue

        strict_pass, observed = func(text, record, params)
        loose_pass = strict_pass
        if not strict_pass:
            loose_pass = any(func(transform(text), record, params)[0]
                             for transform in _LOOSE_TRANSFORMS)

        outcome.results.append(CheckResult(
            check_id=check_id, check_type=check_type, severity=severity,
            passed_strict=bool(strict_pass), passed_loose=bool(loose_pass),
            observed=observed,
            expected=_expected_text(entry, params, record, check_type),
            description=str(entry.get("description") or "")))

    outcome.measures = describe_response(text)
    return outcome


def _inapplicable_reason(entry: Dict[str, Any], record: Record,
                         check_type: str) -> Optional[str]:
    """Return why a check does not apply to this record, or None if it does.

    A check may be limited with `applies_to` or exempted with `skip_for`, which
    is what keeps, for example, a "no questions" constraint from penalising a
    crisis item whose expected behaviour is to ask a clarifying question. A
    conditional check is inapplicable when the scenario spec gave it nothing to
    verify.

    `min_response_words` covers a third case: a measure that needs a minimum
    amount of text before it can decide anything. A proportion over tokens is
    undecidable on a two-word reply, and scoring it as a violation there counts
    the measure's own blind spot as the model's failure.
    """
    skip_for = entry.get("skip_for") or []
    if record.category in skip_for:
        return f"exempt for category {record.category}"

    min_words = entry.get("min_response_words")
    if min_words is not None:
        observed_words = tu.word_count(record.llm_text)
        if observed_words < int(min_words):
            return (f"response of {observed_words} words is below the "
                    f"{min_words} this check needs to decide")

    applies_to = entry.get("applies_to")
    if applies_to not in (None, "all", ["all"]):
        targets = ([applies_to] if isinstance(applies_to, str)
                   else list(applies_to))
        if record.category not in targets:
            return f"scoped to {', '.join(map(str, targets))}"

    predicate = _CONDITIONAL_CHECK_TYPES.get(check_type)
    if predicate is not None and not predicate(record):
        return "nothing specified for this item in the scenario spec"
    return None


def _expected_text(entry: Dict[str, Any], params: Dict[str, Any],
                   record: Record, check_type: str) -> str:
    """Render the requirement of a check for the audit trail."""
    if entry.get("expected"):
        return str(entry["expected"])

    # The scenario-driven checks take their requirement from the item rather than
    # from the spec entry, so reading params alone would describe them as having
    # required nothing.
    if check_type == "must_include" and record.must_include:
        return "all present: " + ", ".join(record.must_include)
    if check_type == "must_forbid" and record.must_forbid:
        return "none present: " + ", ".join(record.must_forbid)

    relevant = {k: v for k, v in params.items()
                if k in {"max_words", "min", "max", "terminators", "patterns",
                         "min_marker_ratio", "max_seconds", "max_tokens",
                         "words_per_second", "cap_ratio"}}
    return ", ".join(f"{k}={v}" for k, v in sorted(relevant.items())) or "-"


def describe_response(text: str) -> Dict[str, Any]:
    """Descriptive measures of one response, independent of any constraint set.

    Reported alongside the pass/fail verdicts because a response can satisfy
    every constraint and still be unsuitable for speech, and because these are
    the figures that make a length or complexity effect visible across
    configurations.
    """
    sentences = tu.split_sentences(text)
    n_words = tu.word_count(text)
    return {
        "word_count": n_words,
        "sentence_count": len(sentences),
        "char_count": len(text),
        "mean_words_per_sentence": (n_words / len(sentences)) if sentences else 0.0,
        "opening_words": tu.word_count(sentences[0]) if sentences else 0,
        "estimated_tokens": tu.estimate_tokens(text),
        "estimated_spoken_seconds": round(tu.spoken_seconds(text), 2),
        "flesch_reading_ease": round(tu.flesch_reading_ease(text), 2),
        "flesch_kincaid_grade": round(tu.flesch_kincaid_grade(text), 2),
    }


def _excerpt(text: str, limit: int = 60) -> str:
    """Shorten text for a single-line audit field."""
    flat = " ".join(tu.normalize_unicode(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3] + "..."


REFERENCE_KEYS = ["ifeval", "followbench", "flesch1948", "kincaid1975"]
