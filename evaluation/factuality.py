#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 2: factuality and hallucination measures.

Terminology follows the taxonomy of Ji et al. (2023): an *intrinsic*
hallucination contradicts the input, an *extrinsic* one asserts something the
input does not support. The two need different evidence, so three complementary
procedures are implemented.

1. Deterministic audit atoms. Numerals, years and capitalised name spans are
   extracted from each response. This is not a metric and makes no correctness
   claim; it is the worksheet that tells a human reviewer exactly which spans
   carry a checkable factual commitment. It costs nothing, runs on every item,
   and is often the fastest route to the specific error in a response that all
   graded scores merely mark as "somewhat wrong".

2. Self-consistency (SelfCheckGPT, Manakul et al., 2023). The same prompt is
   resampled several times at non-zero temperature, and each sentence of the
   evaluated response is scored by how well it is supported by the resamples. A
   fabricated statement tends not to survive resampling, whereas knowledge the
   model actually holds recurs. No external knowledge source is needed, which is
   what makes it usable inside a closed local deployment.

   Deviation from the published method: the original uses BERTScore, an NLI model
   or a question-answering model as the support kernel. Requiring any of those
   would add a deep-learning dependency to a toolkit that must run under the CPU
   preset, so the default kernel is token-level F1 overlap, upgraded to embedding
   cosine when a sentence-embedding model is available. This preserves the
   method's structure but not its published detection performance, and results
   are reported as `selfcheck_*` with the kernel named alongside.

   Note on the run configuration: the pipeline's own runs use temperature 0. The
   resamples for this tier are generated separately at non-zero temperature and
   do not alter the evaluated response, which stays exactly as the run produced
   it.

3. Atomic factual precision (FActScore, Min et al., 2023). A judge model
   decomposes the response into atomic claims and labels each one against a
   knowledge source. Reporting the share of supported claims rather than a
   verdict on the whole response is the point of the method: a response that is
   largely right with one wrong date is a different failure from one that is
   wholly fabricated, and a single score cannot say which occurred.

   The knowledge source is the scenario's reference answer when the spec
   provides one. Without a reference the judge falls back on its own parametric
   knowledge, which is a materially weaker design; that case is flagged in the
   output as `knowledge_source: judge_parametric` and must not be reported as a
   FActScore result without that qualifier.
"""

from dataclasses import dataclass, field
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence

from . import textutils as tu
from .loaders import Record
from .ollama_client import OllamaClient, extract_json_object
from .relevance import EmbeddingBackend, token_f1


# Words that start a sentence and are capitalised for that reason alone. Excluded
# from name-span extraction so that every sentence does not yield a false atom.
_SENTENCE_STARTERS = {
    "The", "This", "That", "These", "Those", "It", "There", "They", "He", "She",
    "We", "You", "I", "A", "An", "As", "At", "But", "And", "Or", "If", "When",
    "While", "Although", "However", "Both", "Its", "Their", "Machine", "Start",
    "Absolutely", "Answer", "Yes", "No", "Sure", "Okay",
}

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\s*(?:%|percent|kg|mg|ml|km|m|cm|"
                        r"hours?|minutes?|days?|weeks?|months?|years?)?\b",
                        re.IGNORECASE)
_NAME_SPAN_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+(?:of|de|van|der|the)\s+)?)"
                           r"(?:\s+[A-Z][a-z]+)+\b")


@dataclass
class AuditAtoms:
    """Deterministically extracted spans that carry a checkable commitment."""

    years: List[str] = field(default_factory=list)
    quantities: List[str] = field(default_factory=list)
    name_spans: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.years) + len(self.quantities) + len(self.name_spans)

    def as_dict(self) -> Dict[str, Any]:
        return {"years": self.years, "quantities": self.quantities,
                "name_spans": self.name_spans, "atom_count": self.count}


def extract_audit_atoms(text: str) -> AuditAtoms:
    """Extract years, quantities and multiword name spans from a response.

    Deliberately over-inclusive: the purpose is that a reviewer never has to
    reread a response looking for the factual commitments, so a spurious span
    costs a glance while a missed one costs an undetected error.
    """
    normalized = tu.normalize_unicode(text)
    years = sorted(set(_YEAR_RE.findall(normalized)))

    quantities = []
    for match in _NUMBER_RE.finditer(normalized):
        span = match.group().strip()
        if span and span not in years and not _YEAR_RE.fullmatch(span):
            quantities.append(span)

    names = []
    for match in _NAME_SPAN_RE.finditer(normalized):
        span = match.group().strip()
        head = span.split()[0]
        if head in _SENTENCE_STARTERS and len(span.split()) < 3:
            continue
        names.append(span)

    return AuditAtoms(years=years,
                      quantities=sorted(set(quantities)),
                      name_spans=sorted(set(names)))


# --------------------------------------------------------------------------
# Self-consistency (SelfCheckGPT)
# --------------------------------------------------------------------------

# Mirrors the request framing in llm_engine.OllamaEngine._build_prompt. The
# resamples must be drawn from the same distribution as the evaluated response,
# so if that framing changes, this must change with it.
_PIPELINE_PROMPT = '{system_prompt}\n\nThe user said: "{user_text}"\n\nAnswer:'


@dataclass
class SelfCheckResult:
    """Sentence-level self-consistency scores for one response."""

    item_id: str
    kernel: str
    n_samples: int
    sentence_scores: List[float] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def mean_inconsistency(self) -> Optional[float]:
        """Mean inconsistency over sentences; higher means less supported."""
        return (statistics.fmean(self.sentence_scores)
                if self.sentence_scores else None)

    @property
    def max_inconsistency(self) -> Optional[float]:
        """Worst sentence in the response.

        Reported next to the mean because a single fabricated sentence inside an
        otherwise accurate answer is the failure mode that matters, and averaging
        over a long response hides exactly that.
        """
        return max(self.sentence_scores) if self.sentence_scores else None

    def flagged_sentences(self, threshold: float = 0.6) -> List[str]:
        """Sentences whose inconsistency exceeds `threshold`.

        The threshold is a screening convention, not a validated cut-off. It has
        to be calibrated against human labels on this dataset before any
        prevalence figure derived from it is reported.
        """
        return [sentence for sentence, score
                in zip(self.sentences, self.sentence_scores)
                if score >= threshold]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selfcheck_kernel": self.kernel,
            "selfcheck_samples": self.n_samples,
            "selfcheck_mean_inconsistency": _round(self.mean_inconsistency),
            "selfcheck_max_inconsistency": _round(self.max_inconsistency),
            "selfcheck_flagged_sentences": self.flagged_sentences(),
            "selfcheck_error": self.error,
        }


def run_selfcheck(record: Record, client: OllamaClient, system_prompt: str,
                  n_samples: int = 5, temperature: float = 0.8,
                  embeddings: Optional[EmbeddingBackend] = None
                  ) -> SelfCheckResult:
    """Score each sentence of a response by its support among stochastic resamples.

    Args:
        record: The item whose response is being checked.
        client: Ollama client configured with the *same model* that produced the
            response; resampling a different model measures cross-model
            disagreement instead of self-consistency.
        system_prompt: The system prompt the run used, so the resamples share the
            evaluated response's prompt framing.
        n_samples: Number of resamples. The original work uses substantially more;
            fewer samples widen the sampling error of each sentence score.
        temperature: Sampling temperature for the resamples.
        embeddings: Optional embedding backend, used as the support kernel when
            available.

    Returns:
        A SelfCheckResult with one inconsistency score per response sentence,
        scaled so that 0 means fully supported and 1 means unsupported.
    """
    kernel = ("embedding_cosine"
              if embeddings is not None and embeddings.available
              else "token_f1_surrogate")
    sentences = tu.split_sentences(record.llm_text)
    result = SelfCheckResult(item_id=record.item_id, kernel=kernel,
                             n_samples=n_samples, sentences=sentences)

    if not sentences:
        result.error = "empty response"
        return result

    prompt = _PIPELINE_PROMPT.format(system_prompt=system_prompt.strip(),
                                     user_text=record.stt_text)
    samples: List[List[str]] = []
    for index in range(n_samples):
        generation = client.generate(prompt, temperature=temperature,
                                     seed=1000 + index)
        if not generation.ok:
            result.error = generation.error
            continue
        sample_sentences = tu.split_sentences(generation.text)
        if sample_sentences:
            samples.append(sample_sentences)

    if not samples:
        result.error = result.error or "no usable resamples"
        return result

    result.n_samples = len(samples)
    for sentence in sentences:
        supports = [_max_support(sentence, sample, embeddings)
                    for sample in samples]
        result.sentence_scores.append(1.0 - statistics.fmean(supports))
    return result


def _max_support(sentence: str, sample_sentences: Sequence[str],
                 embeddings: Optional[EmbeddingBackend]) -> float:
    """Best support for `sentence` among the sentences of one resample."""
    if embeddings is not None and embeddings.available:
        values = [embeddings.similarity(sentence, candidate)
                  for candidate in sample_sentences]
        values = [v for v in values if v == v]        # drop NaN
        return max(0.0, min(1.0, max(values))) if values else 0.0

    return max((token_f1(sentence, candidate)["f1"]
                for candidate in sample_sentences), default=0.0)


# --------------------------------------------------------------------------
# Atomic factual precision (FActScore)
# --------------------------------------------------------------------------

_DECOMPOSE_TEMPLATE = """\
Break the text below into atomic factual claims. An atomic claim states exactly
one fact and is understandable on its own, with pronouns resolved. Ignore
opinions, offers of help and questions. Do not add facts that are not stated.

### Text
{response_text}

### Required output
Return one JSON object and nothing else:
{{"claims": ["<claim 1>", "<claim 2>"]}}
Return at most {max_claims} claims.
"""

_VERIFY_TEMPLATE = """\
Decide whether the claim below is supported.

{source_block}

### Claim
{claim}

### Labels
supported: the claim is correct according to the knowledge above.
unsupported: the claim contradicts the knowledge above, or is factually wrong.
unverifiable: the claim cannot be judged from the knowledge above.

### Required output
Return one JSON object and nothing else:
{{"label": "supported" | "unsupported" | "unverifiable", "reason": "<at most 20 words>"}}
"""

_LABELS = ("supported", "unsupported", "unverifiable")


@dataclass
class FactPrecisionResult:
    """Atomic-claim labels for one response."""

    item_id: str
    knowledge_source: str
    claims: List[Dict[str, str]] = field(default_factory=list)
    error: str = ""

    def count(self, label: str) -> int:
        return sum(1 for claim in self.claims if claim.get("label") == label)

    @property
    def precision(self) -> Optional[float]:
        """Supported share of the claims that could be judged at all.

        Unverifiable claims are excluded from the denominator rather than counted
        as errors, since being unable to check a claim is a property of the
        knowledge source, not of the response.
        """
        judged = self.count("supported") + self.count("unsupported")
        return (self.count("supported") / judged) if judged else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "factprecision_knowledge_source": self.knowledge_source,
            "factprecision_n_claims": len(self.claims),
            "factprecision_supported": self.count("supported"),
            "factprecision_unsupported": self.count("unsupported"),
            "factprecision_unverifiable": self.count("unverifiable"),
            "factprecision_precision": _round(self.precision),
            "factprecision_unsupported_claims": [
                c["claim"] for c in self.claims
                if c.get("label") == "unsupported"],
            "factprecision_error": self.error,
        }


def run_fact_precision(record: Record, client: OllamaClient,
                       max_claims: int = 10) -> FactPrecisionResult:
    """Decompose a response into atomic claims and label each one.

    Args:
        record: The item under evaluation; its `reference_answers`, when present,
            become the knowledge source.
        client: Judge model used for both decomposition and verification.
        max_claims: Cap on claims per response, which bounds the number of
            verification calls.

    Returns:
        A FactPrecisionResult holding every claim with its label and the judge's
        stated reason, so that each label remains auditable.
    """
    has_reference = bool(record.reference_answers)
    result = FactPrecisionResult(
        item_id=record.item_id,
        knowledge_source="scenario_reference" if has_reference
                         else "judge_parametric")

    if not record.llm_text.strip():
        result.error = "empty response"
        return result

    decomposition = client.generate(
        _DECOMPOSE_TEMPLATE.format(response_text=record.llm_text,
                                   max_claims=max_claims),
        temperature=0.0, seed=0, num_predict=400)
    if not decomposition.ok:
        result.error = f"decomposition failed: {decomposition.error}"
        return result

    claims = _parse_claims(decomposition.text, max_claims)
    if not claims:
        result.error = "no claims recovered from decomposition"
        return result

    if has_reference:
        source_block = ("### Knowledge\n"
                        + "\n".join(f"- {ref}" for ref in record.reference_answers))
    else:
        source_block = ("### Knowledge\nNo reference text is available. Use your "
                        "own general knowledge, and answer 'unverifiable' when "
                        "you are not confident.")

    for claim in claims:
        verification = client.generate(
            _VERIFY_TEMPLATE.format(source_block=source_block, claim=claim),
            temperature=0.0, seed=0, num_predict=120)
        label, reason = _parse_label(verification.text if verification.ok else "")
        result.claims.append({"claim": claim, "label": label, "reason": reason})

    return result


def _parse_claims(text: str, max_claims: int) -> List[str]:
    """Recover the claim list from a decomposition response."""
    parsed = extract_json_object(text)
    claims: List[str] = []
    if parsed and isinstance(parsed.get("claims"), list):
        claims = [" ".join(str(c).split()) for c in parsed["claims"]]
    else:
        # Numbered or bulleted lists are the usual fallback shape.
        for line in text.splitlines():
            stripped = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line).strip()
            if len(stripped) > 12 and not stripped.startswith("{"):
                claims.append(stripped)

    seen = set()
    unique = []
    for claim in claims:
        key = claim.lower()
        if claim and key not in seen:
            seen.add(key)
            unique.append(claim)
    return unique[:max_claims]


def _parse_label(text: str) -> tuple:
    """Recover a verification label, defaulting to unverifiable."""
    parsed = extract_json_object(text)
    if parsed:
        label = str(parsed.get("label", "")).strip().lower()
        reason = " ".join(str(parsed.get("reason") or "").split())[:200]
        if label in _LABELS:
            return label, reason

    lowered = text.lower()
    for label in ("unsupported", "unverifiable", "supported"):
        if label in lowered:
            return label, " ".join(text.split())[:200]
    return "unverifiable", "label not recovered"


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(value, digits)


REFERENCE_KEYS = ["selfcheckgpt", "factscore", "hallucination_survey"]
