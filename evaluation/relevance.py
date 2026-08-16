#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 1: relevance and reference-based similarity.

Three families of measure are computed, and they are not interchangeable:

  * Reference-based overlap (SQuAD token-level F1; ROUGE-1 and ROUGE-L) and answer
    agreement (exact match; answer presence). These require a reference answer and
    are meaningful only for items where one correct answer exists. Liu et al. (2016)
    showed that word-overlap metrics correlate weakly with human judgement on
    open-ended dialogue, so they are reported as supporting evidence for closed-form
    items and never as the quality score for conversational ones.

    On a question with a short annotated answer span, answer presence is the
    informative member of this family: the assistant answers in a sentence, so exact
    match is near zero by construction and token F1 is depressed by every word of
    the sentence that is not part of the span, while presence asks the question that
    matters, whether the answer is in there.

  * Lexical grounding in the request (IDF-weighted content-word coverage). This
    asks whether the response actually addresses the words the user said. It
    detects the blunt failure mode - answering a different question - but a high
    value does not establish relevance, since restating the question scores well.
    It is therefore reported as a screening signal, not a relevance score.

  * Embedding similarity, if a sentence-embedding model is installed. This is
    the closest available stand-in for BERTScore (Zhang et al., 2020) without
    adding a required dependency, and it is skipped rather than approximated
    when the model is absent.

The honest division of labour is: this tier finds responses that are obviously
off-target cheaply and over the whole set; graded relevance belongs to the
rubric tiers.
"""

from collections import Counter
import math
import re
from typing import Dict, List, Optional, Sequence

from . import textutils as tu
from .loaders import Record

# Words carrying no topical content. Removed before overlap scoring so that
# matching "the" and "of" cannot inflate a relevance figure.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "as",
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "about",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "can", "could", "will", "would", "shall",
    "should", "may", "might", "must", "i", "me", "my", "you", "your", "he",
    "she", "it", "its", "we", "they", "them", "their", "this", "that", "these",
    "those", "there", "here", "what", "which", "who", "whom", "how", "when",
    "where", "why", "not", "no", "yes", "please", "tell", "s", "t",
    # Conversational openers and closers. They are frequent in spoken input and
    # carry no topic, so leaving them in would let a greeting count as coverage
    # of the request.
    "hello", "hi", "hey", "thanks", "thank", "okay", "ok", "well", "just",
    "like", "want", "need", "know", "think", "say", "said", "get", "give",
    "make", "good", "also", "some", "any", "much", "many", "more", "most",
}

_SUFFIXES = ("ing", "edly", "ies", "ied", "es", "ed", "ly", "s")

# Possessive and contraction endings. Removed before matching because a
# possessive is the same topical word as its base form: without this, the
# "Australia's" in a response does not match the "australia" in the request.
_POSSESSIVE_RE = re.compile(r"'s?$")


def _normalize(text: str) -> List[str]:
    """Lowercase, drop punctuation and articles, and tokenize.

    This is the SQuAD answer-normalization convention (Rajpurkar et al., 2016),
    which strips punctuation entirely. Keeping it identical matters because a
    token-F1 computed under a different normalization is not comparable with
    published figures.
    """
    tokens = [w.lower().replace("'", "") for w in tu.words(text)]
    return [t for t in tokens if t and t not in {"a", "an", "the"}]


def _content_tokens(text: str) -> List[str]:
    """Return topical tokens, with possessives folded and suffixes stripped."""
    out = []
    for raw in (w.lower() for w in tu.words(text)):
        token = _POSSESSIVE_RE.sub("", raw)
        if not token or token in _STOPWORDS or len(token) < 2:
            continue
        out.append(_stem(token))
    return out


def _stem(token: str) -> str:
    """Strip a common inflectional suffix so "explore" matches "exploration".

    A deliberately crude stemmer. It exists so that coverage is not understated
    by inflection alone; it is not accurate enough to build a score on, which is
    why coverage is reported as a screening signal.
    """
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def token_f1(prediction: str, reference: str) -> Dict[str, float]:
    """SQuAD-style token-level precision, recall and F1 against one reference."""
    pred_tokens = _normalize(prediction)
    ref_tokens = _normalize(reference)
    if not pred_tokens or not ref_tokens:
        zero = 1.0 if pred_tokens == ref_tokens else 0.0
        return {"precision": zero, "recall": zero, "f1": zero}

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)}


def exact_match(prediction: str, reference: str) -> float:
    """SQuAD exact match: the normalized response is the reference answer.

    Reported for comparability with the reading-comprehension literature, where a
    system emits the answer span alone. A conversational assistant answers in a
    sentence, so this is expected to be near zero and is not the measure of whether
    the answer was right; `answer_presence` is.
    """
    return float(_normalize(prediction) == _normalize(reference))


def answer_presence(prediction: str, reference: str) -> float:
    """Whether the reference answer occurs, in order and uninterrupted, in the response.

    The answer-presence convention of open-domain question answering (Chen et al.,
    2017; Lee et al., 2019): a response is credited when it contains the gold span
    under SQuAD normalization, which is what a correct answer wrapped in a sentence
    looks like. It is a lenient criterion -- a response that also contradicts itself
    elsewhere still counts -- so it bounds correctness from above and is reported
    together with token F1 rather than instead of it.
    """
    pred = _normalize(prediction)
    ref = _normalize(reference)
    if not ref or len(ref) > len(pred):
        return 0.0
    window = len(ref)
    return float(any(pred[i:i + window] == ref
                     for i in range(len(pred) - window + 1)))


def rouge_n(prediction: str, reference: str, n: int = 1) -> float:
    """ROUGE-N recall (Lin, 2004): share of reference n-grams found in the response."""
    pred = _ngrams(_normalize(prediction), n)
    ref = _ngrams(_normalize(reference), n)
    if not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    return overlap / sum(ref.values())


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure (Lin, 2004), based on the longest common subsequence."""
    pred = _normalize(prediction)
    ref = _normalize(reference)
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    denominator = recall + (beta ** 2) * precision
    return ((1 + beta ** 2) * precision * recall / denominator
            if denominator else 0.0)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length, computed with a rolling row."""
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


class IdfIndex:
    """Inverse document frequency over the evaluated set.

    The corpus is the run under evaluation, which is small. That is acceptable
    for the purpose here - down-weighting words that appear in every item - but
    it means the weights are not transferable between runs, so IDF-weighted
    figures are compared within a run and not across datasets.
    """

    def __init__(self, documents: Sequence[str]):
        self._n_docs = max(1, len(documents))
        document_frequency = Counter()
        for document in documents:
            document_frequency.update(set(_content_tokens(document)))
        self._df = document_frequency

    def weight(self, token: str) -> float:
        """Smoothed IDF weight of a stemmed token."""
        return math.log((self._n_docs + 1) / (self._df.get(token, 0) + 1)) + 1.0


def request_coverage(question: str, answer: str,
                     idf: Optional[IdfIndex] = None) -> float:
    """Share of the request's content words that the response addresses.

    Weighted by IDF when an index is supplied, so that the topical word of the
    utterance counts for more than a word common to every item.
    """
    question_tokens = set(_content_tokens(question))
    if not question_tokens:
        return float("nan")
    answer_tokens = set(_content_tokens(answer))

    if idf is None:
        return len(question_tokens & answer_tokens) / len(question_tokens)

    total = sum(idf.weight(t) for t in question_tokens)
    if total <= 0:
        return float("nan")
    matched = sum(idf.weight(t) for t in question_tokens & answer_tokens)
    return matched / total


def echo_ratio(question: str, answer: str) -> float:
    """Share of the response's content words that merely repeat the request.

    The companion to request_coverage: high coverage with a high echo ratio is a
    restatement rather than an answer. Without this second figure, coverage
    cannot distinguish the two.
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return float("nan")
    question_tokens = set(_content_tokens(question))
    return sum(1 for t in answer_tokens if t in question_tokens) / len(answer_tokens)


class EmbeddingBackend:
    """Optional sentence-embedding similarity.

    Wraps sentence-transformers if it is importable, and reports itself as
    unavailable otherwise. It is deliberately optional: the package is large and
    pulls in a deep-learning runtime, which would make the whole evaluation
    toolkit unusable on a host set up only for the CPU pipeline preset.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._model = None
        self.error: Optional[str] = None
        if not model_name:
            self.error = "not requested"
            return
        try:                                    # pragma: no cover - optional dep
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except Exception as exc:                # pragma: no cover - optional dep
            self.error = f"unavailable ({exc.__class__.__name__}: {exc})"

    @property
    def available(self) -> bool:
        return self._model is not None

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity of two texts, or NaN when the backend is absent."""
        if not self.available or not left.strip() or not right.strip():
            return float("nan")
        import numpy as np
        vectors = self._model.encode([left, right], normalize_embeddings=True)
        return float(np.dot(vectors[0], vectors[1]))


def evaluate_relevance(record: Record,
                       idf: Optional[IdfIndex] = None,
                       embeddings: Optional[EmbeddingBackend] = None
                       ) -> Dict[str, float]:
    """Compute all Tier 1 measures for one record.

    Reference-based entries are present only when the scenario spec supplies a
    reference answer; following the SQuAD convention, the best score over all
    references is reported when several are given.
    """
    metrics: Dict[str, float] = {
        "request_coverage": request_coverage(record.stt_text, record.llm_text, idf),
        "echo_ratio": echo_ratio(record.stt_text, record.llm_text),
    }

    if embeddings is not None and embeddings.available:
        metrics["request_response_cosine"] = embeddings.similarity(
            record.stt_text, record.llm_text)

    if record.has_reference_text:
        # Coverage of what the user meant, alongside coverage of what the
        # recognizer passed on. The model can only work from the second, so the
        # first is not a fairer measure of the model; it is the measure of the
        # assembled system, and the gap between them is the cost the recognizer
        # imposed. Reporting only one of the two attributes that cost to the
        # wrong component.
        metrics["intent_coverage"] = request_coverage(
            record.ori_text, record.llm_text, idf)
        gap = metrics["intent_coverage"] - metrics["request_coverage"]
        metrics["coverage_intent_gap"] = gap
        if embeddings is not None and embeddings.available:
            metrics["intent_response_cosine"] = embeddings.similarity(
                record.ori_text, record.llm_text)

    if record.reference_answers:
        f1_scores = [token_f1(record.llm_text, ref)["f1"]
                     for ref in record.reference_answers]
        metrics["reference_token_f1"] = max(f1_scores)
        metrics["answer_presence"] = max(answer_presence(record.llm_text, ref)
                                         for ref in record.reference_answers)
        metrics["reference_exact_match"] = max(exact_match(record.llm_text, ref)
                                               for ref in record.reference_answers)
        # Length of the answer the item is scored against. A gold span of a dozen
        # words is an extracted passage rather than an answer, and presence over
        # such items measures the extraction, not the model; carrying the length
        # per item is what lets that subset be reported apart.
        metrics["reference_answer_words"] = float(min(
            len(_normalize(ref)) for ref in record.reference_answers))
        metrics["reference_rouge_1"] = max(rouge_n(record.llm_text, ref, 1)
                                           for ref in record.reference_answers)
        metrics["reference_rouge_l"] = max(rouge_l(record.llm_text, ref)
                                           for ref in record.reference_answers)
        if embeddings is not None and embeddings.available:
            metrics["reference_cosine"] = max(
                embeddings.similarity(record.llm_text, ref)
                for ref in record.reference_answers)

    return metrics


def build_idf_index(records: Sequence[Record]) -> IdfIndex:
    """Build an IDF index from every utterance and response in the run.

    The reference utterance is included when present, so that coverage of the
    recognized text and coverage of the intended text are weighted by the same
    index and their difference is attributable to the texts rather than to the
    weighting.

    Reference answers are deliberately left out. They are an optional input, and
    admitting them would make the coverage of a question depend on whether an
    answer key happened to be supplied, so the same run would report two different
    coverage figures under two evaluations.
    """
    documents = []
    for record in records:
        documents.append(record.stt_text)
        documents.append(record.llm_text)
        documents.append(record.ori_text)
    return IdfIndex([d for d in documents if d and d.strip()])


REFERENCE_KEYS = ["squad_f1", "squad2", "answer_presence", "rouge", "bertscore",
                  "howntoeval"]
