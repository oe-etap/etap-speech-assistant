#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How faithfully the recognizer heard the user, measured against the reference text.

The transcript records both what the speaker was supposed to say (`ori_text`, the
prompt text of the recording) and what the recognizer produced (`stt_text`). The
difference between them is the part of every downstream failure that the language
model cannot be blamed for, and separating the two is the point of this module.

Three consequences follow, and they shape how the rest of the toolkit uses these
numbers:

Word error rate is a property of the recognizer and the audio, not of the model
under test. Runs that share a recording set and recognizer configuration produce
identical values, which makes the figure a control: if it moves between two runs
that were supposed to differ only in the language model, the comparison is
confounded.

`ori_text` is the user's intended *question*, not a correct answer. It supports a
relevance measure against what the user meant (see relevance.intent_coverage) and
a stratification of items by how badly the input was garbled. It must not be used
as a reference answer for overlap metrics; the answer to a question is not the
question.

A recognizer that emits no digits is not wrong for writing "two thousand four".
Reference text containing numerals is therefore scored against several spoken
readings of each numeral and credited with the best one, in the same spirit as
SQuAD scoring against multiple reference answers. Without this the error rate of
every item containing a year is overstated by the reading convention alone.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import textutils as tu
from .loaders import Record

# Boundaries of the input-quality strata. An item is clean when the recognizer
# reproduced the reference exactly, and severe once more than a third of the
# reference words were lost or altered, at which point the model is answering a
# materially different question. The bands are fixed here rather than derived
# from the data so that they mean the same thing across datasets.
CLEAN_THRESHOLD = 0.0
SEVERE_THRESHOLD = 1.0 / 3.0

STRATA = ("clean", "mild", "severe")

# Numerals are read out in more ways than one, and only a few are plausible.
# Beyond this many numeric tokens in one reference, only the primary reading is
# tried: the product of the alternatives would grow faster than the added
# accuracy justifies.
MAX_NUMERIC_TOKENS_EXPANDED = 3

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


@dataclass(frozen=True)
class Alignment:
    """Edit counts of one hypothesis against one reference, in words."""

    reference_length: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        """Errors per reference word. Unbounded above, as insertions are counted."""
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.reference_length


def tokens(text: str) -> List[str]:
    """Lowercased word tokens, with punctuation and apostrophes removed.

    Punctuation is dropped because the recognizer emits none, so keeping it would
    charge the recognizer for a convention of the written reference. Apostrophes
    are folded so that "dell's" and "dells" are the same token.
    """
    return [w.lower().replace("'", "") for w in tu.words(text)]


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Levenshtein alignment (Levenshtein, 1966) with the edits counted by type.

    Substitutions, deletions and insertions are reported separately because they
    fail differently downstream: a deleted word leaves the question incomplete,
    while a substitution can silently change what was asked.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return Alignment(0, 0, 0, m)
    if m == 0:
        return Alignment(n, 0, n, 0)

    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(cost[i - 1][j - 1],   # substitution
                                     cost[i - 1][j],       # deletion
                                     cost[i][j - 1])       # insertion

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] \
                and cost[i][j] == cost[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return Alignment(reference_length=n, substitutions=substitutions,
                     deletions=deletions, insertions=insertions)


def number_readings(token: str) -> List[str]:
    """Plausible spoken readings of a numeric token, best guess first.

    Returns an empty list for a token that is not numeric. A four-digit token is
    also offered in its year reading ("nineteen ninety"), which is how a speaker
    pronounces a date and therefore what the recognizer transcribes.
    """
    if not token or not any(c.isdigit() for c in token):
        return []

    if "." in token or "," in token:
        whole, _, fraction = token.replace(",", ".").partition(".")
        head = _cardinal(int(whole)) if whole.isdigit() else whole
        digits = " ".join(_UNITS[int(d)] for d in fraction if d.isdigit())
        return [f"{head} point {digits}".strip()]

    if not token.isdigit():
        return []

    value = int(token)
    readings = [_cardinal(value)]
    if 1100 <= value <= 2099 and len(token) == 4:
        readings.append(_year(value))
    if value >= 100:
        readings.append(" ".join(_UNITS[int(d)] for d in token))
    if 100 <= value <= 9999 and value % 100 != 0:
        readings.append(_cardinal(value, use_and=True))

    unique: List[str] = []
    for reading in readings:
        if reading and reading not in unique:
            unique.append(reading)
    return unique


def _cardinal(value: int, use_and: bool = False) -> str:
    """Read an integer below ten thousand as a cardinal number."""
    if value < 0:
        return f"minus {_cardinal(-value, use_and)}"
    if value < 20:
        return _UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        return _TENS[tens] + (f" {_UNITS[unit]}" if unit else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        head = f"{_UNITS[hundreds]} hundred"
        if not rest:
            return head
        joiner = " and " if use_and else " "
        return head + joiner + _cardinal(rest, use_and)
    if value < 10000:
        thousands, rest = divmod(value, 1000)
        head = f"{_UNITS[thousands]} thousand"
        if not rest:
            return head
        joiner = " and " if use_and and rest < 100 else " "
        return head + joiner + _cardinal(rest, use_and)
    return str(value)


def _year(value: int) -> str:
    """Read a four-digit year the way it is spoken: 1990 -> "nineteen ninety"."""
    century, rest = divmod(value, 100)
    if rest == 0:
        return f"{_cardinal(century)} hundred"
    if rest < 10:
        return f"{_cardinal(century)} o {_UNITS[rest]}"
    return f"{_cardinal(century)} {_cardinal(rest)}"


def reference_variants(reference: str) -> List[List[str]]:
    """Tokenized readings of the reference, expanding numerals where needed.

    The first entry is the literal tokenization. Further entries substitute
    spoken readings for numeric tokens, so that scoring can credit the reading
    the speaker actually used.
    """
    base = tokens(reference)
    numeric = [index for index, token in enumerate(base) if any(c.isdigit()
                                                                for c in token)]
    if not numeric:
        return [base]

    variants: List[List[str]] = []
    limited = len(numeric) > MAX_NUMERIC_TOKENS_EXPANDED
    for index in numeric:
        readings = number_readings(base[index])
        if limited:
            readings = readings[:1]
        for reading in readings:
            expanded = list(base)
            expanded[index:index + 1] = reading.split()
            variants.append(expanded)

    # Every numeric token replaced by its primary reading, for a reference that
    # contains more than one numeral.
    if len(numeric) > 1:
        combined = list(base)
        for index in reversed(numeric):
            readings = number_readings(combined[index])
            if readings:
                combined[index:index + 1] = readings[0].split()
        variants.append(combined)

    unique: List[List[str]] = [base]
    for variant in variants:
        if variant not in unique:
            unique.append(variant)
    return unique


def word_error_rate(reference: str, hypothesis: str) -> Alignment:
    """Word error rate of `hypothesis` against `reference`, best over readings.

    The alignment with the fewest errors over the reference readings is returned,
    so a numeral written as digits is not penalised for having been spoken aloud.
    """
    hypothesis_tokens = tokens(hypothesis)
    best: Optional[Alignment] = None
    for variant in reference_variants(reference):
        candidate = align(variant, hypothesis_tokens)
        if best is None or candidate.errors < best.errors:
            best = candidate
    return best if best is not None else Alignment(0, 0, 0, 0)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate over the concatenated word tokens.

    Computed on the token stream rather than the raw string so that spacing and
    punctuation do not enter it, and reported next to the word error rate because
    a recognizer that misses one letter of a name is not as wrong as one that
    substitutes a different word.
    """
    reference_chars = list(" ".join(tokens(reference)))
    hypothesis_chars = list(" ".join(tokens(hypothesis)))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else float("inf")
    return align(reference_chars, hypothesis_chars).error_rate


def content_recall(reference: str, hypothesis: str) -> float:
    """Share of the reference's topical words that survived recognition.

    The error rate counts every word equally, but a lost function word rarely
    changes the question while a lost topic word always does. This is the figure
    to read when asking whether the model could have answered at all.
    """
    from .relevance import _content_tokens

    reference_tokens = set(_content_tokens(reference))
    if not reference_tokens:
        return float("nan")
    hypothesis_tokens = set(_content_tokens(hypothesis))
    return len(reference_tokens & hypothesis_tokens) / len(reference_tokens)


def stratum_of(error_rate: Optional[float]) -> str:
    """Label an item by how badly the recognizer garbled it."""
    if error_rate is None:
        return "unknown"
    if error_rate <= CLEAN_THRESHOLD:
        return "clean"
    if error_rate <= SEVERE_THRESHOLD:
        return "mild"
    return "severe"


def evaluate_asr_fidelity(record: Record) -> Dict[str, Any]:
    """Every recognizer-fidelity measure for one item.

    Returns an empty mapping when the transcript carries no reference text, since
    a fidelity figure without a reference would be an assumption rather than a
    measurement.
    """
    if not record.ori_text.strip():
        return {}

    alignment = word_error_rate(record.ori_text, record.stt_text)
    return {
        "stt_wer": alignment.error_rate,
        "stt_cer": character_error_rate(record.ori_text, record.stt_text),
        "stt_ref_words": alignment.reference_length,
        "stt_word_errors": alignment.errors,
        "stt_substitutions": alignment.substitutions,
        "stt_deletions": alignment.deletions,
        "stt_insertions": alignment.insertions,
        "stt_exact_match": alignment.errors == 0,
        "stt_content_recall": content_recall(record.ori_text, record.stt_text),
        "stt_stratum": stratum_of(alignment.error_rate),
    }


def corpus_error_rate(records: Sequence[Record]) -> Tuple[float, int, int]:
    """Corpus word error rate, as (rate, total errors, total reference words).

    Pooled over words rather than averaged over items, which is the convention
    for a reported word error rate: a long utterance carries more weight than a
    three-word one because it contains more opportunities to err.
    """
    errors = reference_words = 0
    for record in records:
        if not record.ori_text.strip():
            continue
        alignment = word_error_rate(record.ori_text, record.stt_text)
        errors += alignment.errors
        reference_words += alignment.reference_length
    if reference_words == 0:
        return float("nan"), 0, 0
    return errors / reference_words, errors, reference_words


REFERENCE_KEYS = ["levenshtein", "wer_slu"]
