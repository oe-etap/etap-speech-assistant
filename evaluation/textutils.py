#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared text primitives for the evaluation package.

Sentence segmentation, word tokenization and syllable counting are needed by
several metrics, and every one of them changes its result if these differ. They
therefore live here rather than being reimplemented per module: a response
scored as "two sentences" by the constraint checker must be the same two
sentences the readability formula divides by.

No third-party NLP dependency is used. The segmenter handles the cases that
occur in short spoken-assistant replies (abbreviations, single-letter initials,
decimals, quoted titles); it is not a general-purpose sentence splitter.
"""

import re
import unicodedata
from typing import List

# Sentence-final punctuation. Semicolons and colons are deliberately absent:
# a prompt asking for a sentence "ending with a full stop" is about . ! ?, and
# counting clause boundaries as sentences would silently pass responses that
# never terminate a sentence at all.
_SENTENCE_END = ".!?"

# Characters that belong to the sentence they follow, not to the next one.
_TRAILING = "\"')]}”’»…"

# Tokens whose trailing period does not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc",
    "e.g", "i.e", "cf", "al", "fig", "no", "approx", "inc", "ltd", "co",
    "dept", "univ", "vol", "ca", "ph.d", "u.s", "u.k", "a.m", "p.m",
}

_WORD_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+(?:['’][^\W\d_]+)*",
                      re.UNICODE)

# Function words that are frequent in English and rare as whole words in other
# Latin-script languages. Used only for a coarse "is this English" ratio.
_ENGLISH_MARKERS = {
    "the", "of", "and", "to", "in", "is", "it", "you", "that", "for", "on",
    "with", "as", "are", "this", "be", "have", "from", "or", "an", "but",
    "not", "they", "which", "was", "can", "your", "if", "there", "their",
    "what", "when", "how", "would", "should", "could", "about", "more",
}


def normalize_unicode(text: str) -> str:
    """Fold typographic punctuation so that regexes see one spelling of each mark.

    Model output mixes straight and curly quotes within a single response, and a
    check written against one form silently misses the other.
    """
    text = unicodedata.normalize("NFKC", text)
    return (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2013", "-").replace("\u2014", "-"))


def words(text: str) -> List[str]:
    """Return word tokens, keeping internal apostrophes and decimal numbers.

    "Australia's" counts as one word and "3.14" as one number, which is what a
    word-count constraint on a spoken sentence means.
    """
    return _WORD_RE.findall(normalize_unicode(text))


def word_count(text: str) -> int:
    """Return the number of word tokens in `text`."""
    return len(words(text))


def split_sentences(text: str) -> List[str]:
    """Split `text` into sentences at . ! ? boundaries.

    A boundary is rejected when the period belongs to an abbreviation, to a
    single-letter initial ("Arthur C. Clarke"), or to a decimal number. This
    matters because an over-eager split inflates the sentence count and shrinks
    the mean sentence length, which moves both the constraint verdict and the
    readability score.
    """
    text = normalize_unicode(text).strip()
    if not text:
        return []

    sentences = []
    start = 0
    i = 0
    length = len(text)

    while i < length:
        if text[i] not in _SENTENCE_END:
            i += 1
            continue

        end = i + 1
        while end < length and text[end] in _TRAILING:
            end += 1

        # A boundary needs whitespace or end-of-text after it, otherwise this is
        # an intra-token dot such as "u.s.a" or "3.14".
        if end < length and not text[end].isspace():
            i += 1
            continue

        if text[i] == "." and _is_non_terminal_period(text, start, i):
            i += 1
            continue

        candidate = text[start:end].strip()
        if candidate:
            sentences.append(candidate)
        start = end
        i = end

    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _is_non_terminal_period(text: str, start: int, dot: int) -> bool:
    """Check whether the period at `dot` is part of a token rather than a boundary."""
    if dot > 0 and text[dot - 1].isdigit():
        # "1948." at the end of a sentence is a real boundary; "3.14" is not.
        return dot + 1 < len(text) and text[dot + 1].isdigit()

    token = re.split(r"[\s(\[\"']", text[start:dot])[-1].lower()
    if not token:
        return False
    if len(token) == 1 and token.isalpha():
        return True           # single-letter initial, e.g. "C."
    return token.strip(".") in _ABBREVIATIONS or token in _ABBREVIATIONS


def count_syllables(word: str) -> int:
    """Estimate the syllable count of an English word.

    Vowel-group heuristic with silent-e removal, the same rule family the
    readability formulas were originally tabulated with. Per-word error is
    tolerated because both Flesch formulas use the corpus-level ratio of
    syllables to words, where the errors largely cancel.
    """
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return 0
    if len(token) <= 3:
        return 1
    token = re.sub(r"(?:[^laeiouy]es|[^laeiouy]e|ed)$", "", token)
    token = re.sub(r"^y", "", token)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", token)))


def english_marker_ratio(text: str) -> float:
    """Return the share of tokens that are common English function words.

    A coarse language signal, not a language identifier. It is used only to flag
    a response that left the requested language entirely; short factual replies
    can legitimately score low, so the threshold is set permissively.
    """
    tokens = [w.lower() for w in words(text)]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _ENGLISH_MARKERS) / len(tokens)


def estimate_tokens(text: str) -> int:
    """Estimate the LLM token count of `text`.

    Used only to decide whether a response plausibly hit its `num_predict` cap.
    The word-based factor is calibrated for English prose on subword vocabularies;
    the character floor keeps dense text (numbers, punctuation, code-like spans)
    from being underestimated. It is never reported as a measured token count:
    Ollama's own `eval_count` in the latency log is the ground truth.
    """
    n_words = word_count(text)
    n_chars = len(normalize_unicode(text))
    return int(round(max(n_words * 1.33, n_chars / 6.0)))


def spoken_seconds(text: str, words_per_second: float = 2.5) -> float:
    """Estimate how long `text` takes to speak aloud.

    The default rate corresponds to roughly 150 words per minute, typical for
    neutral read English. Reported as an estimate: the actual duration depends
    on the TTS voice and is measurable from the synthesized WAV.
    """
    return word_count(text) / words_per_second if words_per_second > 0 else 0.0


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease (Flesch, 1948). Higher is easier; 60-70 is plain English."""
    stats = _readability_counts(text)
    if stats is None:
        return float("nan")
    n_words, n_sentences, n_syllables = stats
    return (206.835
            - 1.015 * (n_words / n_sentences)
            - 84.6 * (n_syllables / n_words))


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level (Kincaid et al., 1975), in US school grades."""
    stats = _readability_counts(text)
    if stats is None:
        return float("nan")
    n_words, n_sentences, n_syllables = stats
    return (0.39 * (n_words / n_sentences)
            + 11.8 * (n_syllables / n_words)
            - 15.59)


def _readability_counts(text: str):
    """Return (words, sentences, syllables) or None when the text is too short.

    Both formulas divide by the word and sentence counts, so an empty response
    has no defined readability rather than a score of zero.
    """
    tokens = words(text)
    sentences = split_sentences(text)
    if not tokens or not sentences:
        return None
    return len(tokens), len(sentences), sum(count_syllables(w) for w in tokens)
