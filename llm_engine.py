#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM engine abstraction layer.
Provides OllamaEngine for streaming text generation via the Ollama API.
"""

import json
import time

import requests


# Stop sequences to prevent the model from generating synthetic multi-turn dialogue
_DEFAULT_STOP_TOKENS = ["\nUser:", "\nHuman:", "\n---", "---", "<|end|>", "<|user|>"]

# Fallback used when no prompt file is configured. Prompt variants for
# experiments live in prompts/ instead, see load_system_prompt() in
# mwe_assistant.py.
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, factual but friendly voice assistant. "
    "Answer the user's question in English in 1-3 medium length sentences. "
    "Do not simulate a conversation. Do not generate follow-up questions "
    "or responses from the user. Provide only your single answer."
)

# Sentence-level split points. Chunking here keeps enough context for the TTS
# model to place stress correctly.
_SENTENCE_TERMINATORS = ".?!:;\n"

# Kept with the preceding sentence rather than starting the next chunk
_CLOSING_CHARS = "\"')]}”’»"

DEFAULT_CHUNK_MAX_CHARS = 140


class TextChunker:
    """Split a streaming LLM response into TTS-friendly chunks.

    Sentence boundaries are the primary split points, so every chunk carries a
    full clause or sentence of context for the TTS model. A character limit acts
    as a safety net: without it a single long sentence keeps the TTS idle until
    the whole response has been generated, which is exactly what drives TTFA up.

    Note: abbreviations ("Dr.", "e.g.") are not detected and will split early.
    Decimal points are guarded, since "3.14" is common in factual answers.
    """

    def __init__(self, max_chars: int = DEFAULT_CHUNK_MAX_CHARS, min_chars: int = 3):
        """Initialize the chunker.

        Args:
            max_chars: Safety-net length. Once the buffer exceeds this without a
                sentence terminator, it is split at the last word boundary.
            min_chars: Chunks shorter than this are not emitted on their own;
                the buffer keeps accumulating instead.
        """
        self._max_chars = max_chars
        self._min_chars = min_chars
        self._buffer = ""

    def feed(self, piece: str) -> list:
        """Append newly generated text and return the chunks that are ready.

        Args:
            piece: Text fragment from the LLM stream (usually a single token).

        Returns:
            List of chunk strings, possibly empty.
        """
        self._buffer += piece
        chunks = []
        while True:
            split_at = self._next_split()
            if split_at is None:
                break
            # Always consume: a split point that is never consumed would be
            # rediscovered on every feed() and the buffer would stop advancing.
            chunk, self._buffer = self._buffer[:split_at].strip(), self._buffer[split_at:]
            if self._is_speakable(chunk):
                chunks.append(chunk)
        return chunks

    def flush(self) -> str:
        """Return whatever is left in the buffer and clear it.

        Returns an empty string if the remainder holds nothing speakable, which
        happens when a closing quote trails a sentence that was already emitted.
        """
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder if self._is_speakable(remainder) else ""

    def _next_split(self):
        """Return where to cut the buffer, or None if nothing is ready yet.

        Every returned index is greater than zero, which is what guarantees the
        buffer keeps shrinking.
        """
        split_at = self._find_sentence_end()
        if split_at is None and len(self._buffer) > self._max_chars:
            split_at = self._find_word_boundary()
        return split_at

    @staticmethod
    def _is_speakable(text: str) -> bool:
        """Check that the text holds something to pronounce, not just punctuation."""
        return any(char.isalnum() for char in text)

    def _find_sentence_end(self):
        """Return the index just past the first usable sentence terminator, or None.

        Terminators that would leave too little to pronounce are skipped rather
        than returned, so a leading newline or a stray "." merges into the next
        sentence instead of producing an unspeakable chunk.

        Splitting is deferred while the terminator is still the last character
        of the buffer and could turn out to be part of a decimal number: at that
        point "3." may still become "3.14".
        """
        for i, char in enumerate(self._buffer):
            if char not in _SENTENCE_TERMINATORS:
                continue
            if char == "." and self._is_decimal_point(i):
                continue
            if char == "." and self._may_become_decimal(i):
                return None

            end = self._skip_closing_chars(i + 1)
            candidate = self._buffer[:end].strip()
            if len(candidate) < self._min_chars or not self._is_speakable(candidate):
                continue
            return end
        return None

    def _is_decimal_point(self, index: int) -> bool:
        """Check whether the dot at `index` sits between two digits (e.g. 3.14)."""
        if index == 0 or index + 1 >= len(self._buffer):
            return False
        return self._buffer[index - 1].isdigit() and self._buffer[index + 1].isdigit()

    def _may_become_decimal(self, index: int) -> bool:
        """Check for a trailing "<digit>." that the next token could extend."""
        return (index + 1 >= len(self._buffer)
                and index > 0
                and self._buffer[index - 1].isdigit())

    def _skip_closing_chars(self, index: int) -> int:
        """Advance past closing quotes and brackets so they stay with the sentence."""
        while index < len(self._buffer) and self._buffer[index] in _CLOSING_CHARS:
            index += 1
        return index

    def _find_word_boundary(self):
        """Return the last whitespace index within max_chars, or the hard limit."""
        window = self._buffer[:self._max_chars]
        cut = window.rfind(" ")
        return cut + 1 if cut > 0 else self._max_chars


class OllamaEngine:
    """Ollama-based LLM engine with streaming generation and cancellation support."""

    def __init__(self, model: str, url: str, system_prompt: str = None,
                 stop_tokens: list = None, max_tokens: int = 150,
                 temperature: float = 0.7, seed: int = None,
                 chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS):
        """Initialize the Ollama engine.

        Args:
            model: Ollama model name (e.g. "phi3:mini").
            url: Ollama generate endpoint URL.
            system_prompt: Custom system prompt (uses default if None).
            stop_tokens: Stop sequences to prevent multi-turn hallucination.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            seed: Sampling seed. Combined with temperature 0 this makes runs
                reproducible, which is what an A/B comparison needs: response
                length varies enough between runs to hide the effect under test.
            chunk_max_chars: Safety-net chunk length passed to TextChunker.
        """
        self._model = model
        self._url = url
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._stop_tokens = stop_tokens if stop_tokens is not None else _DEFAULT_STOP_TOKENS
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed
        self._chunk_max_chars = chunk_max_chars
        self._session = requests.Session()
        self._current_response = None
        self._cancel_requested = False

    def _build_prompt(self, user_text: str) -> str:
        """Wrap the recognized text in the framing the model is asked to answer.

        The system prompt leads, so every request shares a long identical prefix
        and differs only in the user's words. That is what makes warmup() worth
        doing: the prefix can stay in the server's cache between requests.
        """
        return (
            f'{self._system_prompt}\n\n'
            f'The user said: "{user_text}"\n\n'
            f'Answer:'
        )

    def warmup(self):
        """Load the model and leave the shared prompt prefix in its KV cache.

        Sending the real framing rather than an empty string costs nothing extra
        and means the first utterance is evaluated the way every later one will
        be: only its own words are new. Without it the first request pays to
        evaluate the whole system prompt, which shows up as an outlier several
        times the size of every other measurement in the run.
        """
        try:
            self._session.post(self._url, json={
                "model": self._model,
                "prompt": self._build_prompt(""),
                "options": {"num_predict": 1}
            }, timeout=120)
        except Exception as e:
            print(f"[WARN] Failed to warmup Ollama: {e}")

    def cancel(self):
        """Cancel the current streaming response by closing the HTTP connection.

        Closing the response drops the socket, which makes Ollama abort the
        generation server-side instead of finishing it for a client that is
        no longer listening.
        """
        self._cancel_requested = True
        resp = self._current_response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
            self._current_response = None

    def generate_stream(self, user_text: str):
        """Stream LLM response, yielding TTS-ready chunks as they are generated.

        Chunks are cut at sentence boundaries by TextChunker, with a character
        limit as a safety net, so the TTS can start on the first sentence while
        the rest of the response is still being generated.

        Yields dicts with keys:
            - "text" (str): The synthesizable text chunk.
            - "ollama_stats" (dict|None): Server-side metrics (only on last yield).
            - "cancelled" (bool): True if the stream was cancelled mid-generation.
            - "first_token_t" (float|None): perf_counter timestamp of the first
              generated token, for measuring time-to-first-token separately from
              time-to-first-chunk.

        The last yielded dict may have empty "text" and non-None "ollama_stats"
        containing Ollama's server-side performance metrics.
        """
        # Cancel any previous stream before starting a new one
        self.cancel()
        self._cancel_requested = False

        prompt = self._build_prompt(user_text)

        chunker = TextChunker(max_chars=self._chunk_max_chars)
        ollama_stats = None
        first_token_t = None

        try:
            # Two different "stream" flags are needed here and they mean
            # different things:
            #   - "stream": True in the JSON body tells Ollama to emit NDJSON.
            #   - stream=True below tells requests NOT to buffer the whole body
            #     before returning. Without it, post() blocks until generation
            #     finishes and no token can reach the TTS early.
            options = {
                "num_predict": self._max_tokens,
                "temperature": self._temperature,
                "stop": self._stop_tokens,
            }
            if self._seed is not None:
                options["seed"] = self._seed

            r = self._session.post(self._url, json={
                "model": self._model,
                "prompt": prompt,
                "stream": True,
                "options": options,
            }, stream=True, timeout=120)
            # Assign before raise_for_status() so cancel() can close a response
            # that is still starting up.
            self._current_response = r

            with r:
                r.raise_for_status()

                socket_buffer = ""
                # chunk_size=None yields one HTTP chunk (roughly one token) at a
                # time for a chunked response, but the chunk may hold a partial
                # NDJSON line, so lines are reassembled here.
                for raw in r.iter_content(chunk_size=None):
                    if not raw:
                        continue
                    socket_buffer += raw.decode("utf-8", errors="ignore")
                    while "\n" in socket_buffer:
                        line, socket_buffer = socket_buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        data = json.loads(line)

                        # Capture server-side stats from the final message
                        if data.get("done", False):
                            ollama_stats = {
                                "prompt_eval_count": data.get("prompt_eval_count", 0),
                                "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
                                "eval_count": data.get("eval_count", 0),
                                "eval_duration_ns": data.get("eval_duration", 0),
                                "total_duration_ns": data.get("total_duration", 0),
                            }

                        piece = data.get("response", "")
                        if piece and first_token_t is None:
                            first_token_t = time.perf_counter()

                        for chunk in chunker.feed(piece):
                            yield {"text": chunk, "ollama_stats": None,
                                   "cancelled": False, "first_token_t": first_token_t}

            # Flush whatever did not reach a split point
            remainder = chunker.flush()
            if remainder:
                yield {"text": remainder, "ollama_stats": None,
                       "cancelled": False, "first_token_t": first_token_t}

            # Yield final metadata-only entry with server-side stats
            if ollama_stats:
                yield {"text": "", "ollama_stats": ollama_stats,
                       "cancelled": False, "first_token_t": first_token_t}

        except Exception as e:
            if self._cancel_requested:
                # Closing the response mid-stream surfaces as a connection or
                # I/O error; that is the expected outcome of cancel().
                yield {"text": "", "ollama_stats": None, "cancelled": True,
                       "first_token_t": first_token_t}
            else:
                yield {"text": f"(LLM call failed: {e})", "ollama_stats": None,
                       "cancelled": False, "first_token_t": first_token_t}
        finally:
            self._current_response = None
