#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal non-streaming Ollama client for the evaluation tiers.

The pipeline's own OllamaEngine streams tokens because time-to-first-audio
depends on it. Evaluation has the opposite requirement: it wants the complete
response and nothing else, so a separate blocking client keeps the latency-
critical code path unchanged and avoids inheriting its chunking behaviour.

Both the judge tier and the self-consistency tier use this client, which also
means an evaluation can be served by the same local Ollama instance as the
pipeline itself. No text leaves the host.
"""

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

DEFAULT_URL = "http://localhost:11434/api/generate"

# Reachability verdicts from OllamaClient.probe(), keyed by (url, model).
_PROBE_CACHE: Dict[Any, Optional[str]] = {}


def reset_probe_cache() -> None:
    """Forget cached reachability verdicts, so the next probe asks again."""
    _PROBE_CACHE.clear()


@dataclass
class GenerationResult:
    """One completion, with the accounting needed to report evaluation cost."""

    text: str
    ok: bool
    error: str = ""
    eval_count: int = 0
    duration_s: float = 0.0


@dataclass
class OllamaClient:
    """Blocking single-shot generation against a local Ollama server."""

    model: str
    url: str = DEFAULT_URL
    timeout: float = 300.0
    temperature: float = 0.0
    seed: Optional[int] = 0
    num_predict: int = 512
    retries: int = 2
    _session: Any = field(default=None, repr=False)
    calls: int = field(default=0, repr=False)
    failures: int = field(default=0, repr=False)
    total_eval_tokens: int = field(default=0, repr=False)
    total_seconds: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self._session = requests.Session()

    def probe(self) -> Optional[str]:
        """Return None if the model answers, or a human-readable reason if not.

        Called once before a tier starts so that an unreachable server or a
        missing model tag is reported as a configuration problem, rather than
        appearing later as every item scoring zero. The timeout is short: the
        point of probing is to fail fast when the tier cannot run at all.

        The verdict is cached per server and model tag, because several tiers
        usually share one, and asking each time only repeats the same wait.
        """
        cache_key = (self.url, self.model)
        if cache_key in _PROBE_CACHE:
            return _PROBE_CACHE[cache_key]

        result = self.generate("Reply with the single word: ready.",
                               num_predict=8, timeout=min(self.timeout, 30.0))
        verdict = None if result.ok else (result.error or "unknown error")
        _PROBE_CACHE[cache_key] = verdict
        return verdict

    def generate(self, prompt: str,
                 temperature: Optional[float] = None,
                 seed: Optional[int] = None,
                 num_predict: Optional[int] = None,
                 stop: Optional[List[str]] = None,
                 timeout: Optional[float] = None) -> GenerationResult:
        """Generate one completion, retrying only transient transport failures.

        Deterministic settings are the default: temperature 0 with a fixed seed,
        so that re-running an evaluation on unchanged inputs reproduces the same
        scores. Sampling is opt-in per call, which is what the self-consistency
        tier needs.

        A refused connection is not retried. Nothing about it is transient, and
        retrying it turned a tier that cannot run into a minute of waiting before
        it said so.
        """
        options: Dict[str, Any] = {
            "num_predict": self.num_predict if num_predict is None else num_predict,
            "temperature": self.temperature if temperature is None else temperature,
        }
        effective_seed = self.seed if seed is None else seed
        if effective_seed is not None:
            options["seed"] = effective_seed
        if stop:
            options["stop"] = stop

        payload = {"model": self.model, "prompt": prompt,
                   "stream": False, "options": options}

        last_error = ""
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                response = self._session.post(
                    self.url, json=payload,
                    timeout=self.timeout if timeout is None else timeout)
                response.raise_for_status()
                data = response.json()
                elapsed = time.perf_counter() - started

                self.calls += 1
                self.total_seconds += elapsed
                self.total_eval_tokens += int(data.get("eval_count") or 0)

                return GenerationResult(
                    text=str(data.get("response") or ""),
                    ok=True,
                    eval_count=int(data.get("eval_count") or 0),
                    duration_s=elapsed)
            except Exception as exc:
                last_error = _describe_error(exc, self.url, self.model)
                if _is_permanent(exc) or attempt >= self.retries:
                    break
                time.sleep(1.0 + attempt)

        self.calls += 1
        self.failures += 1
        return GenerationResult(text="", ok=False, error=last_error)

    def cost_summary(self) -> Dict[str, Any]:
        """Report how much generation this client performed, for the run log."""
        return {
            "model": self.model,
            "calls": self.calls,
            "failures": self.failures,
            "generated_tokens": self.total_eval_tokens,
            "wall_seconds": round(self.total_seconds, 2),
        }


def _describe_error(exc: Exception, url: str, model: str) -> str:
    """Summarise a transport failure in one line naming its likely cause.

    The raw exception from a refused connection is a nested urllib3 chain that
    runs to several hundred characters, which then lands verbatim in the report
    header. What a reader needs is which of the two setup steps is missing: the
    server, or the model tag.
    """
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (f"cannot reach an Ollama server at {url} "
                f"(start it with 'ollama serve')")
    if isinstance(exc, requests.exceptions.Timeout):
        return f"{model} did not respond within the timeout"

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 404:
        return (f"the server has no model tagged '{model}' "
                f"(pull it with 'ollama pull {model}')")
    if status is not None:
        detail = (getattr(response, "text", "") or "").strip().replace("\n", " ")
        if len(detail) > 160:
            detail = detail[:157] + "..."
        return f"server returned HTTP {status}" + (f": {detail}" if detail else "")
    return f"{exc.__class__.__name__}: {exc}"


def _is_permanent(exc: Exception) -> bool:
    """Whether an error will recur identically on retry.

    A refused connection means no server is listening and a 404 means the model
    tag does not exist; both are configuration problems that a second attempt
    cannot change. Timeouts and server-side errors are treated as transient.
    """
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status is not None and 400 <= int(status) < 500


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Recover the first complete JSON object from a model response.

    Small local models wrap JSON in prose or a fenced code block even when told
    not to, so the raw response is rarely parseable as-is. Braces are matched
    with a depth counter that ignores brackets inside string literals, then the
    candidate is parsed; a light repair pass handles trailing commas and single
    quotes, which are the two malformations that actually recur.
    """
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1).strip())

    block = _first_balanced_object(text)
    if block:
        candidates.append(block)
    candidates.append(text.strip())

    for candidate in candidates:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed
    return None


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the substring spanning the first brace-balanced object."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object candidate, applying minimal repairs on failure."""
    for attempt in (candidate, _repair(candidate)):
        try:
            parsed = json.loads(attempt)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _repair(candidate: str) -> str:
    """Remove trailing commas and convert single-quoted keys and values."""
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    repaired = re.sub(r"'([^'\"]*)'\s*:", r'"\1":', repaired)
    return re.sub(r":\s*'([^'\"]*)'", r': "\1"', repaired)
