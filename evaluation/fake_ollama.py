#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A stand-in Ollama server for exercising the model-backed evaluation tiers.

Run with:  python -m evaluation.fake_ollama [--port 11434]

The self-test covers the judge and self-consistency tiers with a stub client,
which replaces `generate()` and therefore never touches HTTP. That leaves the
transport itself untested: request shape, response parsing, timeouts, and the
report sections that only appear once a judge actually produces scores. This
server closes that gap without needing a model on the machine.

It answers `/api/generate` with the field names the real server uses, and it
infers from the prompt which tier is asking, because each tier expects a
different shape of reply. Scores are derived from the prompt text so that a run
is reproducible and so the aggregate is not uniform across dimensions.

This is a test double, not a mock of model behaviour: the ratings it returns say
nothing about response quality and must never appear in reported results.
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict


def _stable_score(prompt: str, scale_min: int, scale_max: int) -> int:
    """Pick a rating that varies by dimension but repeats for the same prompt.

    A constant rating would hide bugs in aggregation: a weighted composite, an
    unweighted mean and a median all agree when every input is identical.
    """
    span = scale_max - scale_min + 1
    digest = sum(ord(char) * (index + 1) for index, char in enumerate(prompt))
    return scale_min + (digest % span)


def _scale_bounds(prompt: str) -> tuple:
    match = re.search(r"integer\s+(\d+)\s*-\s*(\d+)", prompt)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1, 5


def _reply_for(prompt: str) -> str:
    """Produce the reply shape the asking tier expects."""
    lowered = prompt.lower()

    if '"winner"' in lowered:
        # Pairwise comparison. Answering "A" in both presentation orders is
        # pure position bias, which the position-bias control must catch.
        return json.dumps({"winner": "A", "reason": "stub prefers the first"})

    if '"claims"' in lowered:
        return json.dumps({"claims": ["a stub atomic claim about a capital city",
                                      "a second stub atomic claim"]})

    if '"label"' in lowered:
        # Cycle the labels so the precision denominator excludes the
        # unverifiable one, which a single constant label would not exercise.
        label = ("supported", "unsupported", "unverifiable")[
            len(prompt) % 3]
        return json.dumps({"label": label, "reason": "stub verification"})

    if '"score"' in lowered:
        scale_min, scale_max = _scale_bounds(prompt)
        return json.dumps({"score": _stable_score(prompt, scale_min, scale_max),
                           "reason": "stub rating, not a quality signal"})

    if "reply with the single word" in lowered:
        return "ready"

    # Self-consistency resampling asks for a fresh answer to the user's turn.
    return "Canberra is the capital of Australia, chosen as a compromise."


class _Server(HTTPServer):
    """An HTTP server that refuses to share its port.

    HTTPServer sets SO_REUSEADDR, which on Windows lets a second instance bind
    the same port without error; connections then go to whichever socket the
    stack picks. A stale instance left running silently answers the requests, so
    edits to this file appear to have no effect. Failing the bind instead makes
    that state impossible to enter unnoticed.
    """

    allow_reuse_address = False


class _Handler(BaseHTTPRequestHandler):

    def do_POST(self) -> None:  # noqa: N802  (name fixed by BaseHTTPRequestHandler)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload: Dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "malformed JSON")
            return

        if not self.path.startswith("/api/generate"):
            self.send_error(404, "unknown endpoint")
            return

        text = _reply_for(str(payload.get("prompt") or ""))
        body = json.dumps({
            "model": payload.get("model", "stub"),
            "response": text,
            "done": True,
            "eval_count": len(text.split()),
        }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        """Stay quiet: request logs would bury the evaluation's own output."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    try:
        server = _Server((args.host, args.port), _Handler)
    except OSError as exc:
        print(f"Cannot bind {args.host}:{args.port} ({exc}). Another instance, "
              f"or a real Ollama, is already listening there.", flush=True)
        return 1
    print(f"Fake Ollama listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
