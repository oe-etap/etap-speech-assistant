#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handoff channels between the pipeline worker threads.

The stages of the pipeline need different handoff semantics, and picking the
wrong one costs latency. TTS chunks are a stream that must all be spoken, so a
plain queue.Queue is right for them. Utterances are not: only the newest one is
worth answering, which is what UtteranceMailbox provides.
"""

import threading


class UtteranceMailbox:
    """Single-slot handoff carrying only the newest finalized utterance.

    A queue is the wrong shape for turn-taking: once the user has said something
    new, finishing an answer to the previous utterance is wasted work. One slot
    means a superseded utterance is dropped rather than queued, and the LLM
    worker can cheaply check mid-generation whether it has been overtaken.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._text = None
        self._closed = False

    def put(self, text: str) -> None:
        """Hand over an utterance, replacing any that has not been picked up."""
        with self._condition:
            self._text = text
            self._condition.notify_all()

    def close(self) -> None:
        """Signal that no further utterances will arrive."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def take(self):
        """Block until an utterance arrives; return None once closed and empty."""
        with self._condition:
            while self._text is None and not self._closed:
                self._condition.wait()
            text, self._text = self._text, None
            return text

    def has_pending(self) -> bool:
        """Check without blocking whether a newer utterance is waiting."""
        with self._condition:
            return self._text is not None
