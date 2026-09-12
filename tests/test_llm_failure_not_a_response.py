#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A generation that failed must not be measured, spoken or logged as a reply.

Before this was fixed, an unreachable Ollama produced a complete-looking item:
the exception text was yielded as an ordinary chunk, so the TTS synthesized it,
`ttfa` timed the synthesis of an error message -- 4081 ms against the same
recording's real 1435 ms -- and `transcripts.jsonl` carried the exception as the
assistant's answer. Nothing downstream could tell it from a slow configuration.

These tests pin the three places that has to be prevented: the engine says it
failed rather than disguising it, the worker refuses to feed the TTS, and the
transcript keeps the error out of `llm_text`.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assistant
import llm_engine
from pipeline_channels import UtteranceMailbox
import queue as queue_mod


class FailureSignalTest(unittest.TestCase):
    """What `generate_stream` yields when the request cannot be made.

    No mocking of the engine's internals: pointing it at a closed port
    exercises the same `except` branch a 500 takes, and a refused connection
    comes back immediately.
    """

    DEAD_URL = "http://localhost:1/api/generate"

    def _engine(self):
        return llm_engine.OllamaEngine(model="gemma3:1b-it-q4_K_M",
                                       url=self.DEAD_URL,
                                       system_prompt="be brief")

    def test_the_error_is_flagged_and_the_text_is_empty(self):
        """The message must not ride in `text`.

        A caller that forwards `text` without checking then synthesizes
        nothing, which is the safe failure rather than speaking the exception
        and logging it as the assistant's reply.
        """
        chunks = list(self._engine().generate_stream("hello"))
        self.assertTrue(chunks, "a failure must still yield something")
        last = chunks[-1]
        self.assertTrue(last.get("failed"))
        self.assertFalse(last.get("cancelled"))
        self.assertEqual(last.get("text"), "")
        self.assertTrue(last.get("error"))

    def test_a_cancel_requested_beforehand_does_not_mask_a_failure(self):
        """Each generation starts with a clean cancel flag.

        `generate_stream` resets `_cancel_requested` before issuing the
        request, so a cancel belonging to the previous utterance cannot make
        the next one's failure look like a barge-in -- which would hide a dead
        server behind an outcome the pipeline treats as normal. The in-flight
        cancel path itself is pinned by tests/test_bargein_accounting.py.
        """
        engine = self._engine()
        engine.cancel()
        last = list(engine.generate_stream("hello"))[-1]
        self.assertTrue(last.get("failed"))
        self.assertFalse(last.get("cancelled"))


class WorkerRefusesToSpeakTheErrorTest(unittest.TestCase):
    """`llm_worker` must put nothing synthesizable on the TTS queue."""

    def test_no_text_reaches_the_tts_and_the_error_is_recorded(self):
        engine = mock.Mock()
        engine.generate_stream.return_value = iter([
            {"text": "", "ollama_stats": None, "cancelled": False,
             "failed": True, "error": "500 Server Error",
             "first_token_t": None},
        ])
        mailbox = UtteranceMailbox()
        mailbox.put("what did she own")
        # close(), not put(None): the mailbox holds a single slot, so put(None)
        # erases the utterance rather than signalling the end, and take() then
        # blocks for ever.
        mailbox.close()
        tts_queue = queue_mod.Queue()
        metrics = {"full_assistant_text": "", "llm_chunk_count": 0,
                   "llm_ttfc_ms": None, "llm_ttft_ms": None,
                   "first_chunk_chars": None, "llm_error": None,
                   "ollama_stats": None, "ttft_stats": None,
                   "ttfc_stats": None, "end_stats": None}

        assistant.llm_worker(engine, mailbox, tts_queue, metrics)

        self.assertIn("500 Server Error", metrics["llm_error"] or "")
        self.assertEqual(metrics["full_assistant_text"], "")
        self.assertEqual(metrics["llm_chunk_count"], 0)

        drained = []
        while not tts_queue.empty():
            drained.append(tts_queue.get())
        speakable = [m for m in drained
                     if isinstance(m, dict) and m.get("type") == "text"]
        self.assertEqual(speakable, [],
                         "a failed generation must hand the TTS nothing to say")


class TranscriptKeepsTheErrorOutOfTheAnswerTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.item = assistant.ItemWork(
            index=1, name="00012", filename="00012.wav", ori_text="reference",
            ref_speech_end_s=None, audio_source=None, normalized_wav=None,
            user_text=None)

    def _record(self):
        path = Path(self.tmp.name) / "transcripts.jsonl"
        return json.loads(path.read_text(encoding="utf-8").strip())

    def test_a_failure_leaves_llm_text_empty_and_names_the_reason(self):
        """The evaluation package scores `llm_text`.

        An exception string there would be graded as though the assistant had
        answered, and the transcripts travel to other tools by filename rather
        than by run directory, so the failure has to travel with them.
        """
        assistant.write_transcript_record(
            self.tmp.name, self.item, "recognized words", "", "vosk",
            llm_error="500 Server Error")
        record = self._record()
        self.assertEqual(record["llm_text"], "")
        self.assertIn("500", record["llm_error"])
        self.assertEqual(record["stt_text"], "recognized words")

    def test_a_healthy_item_carries_no_error_field(self):
        assistant.write_transcript_record(
            self.tmp.name, self.item, "recognized words", "the answer", "vosk")
        record = self._record()
        self.assertEqual(record["llm_text"], "the answer")
        self.assertNotIn("llm_error", record)


if __name__ == "__main__":
    unittest.main()
