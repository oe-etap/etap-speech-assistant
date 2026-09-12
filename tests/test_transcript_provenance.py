#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What a text-mode row's `stt_engine` means, and where it comes from.

A text-mode run loads no recognizer, so the column cannot mean "what ran here".
It means what produced the text that ran -- the thing two text-mode cells must
agree on before their figures may be pooled, since the same LLM answering a
Vosk transcript and a Whisper transcript is not answering the same question.

These tests pin that contract at the two places it can break: reading the
provenance out of a transcripts file, and refusing a file that names more than
one recognizer.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assistant


def write_jsonl(directory, name, records):
    path = Path(directory) / name
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def record(filename, engine=None):
    entry = {"filename": filename, "ori_text": "reference",
             "stt_text": "recognized", "llm_text": ""}
    if engine is not None:
        entry["stt_engine"] = engine
    return entry


class TranscriptProvenanceTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_one_recognizer_is_reported(self):
        path = write_jsonl(self.tmp.name, "vosk.jsonl",
                           [record("a.wav", "vosk"), record("b.wav", "vosk")])
        records = assistant.load_transcript_items(path)
        self.assertEqual(assistant.transcript_stt_engine(records, path), "vosk")

    def test_two_recognizers_are_refused_rather_than_merged(self):
        """One file mixing two recognizers is not one experiment.

        Picking a winner would stamp every row with a value true of only some
        of them, which is the failure RUN_CONTEXT_COLUMNS exists to prevent.
        """
        path = write_jsonl(self.tmp.name, "mixed.jsonl",
                           [record("a.wav", "vosk"), record("b.wav", "whisper")])
        records = assistant.load_transcript_items(path)
        with self.assertRaises(ValueError) as caught:
            assistant.transcript_stt_engine(records, path)
        self.assertIn("more than one recognizer", str(caught.exception))
        self.assertIn("vosk", str(caught.exception))
        self.assertIn("whisper", str(caught.exception))

    def test_a_file_from_before_the_field_existed_reports_empty(self):
        """Empty is honest, and still a distinct value.

        An unknown provenance must not silently pool with a known one, and ""
        differs from "vosk" for RUN_CONTEXT_COLUMNS just as any two engine
        names differ from each other.
        """
        path = write_jsonl(self.tmp.name, "legacy.jsonl",
                           [record("a.wav"), record("b.wav")])
        records = assistant.load_transcript_items(path)
        self.assertEqual(assistant.transcript_stt_engine(records, path), "")

    def test_blank_values_do_not_count_as_a_second_recognizer(self):
        """A half-annotated file has one recognizer, not one and a blank."""
        path = write_jsonl(self.tmp.name, "partial.jsonl",
                           [record("a.wav", "vosk"), record("b.wav", ""),
                            record("c.wav")])
        records = assistant.load_transcript_items(path)
        self.assertEqual(assistant.transcript_stt_engine(records, path), "vosk")

    def test_provenance_does_not_replace_the_filename_requirement(self):
        """An engine name cannot stand in for the identity the join needs."""
        path = write_jsonl(self.tmp.name, "nameless.jsonl",
                           [{"stt_text": "x", "stt_engine": "vosk"}])
        with self.assertRaises(ValueError) as caught:
            assistant.load_transcript_items(path)
        self.assertIn("no filename", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
