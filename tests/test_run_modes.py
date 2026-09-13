#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What each run mode measures, and what it must leave unmeasured.

The three modes exist to make the campaign affordable: an `--asr-only` pass pays
realtime pacing once and freezes the transcripts, and every configuration cell
then runs from those under `--input-mode text` at full speed. Both halves are
only useful if the rows they write can be told apart from a full run's, and the
failure mode is silent -- `aggregate_logs.py` cannot distinguish a zero from a
measurement, so a placeholder row or a zeroed `extra_json` key drags an aggregate
down without anything looking wrong. That is what most of this file asserts.

Two further things can only be settled by running the real `main()`:

  - whether the `item` column agrees across modes. Task 07 joins a text-mode row
    to the file-mode row for the same recording on it, and
    `evaluation/comparison.py` pairs items on the filename, so a mismatch breaks
    both joins and neither complains.
  - whether the engines a mode has no use for are left unbuilt. Skipping
    generation is not enough: a resident Ollama model holds VRAM for the length
    of the pass and a loaded Piper holds its thread pool, so the assertion has to
    be that the constructor was never called.

The three launches are chained the way the campaign chains them -- the ASR pass
writes the transcripts that the text-mode launch reads, with no step in between
-- so the handoff is exercised rather than described.

Engine doubles rather than Vosk, Ollama and Piper, and deliberately simpler than
the ones in test_bargein_accounting.py: nothing here turns on a wall-clock
instant, only on which rows exist, so a double that records its own construction
is worth more than one with realistic timings.

Run:  python -m unittest discover -s tests -v
"""

import csv
import json
import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assistant
import transcript_stability
from audio_sources import SAMPLE_RATE

# Where the scripted final lands, in chunks from the start of the stream, and the
# anchor it reports. Both well inside a 3 s file so the endpointer fires before
# the audio runs out and stt_endpoint_delay measures the endpoint rather than the
# flush at end of file.
CHUNK_MS = 100
FINAL_AT_CHUNK = 8
SPEECH_END_S = 0.70

# What one synthesis costs the double. Piper's real first chunk is 123-226 ms
# depending on its thread pinning; this only has to exceed the millisecond the
# CSV truncates to, so that tts_first_chunk carries a measurement rather than a
# zero that would read the same as the stage being absent.
TTS_SYNTH_S = 0.02

ITEMS = ("00004.wav", "00005.wav", "00012.wav")
UTTERANCES = {
    "00004": "what is the name of the campus tv station",
    "00005": "what did the bureau use to ensure safe travel",
    "00012": "what did his mother own when he was a child",
}


def write_silence(path, seconds):
    """A mono 16 kHz PCM file of the requested length.

    The scripted recognizer never looks at the samples, only at how many chunks
    of them realtime pacing has delivered.
    """
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))


class Builds(object):
    """A factory that records every engine it is asked to construct.

    The point of --asr-only is that the LLM and TTS are never built, which no
    assertion about the CSV can establish: a mode that built them and then
    declined to use them would write exactly the same rows.
    """

    def __init__(self, engine):
        self.engine = engine
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.engine


class ScriptedSTT(object):
    """Emits one final per item at a fixed position in the stream.

    The text is keyed off the file being read so that each item carries its own
    utterance through to the transcripts file, which is what the text-mode launch
    then reads back.
    """

    def __init__(self):
        self.item = None

    def warmup(self):
        pass

    def transcribe_stream(self, chunks):
        for position, _ in enumerate(chunks, start=1):
            if position == FINAL_AT_CHUNK:
                yield {"is_final": True,
                       "text": UTTERANCES[self.item],
                       "speech_end_s": SPEECH_END_S}


class ScriptedLLM(object):
    """Answers in one chunk, with the server-side stats a real reply carries."""

    def __init__(self):
        self.prompts = []

    def warmup(self):
        return True

    def placement(self):
        return None

    def cancel(self):
        pass

    def generate_stream(self, user_text):
        self.prompts.append(user_text)
        import time
        first_token_t = time.perf_counter()
        yield {"text": "An answer.", "ollama_stats": None, "cancelled": False,
               "first_token_t": first_token_t}
        yield {"text": "", "cancelled": False, "first_token_t": first_token_t,
               "ollama_stats": {"prompt_eval_count": 40,
                                "prompt_eval_duration_ns": 40_000_000,
                                "eval_count": 60,
                                "eval_duration_ns": 600_000_000,
                                "total_duration_ns": 700_000_000}}


class ScriptedTTS(object):
    """Returns a fixed length of audio, so a response WAV really appears."""

    sample_rate = SAMPLE_RATE

    def __init__(self):
        self.calls = []

    def warmup(self):
        pass

    def synthesize(self, text):
        import time
        time.sleep(TTS_SYNTH_S)
        self.calls.append(text)
        return b"\x00\x00" * int(SAMPLE_RATE * 0.25), SAMPLE_RATE


class Launch(object):
    """One run of the real main(), and what it left behind."""

    def __init__(self, rows, run_dir, builds, llm, tts):
        self.rows = rows
        self.run_dir = run_dir
        self.builds = builds
        self.llm = llm
        self.tts = tts

    @property
    def stages(self):
        return {row["stage"] for row in self.rows}

    @property
    def items(self):
        return {row["item"] for row in self.rows}

    def stage(self, item, name):
        for row in self.rows:
            if row["item"] == item and row["stage"] == name:
                return row
        raise AssertionError(f"no {name} row for {item}")

    def transcripts(self):
        path = os.path.join(self.run_dir, "transcripts.jsonl")
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def run_pipeline(work_dir, extra_argv, stt_item_order=None):
    """Drive assistant.main() with scripted engines and return what it wrote."""
    os.makedirs(work_dir, exist_ok=True)
    stt, llm, tts = ScriptedSTT(), ScriptedLLM(), ScriptedTTS()
    builds = {"VoskEngine": Builds(stt), "OllamaEngine": Builds(llm),
              "PiperEngine": Builds(tts)}

    # The recognizer needs to know which item it is on, and main() does not tell
    # an engine that. The item order is the sorted --audio order, which the
    # caller knows because it wrote the files.
    pending = list(stt_item_order or [])

    original_plan_items = assistant.plan_items

    def plan_items(*args, **kwargs):
        for item in original_plan_items(*args, **kwargs):
            if pending:
                stt.item = pending.pop(0)
            yield item

    patched = dict(builds)
    patched["plan_items"] = plan_items
    # Nothing about a GPU or the Ollama server's memory is under test, and both
    # poll whatever machine this runs on.
    patched["start_gpu_monitor"] = lambda: None
    patched["stop_gpu_monitor"] = lambda: None
    patched["start_llm_memory_monitor"] = lambda url, placement: None

    latency_csv = os.path.join(work_dir, "latency.csv")
    out_dir = os.path.join(work_dir, "out")
    saved = {name: getattr(assistant, name) for name in patched}
    argv = sys.argv
    try:
        for name, replacement in patched.items():
            setattr(assistant, name, replacement)
        sys.argv = ["assistant.py",
                    "--stt-engine", "vosk", "--tts-engine", "piper",
                    "--audio-chunk-ms", str(CHUNK_MS),
                    "--out-dir", out_dir,
                    "--latency-csv", latency_csv,
                    "--no-summary"] + list(extra_argv)
        assert assistant.main() == 0
    finally:
        sys.argv = argv
        for name, original in saved.items():
            setattr(assistant, name, original)

    with open(latency_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["extra"] = json.loads(row["extra_json"])
    run_dir = os.path.join(out_dir, sorted(os.listdir(out_dir))[-1])
    return Launch(rows, run_dir, {k: v.count for k, v in builds.items()}, llm, tts)


class RunModeTest(unittest.TestCase):
    """Three chained launches: an ASR pass, a text run off it, and a full run."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="run-modes-test-")
        audio_dir = os.path.join(cls._tmp.name, "audio")
        os.makedirs(audio_dir)
        for name in ITEMS:
            write_silence(os.path.join(audio_dir, name), 3.0)
        order = [os.path.splitext(name)[0] for name in sorted(ITEMS)]

        cls.asr = run_pipeline(
            os.path.join(cls._tmp.name, "asr"),
            ["--asr-only", "--audio", audio_dir], order)

        # No conversion step: the pass's own transcripts file is the input.
        cls.text = run_pipeline(
            os.path.join(cls._tmp.name, "text"),
            ["--input-mode", "text",
             "--transcripts", os.path.join(cls.asr.run_dir, "transcripts.jsonl")])

        cls.full = run_pipeline(
            os.path.join(cls._tmp.name, "full"),
            ["--audio", audio_dir], order)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ---------- 1. the ASR-only pass ----------

    def test_asr_only_writes_the_two_recognizer_stages_and_nothing_else(self):
        self.assertEqual({"stt", "stt_endpoint_delay"}, self.asr.stages)

    def test_asr_only_builds_no_llm_and_no_tts(self):
        self.assertEqual(1, self.asr.builds["VoskEngine"])
        self.assertEqual(0, self.asr.builds["OllamaEngine"],
                         "a resident Ollama model holds VRAM for the length of "
                         "the pass even if it is never asked to generate")
        self.assertEqual(0, self.asr.builds["PiperEngine"],
                         "a loaded Piper holds its ONNX Runtime thread pool "
                         "whether or not it is asked to speak")
        self.assertEqual([], self.asr.llm.prompts)
        self.assertEqual([], self.asr.tts.calls)

    def test_asr_only_keeps_the_speech_end_anchor_visible(self):
        for item in self.asr.items:
            extra = self.asr.stage(item, "stt")["extra"]
            self.assertIn("speech_end_source", extra)
            self.assertEqual(1, extra["endpoint_fire_count"])

    def test_asr_only_writes_transcripts_with_no_answer(self):
        records = self.asr.transcripts()
        self.assertEqual([name for name in sorted(ITEMS)],
                         [r["filename"] for r in records])
        for record in records:
            self.assertTrue(record["stt_text"])
            self.assertEqual("", record["llm_text"])

    # ---------- 2. the stages text mode must not write ----------

    def test_text_mode_writes_no_recognizer_or_ttfa_rows(self):
        for stage in ("stt", "stt_endpoint_delay", "ttfa"):
            self.assertNotIn(stage, self.text.stages,
                             f"{stage} has no meaning without audio, and "
                             f"aggregate_logs.py cannot tell a zero from a "
                             f"measurement")

    def test_text_mode_still_writes_every_stage_it_can_measure(self):
        self.assertEqual(
            {"llm_prompt_eval", "llm_ttft", "llm_ttfc", "llm_first_chunk_fill",
             "tts_first_chunk", "llm_eval", "tts_total", "e2e_response_ready"},
            self.text.stages)

    def test_text_mode_omits_the_extra_json_keys_read_off_the_audio(self):
        for row in self.text.rows:
            for key in ("input_duration_ms", "stt_rtf"):
                self.assertNotIn(key, row["extra"],
                                 f"{key} on a {row['stage']} row would be "
                                 f"averaged as though it had been measured")

    def test_text_mode_blanks_the_columns_describing_the_audio(self):
        for row in self.text.rows:
            self.assertEqual("text", row["input_mode"])
            self.assertEqual("", row["audio_pacing"])
            self.assertEqual("", row["utterance_trigger"],
                             "nothing released this utterance; it arrived final")

    def test_text_mode_builds_no_recognizer(self):
        self.assertEqual(0, self.text.builds["VoskEngine"])

    # ---------- 3. the TTS has to keep running ----------

    def test_text_mode_synthesizes_as_file_mode_does(self):
        # tts_first_chunk is a term of the reconstructed TTFA, and Piper's first
        # chunk only differs between thread settings while the LLM generates
        # underneath it, so a text-mode run with no TTS would measure a machine
        # that never runs.
        self.assertEqual(len(ITEMS), len(self.text.tts.calls))
        for item in self.text.items:
            self.assertGreater(
                int(self.text.stage(item, "tts_first_chunk")["duration_ms"]), 0)
            self.assertGreater(
                self.text.stage(item, "e2e_response_ready")["extra"]["output_duration_ms"], 0)

    # ---------- 4. item identity across the modes ----------

    def test_item_names_agree_across_every_mode(self):
        expected = {os.path.splitext(name)[0] for name in ITEMS}
        self.assertEqual(expected, self.full.items)
        self.assertEqual(expected, self.asr.items)
        self.assertEqual(expected, self.text.items,
                         "task 07 joins the modes on this column and "
                         "evaluation/comparison.py pairs items on the filename")

    def test_text_mode_carries_the_transcript_identity_through(self):
        self.assertEqual([r["filename"] for r in self.asr.transcripts()],
                         [r["filename"] for r in self.text.transcripts()])
        # Answered from the frozen text, not re-recognized.
        self.assertEqual([r["stt_text"] for r in self.asr.transcripts()],
                         self.text.llm.prompts)

    def test_e2e_response_ready_drops_the_speech_it_no_longer_waits_through(self):
        # Both modes emit the stage; in text mode it runs from the request, so it
        # no longer contains the recording. input_mode keeps the two unpooled.
        for item in self.text.items:
            from_request = int(self.text.stage(item, "e2e_response_ready")["duration_ms"])
            whole_item = int(self.full.stage(item, "e2e_response_ready")["duration_ms"])
            self.assertLess(from_request, whole_item)

    # ---------- 5. a transcripts file that cannot name its items ----------

    def test_a_transcripts_record_without_a_filename_is_rejected_at_load(self):
        for payload in ('{"stt_text": "no filename here"}',
                        '{"filename": "", "stt_text": "empty filename"}',
                        '{"filename": "   ", "stt_text": "blank filename"}'):
            path = os.path.join(self._tmp.name, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
            with self.assertRaises(ValueError):
                assistant.load_transcript_items(path)

    def test_an_empty_transcripts_file_is_rejected(self):
        path = os.path.join(self._tmp.name, "empty.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n")
        with self.assertRaises(ValueError):
            assistant.load_transcript_items(path)

    def test_the_transcripts_file_a_run_writes_loads_back_unchanged(self):
        records = assistant.load_transcript_items(
            os.path.join(self.asr.run_dir, "transcripts.jsonl"))
        self.assertEqual([r["filename"] for r in self.asr.transcripts()],
                         [r["filename"] for r in records])


class TranscriptStabilityTest(unittest.TestCase):
    """The agreement figures, on transcripts a run actually wrote."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="stability-test-")
        self.base = [
            {"filename": "00004.wav", "stt_text": "what is the name of the campus tv station"},
            {"filename": "00005.wav", "stt_text": "what did the bureau use to ensure safe travel"},
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def write_pass(self, name, records, fires=None):
        run_dir = os.path.join(self._tmp.name, name)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "transcripts.jsonl"), "w",
                  encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        if fires is not None:
            path = os.path.join(run_dir, "latency_log_test.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                out = csv.writer(fh)
                out.writerow(["item", "stage", "extra_json"])
                for item, count in fires.items():
                    out.writerow([item, "stt",
                                  json.dumps({"endpoint_fire_count": count})])
        return transcript_stability.load_pass(name, run_dir)

    def test_two_identical_passes_agree_on_everything(self):
        left = self.write_pass("A", self.base, {"00004": 1, "00005": 1})
        right = self.write_pass("B", self.base, {"00004": 1, "00005": 1})
        report, differing = transcript_stability.format_report([left, right])
        self.assertEqual(0, differing)
        self.assertIn("identical across all passes : 2 of 2", report)

    def test_one_substituted_word_is_counted_and_rated(self):
        changed = [dict(self.base[0], stt_text="what is the name of the campus tv channel"),
                   self.base[1]]
        left = self.write_pass("A", self.base, {"00004": 1, "00005": 1})
        right = self.write_pass("B", changed, {"00004": 1, "00005": 1})
        report, differing = transcript_stability.format_report([left, right])
        self.assertEqual(1, differing)
        # One substitution in a nine-word item: 1/9 over the differing item, and
        # 1/18 over both. The gap between the two columns is the whole point of
        # reporting both -- the corpus figure stays small while an item flipped.
        self.assertIn("0.1111", report)
        self.assertIn("0.0556", report)

    def test_an_item_only_one_pass_recognized_is_reported_apart(self):
        left = self.write_pass("A", self.base, {"00004": 1, "00005": 1})
        right = self.write_pass("B", self.base[:1], {"00004": 1})
        report, differing = transcript_stability.format_report([left, right])
        self.assertEqual(0, differing, "the item they share is identical")
        self.assertIn("present in some but not all: 1", report)

    def test_a_differing_endpoint_fire_count_is_counted(self):
        left = self.write_pass("A", self.base, {"00004": 1, "00005": 1})
        right = self.write_pass("B", self.base, {"00004": 2, "00005": 1})
        report, _ = transcript_stability.format_report([left, right])
        self.assertIn("differing                   : 1 of 2", report)
        self.assertIn("fires 1/2", report)

    def test_a_pass_with_no_latency_csv_is_still_compared_on_its_text(self):
        left = self.write_pass("A", self.base)
        right = self.write_pass("B", self.base)
        report, differing = transcript_stability.format_report([left, right])
        self.assertEqual(0, differing)
        self.assertIn("endpoint_fire_count: not recorded by every pass", report)

    def test_the_error_rate_is_over_the_corpus_not_the_mean_of_items(self):
        # One error in a one-word item and none in a nine-word one is 1/10, not
        # the 0.5 a mean of per-item rates would give.
        self.assertAlmostEqual(
            1 / 10,
            transcript_stability.error_rate(
                [("a", "b"), ("one two three four five six seven eight nine",
                              "one two three four five six seven eight nine")]))


if __name__ == "__main__":
    unittest.main()
