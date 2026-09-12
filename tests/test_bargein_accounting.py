#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What a double endpoint fire does to the latency CSV.

An endpointer that fires before the speaker has finished occurs in 1.1% of the
archived file-mode items, which is too rare to wait for and too common to
ignore. This drives the real pipeline -- assistant.main(), its three worker
threads, its CSV writer -- with scripted engines in place of Vosk, Ollama and
Piper, so the double fire happens on demand.

Scripted, not mocked: the STT yields its finals at chosen positions in the
stream, the LLM takes a chosen time to reach each chunk, and the TTS sleeps a
chosen time and hands back a chosen length of audio. Everything between them is
the code under test, FileAudioSource's realtime pacing included, so the
wall-clock instants the metrics are differences of are real ones.

Two questions it answers, both of which can only be settled by running it:

  - which of the two generations `ttfa` measures to, under a metadata anchor
    that is the same on every fire and under the engine's own word timings,
    where the anchor moves with each fire;
  - whether `tts_total` and `output_duration_ms` cover the same set of WAVs.

Run:  python -m unittest discover -s tests -v
"""

import csv
import json
import os
import sys
import tempfile
import time
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assistant
from audio_sources import SAMPLE_RATE, FileAudioSource

# The scripted timings. Chosen so that every instant the assertions compare is
# further apart than any plausible scheduling jitter: the two candidate first
# chunks are a whole generation apart (800 ms), and the two candidate word-timing
# anchors 750 ms.
CHUNK_MS = 100                  # audio chunk handed to the STT
LLM_FIRST_TOKEN_S = 0.12        # request sent -> first token
LLM_FIRST_CHUNK_S = 0.20        # request sent -> first synthesizable chunk
LLM_CHUNK_GAP_S = 0.20          # chunk -> chunk
LLM_CHUNKS_PER_ANSWER = 6
TTS_SYNTH_S = 0.10              # wall-clock cost of synthesizing one chunk
TTS_AUDIO_S = 0.50              # audio produced by one chunk

# Where the finals land, in chunks from the start of the stream.
FIRE_1_CHUNK = 8                # 0.8 s
FIRE_2_CHUNK = 16               # 1.6 s

# The anchors. METADATA_SPEECH_END_S is what a metadata.csv supplies and is the
# same on every fire; the WORD_TIMING pair is what an engine reports, one per
# fire, which is the case the anchor moves in.
METADATA_SPEECH_END_S = 0.70
WORD_TIMING_SPEECH_END_S = (0.70, 1.45)

# perf_counter is monotonic and the pipeline reports whole milliseconds, so the
# only slack needed is the truncation plus the few microseconds between the
# instant the scripted engine records and the instant the worker records.
TOLERANCE_MS = 5

# The decomposition ttfa = stt_endpoint_delay + llm_ttfc + tts_first_chunk does
# not close exactly: the queue handoffs on either side of the LLM request sit
# inside ttfa and inside none of the three terms. Over three repeats here, +2 to
# +3 ms on a single fire and +7 to +9 ms on a double. The real pipeline is
# cheaper at it -- Vosk, gemma3:1b and Piper over 00004 and 00005 give +3 and
# +2 ms -- so the bound is set by this harness, not by production. Either way it
# is a handoff; a ttfa taken from the discarded generation would be off by the
# 800 ms between the two candidate chunks.
HANDOFF_TOLERANCE_MS = 30


def write_silence(path, seconds):
    """A mono 16 kHz PCM file of the requested length, which is all the
    scripted STT needs: it never looks at the samples, only at how many chunks
    of them realtime pacing has delivered."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))


class RecordingFileAudioSource(FileAudioSource):
    """FileAudioSource that remembers every anchor it converted.

    speech_end_t() is where an offset in the audio becomes a wall-clock instant,
    and the value of the last call is the one every speech-end anchored metric
    on the row is measured from. Recording it here is the only way for the test
    to name that instant without recomputing it, which would be asserting the
    code against itself.
    """

    instances = []

    def __init__(self, wav_path, *args, **kwargs):
        super().__init__(wav_path, *args, **kwargs)
        self.path = str(wav_path)
        self.anchors = []
        RecordingFileAudioSource.instances.append(self)

    def speech_end_t(self, speech_end_s):
        anchor = super().speech_end_t(speech_end_s)
        self.anchors.append((speech_end_s, anchor))
        return anchor


class ScriptedSTT:
    """Yields finals at fixed positions in the stream, one script per item.

    The script is picked by the input file's name rather than by call order, so
    the test does not depend on the order main() happens to sort --audio into.
    """

    def __init__(self, scripts):
        self.scripts = scripts

    def warmup(self):
        pass

    def _script_for_current_item(self):
        path = RecordingFileAudioSource.instances[-1].path
        for stem, script in self.scripts.items():
            if stem in os.path.basename(path):
                return script
        raise AssertionError(f"no script for {path}")

    def transcribe_stream(self, chunks):
        script = self._script_for_current_item()
        for position, _ in enumerate(chunks, start=1):
            for at_chunk, text, speech_end_s in script:
                if at_chunk == position:
                    yield {"is_final": True, "text": text,
                           "speech_end_s": speech_end_s}


class ScriptedLLM:
    """Answers in LLM_CHUNKS_PER_ANSWER chunks, tagged with the generation.

    The tag is what lets the assertions say which generation a WAV, a synthesis
    call or a first-chunk instant belonged to. Follows OllamaEngine's contract:
    cancel() closes the stream, and a cancelled generation yields a final dict
    with no server-side stats rather than raising.
    """

    def __init__(self):
        self.generation = 0
        self._cancelled = False

    def warmup(self):
        return True

    def placement(self):
        return None

    def cancel(self):
        self._cancelled = True

    def generate_stream(self, user_text):
        self._cancelled = False
        self.generation += 1
        tag = f"a{self.generation}"

        time.sleep(LLM_FIRST_TOKEN_S)
        first_token_t = time.perf_counter()
        time.sleep(LLM_FIRST_CHUNK_S - LLM_FIRST_TOKEN_S)

        for index in range(LLM_CHUNKS_PER_ANSWER):
            if self._cancelled:
                yield {"text": "", "ollama_stats": None, "cancelled": True,
                       "first_token_t": first_token_t}
                return
            yield {"text": f"{tag} chunk {index}.", "ollama_stats": None,
                   "cancelled": False, "first_token_t": first_token_t}
            time.sleep(LLM_CHUNK_GAP_S)

        yield {"text": "", "cancelled": False, "first_token_t": first_token_t,
               "ollama_stats": {"prompt_eval_count": 40,
                                "prompt_eval_duration_ns": 40_000_000,
                                "eval_count": 60,
                                "eval_duration_ns": 600_000_000,
                                "total_duration_ns": 700_000_000}}


class ScriptedTTS:
    """Synthesizes in fixed time and returns a fixed length of audio.

    Both constants matter: TTS_SYNTH_S is what tts_total accumulates and
    TTS_AUDIO_S is what output_duration_ms measures, so a disagreement about
    which chunks belong to the surviving answer shows up as a different
    multiple of each.
    """

    sample_rate = SAMPLE_RATE

    def __init__(self):
        self.calls = []

    def warmup(self):
        pass

    def synthesize(self, text):
        time.sleep(TTS_SYNTH_S)
        self.calls.append({"tag": text.split()[0],
                           "text": text,
                           "done_t": time.perf_counter()})
        return b"\x00\x00" * int(SAMPLE_RATE * TTS_AUDIO_S), SAMPLE_RATE

    def tags(self):
        return [call["tag"] for call in self.calls]

    def first_call_tagged(self, tag):
        for call in self.calls:
            if call["tag"] == tag:
                return call
        raise AssertionError(f"the TTS was never given a {tag} chunk")


def run_pipeline(work_dir, with_metadata):
    """Run one launch of the real pipeline over three scripted items.

    The items are a double fire, a single fire, and a recording nothing was
    recognized in -- the last so that the fire count is checked on a row where
    there is no answer to hang it on.
    """
    audio_dir = os.path.join(work_dir, "audio")
    os.makedirs(audio_dir)
    write_silence(os.path.join(audio_dir, "01_double.wav"), 3.0)
    write_silence(os.path.join(audio_dir, "02_single.wav"), 3.0)
    write_silence(os.path.join(audio_dir, "03_silent.wav"), 1.0)

    if with_metadata:
        with open(os.path.join(audio_dir, "metadata.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            out = csv.writer(fh)
            out.writerow(["filename", "question", "speech_end_ms"])
            for name in ("01_double.wav", "02_single.wav", "03_silent.wav"):
                out.writerow([name, "what did he inherit",
                              int(METADATA_SPEECH_END_S * 1000)])

    # With a metadata.csv the anchor is fixed, so the engine's own reading is
    # never read; without one it is the only anchor there is, and it moves.
    first_end, second_end = WORD_TIMING_SPEECH_END_S
    scripts = {
        "01_double": [(FIRE_1_CHUNK, "what did he", first_end),
                      (FIRE_2_CHUNK, "inherit", second_end)],
        "02_single": [(FIRE_1_CHUNK, "what did he inherit", first_end)],
        "03_silent": [],
    }

    stt, llm, tts = ScriptedSTT(scripts), ScriptedLLM(), ScriptedTTS()
    RecordingFileAudioSource.instances = []

    latency_csv = os.path.join(work_dir, "latency.csv")
    patched = {
        "VoskEngine": lambda *a, **k: stt,
        "OllamaEngine": lambda *a, **k: llm,
        "PiperEngine": lambda *a, **k: tts,
        "FileAudioSource": RecordingFileAudioSource,
        # Nothing about a GPU or the Ollama server's memory is under test here,
        # and both poll the machine the test happens to run on.
        "start_gpu_monitor": lambda: None,
        "stop_gpu_monitor": lambda: None,
        "start_llm_memory_monitor": lambda url, placement: None,
    }
    saved = {name: getattr(assistant, name) for name in patched}
    argv = sys.argv
    try:
        for name, replacement in patched.items():
            setattr(assistant, name, replacement)
        sys.argv = [
            "assistant.py",
            "--stt-engine", "vosk", "--tts-engine", "piper",
            "--input-mode", "file", "--audio-pacing", "realtime",
            "--file-realtime-trigger", "endpoint",
            "--audio-chunk-ms", str(CHUNK_MS),
            "--audio", audio_dir,
            "--out-dir", os.path.join(work_dir, "out"),
            "--latency-csv", latency_csv,
            "--no-summary",
        ]
        assert assistant.main() == 0
    finally:
        sys.argv = argv
        for name, original in saved.items():
            setattr(assistant, name, original)

    with open(latency_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["extra"] = json.loads(row["extra_json"])
    return rows, tts, RecordingFileAudioSource.instances


class BargeInAccountingTest(unittest.TestCase):
    """One launch per anchor, shared by every assertion below."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="bargein-test-")
        cls.metadata_run = run_pipeline(
            os.path.join(cls._tmp.name, "metadata"), with_metadata=True)
        cls.word_timing_run = run_pipeline(
            os.path.join(cls._tmp.name, "word_timings"), with_metadata=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def stage(rows, item, name):
        for row in rows:
            if row["item"] == item and row["stage"] == name:
                return row
        raise AssertionError(f"no {name} row for {item}")

    def assert_double_fire_happened(self, tts):
        """The scripted timings have to actually produce a barge-in, or every
        assertion below would pass against a pipeline that never saw one."""
        self.assertIn("a1", tts.tags(), "the discarded generation synthesized "
                                        "nothing, so there was no double fire "
                                        "to measure")
        self.assertIn("a2", tts.tags())

    # ---------- 1. the fire count ----------

    def test_fire_count_on_every_stt_row(self):
        for label, (rows, _, _) in (("metadata", self.metadata_run),
                                    ("word timings", self.word_timing_run)):
            stt_rows = [r for r in rows if r["stage"] == "stt"]
            self.assertEqual(len(stt_rows), 3, label)
            for row in stt_rows:
                self.assertIn("endpoint_fire_count", row["extra"],
                              f"{label}: missing on {row['item']}")
            counts = {r["item"]: r["extra"]["endpoint_fire_count"]
                      for r in stt_rows}
            self.assertEqual(counts, {"01_double": 2, "02_single": 1,
                                      "03_silent": 0}, label)

    # ---------- 2. tts_total and output_duration_ms ----------

    def test_both_cover_the_surviving_answer_only(self):
        rows, tts, _ = self.metadata_run
        self.assert_double_fire_happened(tts)

        surviving = [c for c in tts.calls if c["tag"] == "a2"]
        discarded = [c for c in tts.calls if c["tag"] == "a1"]
        self.assertTrue(discarded, "nothing was thrown away")

        e2e = self.stage(rows, "01_double", "e2e_response_ready")["extra"]
        tts_total_ms = int(self.stage(rows, "01_double", "tts_total")["duration_ms"])

        # Same chunk count reached through two independent constants: the audio
        # each chunk produced, and the time each took to synthesize.
        self.assertEqual(e2e["output_duration_ms"],
                         round(len(surviving) * TTS_AUDIO_S * 1000))
        self.assertEqual(e2e["discarded_output_duration_ms"],
                         round(len(discarded) * TTS_AUDIO_S * 1000))
        self.assertAlmostEqual(tts_total_ms / 1000.0,
                               len(surviving) * TTS_SYNTH_S,
                               delta=0.2 * len(surviving))

        # Both WAVs are still on disk -- the count is what identifies the item.
        self.assertEqual(e2e["output_wav_count"], 2)
        # And the path on the row is the answer the rest of the row describes.
        self.assertTrue(e2e["output_wav"].endswith("_r2.wav"), e2e["output_wav"])

    def test_a_single_fire_discards_nothing(self):
        rows, tts, _ = self.metadata_run
        e2e = self.stage(rows, "02_single", "e2e_response_ready")["extra"]
        self.assertEqual(e2e["discarded_output_duration_ms"], 0)
        self.assertEqual(e2e["output_wav_count"], 1)
        self.assertEqual(e2e["output_duration_ms"],
                         round(LLM_CHUNKS_PER_ANSWER * TTS_AUDIO_S * 1000))

    # ---------- 3. which generation ttfa measures to ----------

    def ttfa_candidates(self, run):
        """The two instants ttfa could plausibly have recorded, in ms from the
        anchor the pipeline itself converted."""
        _, tts, sources = run
        self.assert_double_fire_happened(tts)
        source = next(s for s in sources if "01_double" in s.path)
        _, anchor = source.anchors[-1]
        return {tag: (tts.first_call_tagged(tag)["done_t"] - anchor) * 1000
                for tag in ("a1", "a2")}

    def test_ttfa_measures_to_the_surviving_generation_metadata_anchor(self):
        rows, _, _ = self.metadata_run
        candidates = self.ttfa_candidates(self.metadata_run)
        ttfa_ms = int(self.stage(rows, "01_double", "ttfa")["duration_ms"])

        self.assertEqual(
            self.stage(rows, "01_double", "stt")["extra"]["speech_end_source"],
            "metadata")
        self.assertAlmostEqual(ttfa_ms, candidates["a2"], delta=TOLERANCE_MS)
        # Stated as well as implied: the discarded generation's first chunk is
        # nowhere near, so this is not a tolerance that happens to cover both.
        self.assertGreater(abs(ttfa_ms - candidates["a1"]), 500)

    def test_ttfa_measures_to_the_surviving_generation_word_timings(self):
        rows, _, sources = self.word_timing_run
        candidates = self.ttfa_candidates(self.word_timing_run)
        ttfa_ms = int(self.stage(rows, "01_double", "ttfa")["duration_ms"])

        self.assertEqual(
            self.stage(rows, "01_double", "stt")["extra"]["speech_end_source"],
            "stt_word_timings")
        # The anchor moved: the row is measured from the second fire's reading,
        # not the first, which is the difference from the metadata case.
        source = next(s for s in sources if "01_double" in s.path)
        self.assertEqual([offset for offset, _ in source.anchors],
                         list(WORD_TIMING_SPEECH_END_S))
        self.assertAlmostEqual(ttfa_ms, candidates["a2"], delta=TOLERANCE_MS)
        self.assertGreater(abs(ttfa_ms - candidates["a1"]), 500)

    def residual_ms(self, rows, item):
        """ttfa minus the three terms it is reconstructed from."""
        terms = {name: int(self.stage(rows, item, name)["duration_ms"])
                 for name in ("ttfa", "stt_endpoint_delay", "llm_ttfc",
                              "tts_first_chunk")}
        return terms["ttfa"] - (terms["stt_endpoint_delay"]
                                + terms["llm_ttfc"]
                                + terms["tts_first_chunk"]), terms

    def test_the_decomposition_survives_a_double_fire(self):
        """ttfa and stt_endpoint_delay have to describe the same fire.

        They both anchor on speech_end_t and stt_endpoint_delay ends at the
        last fire, so if ttfa ended at the discarded generation's chunk the
        residual would go sharply negative -- by the length of a generation,
        not by a handoff.
        """
        for label, run in (("metadata", self.metadata_run),
                           ("word timings", self.word_timing_run)):
            rows = run[0]
            double, double_terms = self.residual_ms(rows, "01_double")
            single, single_terms = self.residual_ms(rows, "02_single")
            self.assertGreaterEqual(double, 0, f"{label}: {double_terms}")
            self.assertLessEqual(double, HANDOFF_TOLERANCE_MS,
                                 f"{label}: {double_terms}")
            self.assertLessEqual(abs(double - single), HANDOFF_TOLERANCE_MS,
                                 f"{label}: double {double_terms} "
                                 f"vs single {single_terms}")

    def test_the_two_anchors_disagree_by_the_word_timing_shift(self):
        """A guard on the test itself: if both runs anchored on the same
        instant, the word-timing case would prove nothing."""
        metadata_ttfa = int(self.stage(self.metadata_run[0], "01_double",
                                       "ttfa")["duration_ms"])
        word_ttfa = int(self.stage(self.word_timing_run[0], "01_double",
                                   "ttfa")["duration_ms"])
        expected_shift = (WORD_TIMING_SPEECH_END_S[1] - METADATA_SPEECH_END_S) * 1000
        self.assertAlmostEqual(metadata_ttfa - word_ttfa, expected_shift,
                               delta=50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
