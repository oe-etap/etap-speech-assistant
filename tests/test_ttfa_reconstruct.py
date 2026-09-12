#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What the TTFA reconstruction must keep true when someone edits it later.

The reconstruction replaces a directly measured number with a computed one,
so the ways it can be wrong are all quiet. Four of them are pinned here
because nothing downstream would notice:

  - **summing the stage medians instead of the per-item sums.** It is the
    obvious simplification, it looks like a cleanup, and on
    rank-correlated stages it even gives nearly the right answer. The
    fixture below is deliberately anti-correlated (Spearman exactly -1), so
    the two routes give 1042 ms and 46 ms and the mistake cannot pass.
  - **keeping the warm-up item.** One item per launch is bimodal: the model
    load lands inside the measured `ttfa` and inside none of the three terms
    that reconstruct it. Its median is the same as everyone else's, which is
    why it has to be excluded by rule rather than spotted -- what one
    retained warm-up item destroys is the tail.
  - **a percentile that is not `run_statistics.percentile`.** A truncated
    index agrees with it almost everywhere, which is what makes the
    disagreement dangerous: on a short heavy-tailed sample it reads 50000 ms
    where the interpolating estimator reads 2595 ms.
  - **letting an unmatched item through.** The join is what says the two
    arms describe the same recordings; an item silently dropped from it
    still leaves a plausible-looking median behind.

Fixtures are written as latency CSVs rather than constructed in memory, so
the reader under test is the one the campaign will actually use.

Run:  python -m unittest discover -s tests -v
"""

import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import campaign_report as creport          # noqa: E402
import run_statistics as rstat             # noqa: E402
import ttfa_reconstruct as recon           # noqa: E402

HEADER = ["ts_iso", "mode", "stt_engine", "tts_engine", "input_mode", "audio_pacing",
          "utterance_trigger", "cell_id", "launch_id", "item", "stage", "duration_ms",
          "cpu_percent", "ram_percent", "rss_mb", "gpu_util_percent", "gpu_mem_used_mb",
          "gpu_mem_total_mb", "gpu_name", "extra_json"]


def write_csv(path, cell, launch, input_mode, rows):
    """One latency CSV in the shape `assistant.py` writes, from (item, stage, ms)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for item, stage, duration in rows:
            writer.writerow({
                "ts_iso": "2026-09-12T05:00:00", "mode": "file",
                "stt_engine": "vosk", "tts_engine": "piper",
                "input_mode": input_mode,
                "audio_pacing": "realtime" if input_mode == "file" else "",
                "utterance_trigger": "endpoint" if input_mode == "file" else "",
                "cell_id": cell, "launch_id": launch, "item": item, "stage": stage,
                "duration_ms": duration, "cpu_percent": 10, "ram_percent": 10,
                "rss_mb": 100, "gpu_util_percent": "", "gpu_mem_used_mb": "",
                "gpu_mem_total_mb": "", "gpu_name": "", "extra_json": "{}",
            })
    return path


def asr_csv(root, launch, endpoint_by_item, cell="asr-pass"):
    rows = []
    for item, delay in endpoint_by_item:
        rows.append((item, "stt", 300.0))
        rows.append((item, "stt_endpoint_delay", delay))
    return write_csv(os.path.join(root, "asr", launch, "latency_log_asr.csv"),
                     cell, launch, "file", rows)


def text_csv(root, launch, per_item, cell="17-gemma3", ttft_offset=-100.0):
    """per_item: [(item, llm_ttfc, tts_first_chunk), ...] in the order they ran."""
    rows = []
    for item, ttfc, tts in per_item:
        rows.append((item, "llm_ttft", ttfc + ttft_offset))
        rows.append((item, "llm_ttfc", ttfc))
        rows.append((item, "tts_first_chunk", tts))
        rows.append((item, "tts_total", 900.0))
    return write_csv(os.path.join(root, "text", cell, launch, "latency_log_text.csv"),
                     cell, launch, "text", rows)


def reconstruct_from(root, asr_launch, text_launch, offset_ms=0.0):
    """Read both arms back off disk and join them, as the CLI does."""
    asr_arm = recon.load_arm("asr", recon.Path(os.path.join(root, "asr")))
    text_arm = recon.load_arm("text", recon.Path(os.path.join(root, "text")))
    asr_run, problem = recon.pick_canonical(asr_arm, asr_launch)
    assert asr_run is not None, problem
    text_run = next(run for run in text_arm.runs if run.launch == text_launch)
    return recon.reconstruct(asr_run, text_run, offset_ms)


# The stages are ranked in exactly opposite order across the five timed items
# (Spearman -1), which is the case the per-item sum exists for: the item with
# the fastest endpoint is the one with the slowest first chunk, so neither
# stage's median item is the median of the sums. Sums per item: 1042, 1042,
# 46, 1042, 1042 -> median 1042. Sum of the stage medians: 3 + 3 + 40 = 46.
ANTI_CORRELATED_ENDPOINT = [("warm", 5.0), ("i1", 1.0), ("i2", 2.0),
                            ("i3", 3.0), ("i4", 1000.0), ("i5", 1001.0)]
ANTI_CORRELATED_TEXT = [("warm", 50000.0, 40.0), ("i1", 1001.0, 40.0),
                        ("i2", 1000.0, 40.0), ("i3", 3.0, 40.0),
                        ("i4", 2.0, 40.0), ("i5", 1.0, 40.0)]
PER_ITEM_MEDIAN_MS = 1042.0
SUM_OF_MEDIANS_MS = 46.0


class AntiCorrelatedStagesTest(unittest.TestCase):
    """The per-item sum, against the sum-of-medians shortcut that replaces it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttfa-recon-anti-")
        self.addCleanup(_rmtree, self.tmp)
        asr_csv(self.tmp, "20260912_050000", ANTI_CORRELATED_ENDPOINT)
        text_csv(self.tmp, "20260912_051000", ANTI_CORRELATED_TEXT)
        self.rec = reconstruct_from(self.tmp, "20260912_050000", "20260912_051000")

    def test_fixture_really_is_anti_correlated(self):
        """Guards the guard: a later edit to the fixture must not soften it."""
        items = self.rec.join.matched
        endpoint = [self.rec.parts[i]["stt_endpoint_delay"] for i in items]
        ttfc = [self.rec.parts[i]["llm_ttfc"] for i in items]
        self.assertEqual(_rank(endpoint), list(reversed(_rank(ttfc))))

    def test_each_item_is_summed_on_its_own(self):
        values = self.rec.values
        self.assertEqual(sorted(values.values()), [46.0, 1042.0, 1042.0, 1042.0, 1042.0])

    def test_median_is_of_the_sums_not_a_sum_of_medians(self):
        median = recon.describe(list(self.rec.values.values()))["p50"]
        self.assertEqual(median, PER_ITEM_MEDIAN_MS)
        # What collapsing each stage to its median first would have produced.
        self.assertNotEqual(median, SUM_OF_MEDIANS_MS)

    def test_sum_of_medians_is_the_wrong_answer_here(self):
        """Spells out the shortcut, so the two numbers are visible side by side."""
        shortcut = sum(
            rstat.percentile([self.rec.parts[i][stage] for i in self.rec.join.matched], 0.5)
            for stage in recon.RECONSTRUCTION_STAGES)
        self.assertEqual(shortcut, SUM_OF_MEDIANS_MS)
        self.assertEqual(recon.describe(list(self.rec.values.values()))["p50"],
                         PER_ITEM_MEDIAN_MS)

    def test_reported_median_comes_from_the_per_item_sums(self):
        """The same assertion through the CLI, where a report-level shortcut would hide."""
        report = _run_cli(["--asr-arm", os.path.join(self.tmp, "asr"),
                           "--text-arm", os.path.join(self.tmp, "text"),
                           "--offset-ms", "0"])
        self.assertIn("median=1042 ms", report)
        self.assertNotIn("median=46 ms", report)


class WarmupExclusionTest(unittest.TestCase):
    """One bimodal first item per launch, excluded the way every sibling does."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttfa-recon-warm-")
        self.addCleanup(_rmtree, self.tmp)
        asr_csv(self.tmp, "20260912_050000", ANTI_CORRELATED_ENDPOINT)
        text_csv(self.tmp, "20260912_051000", ANTI_CORRELATED_TEXT)
        self.rec = reconstruct_from(self.tmp, "20260912_050000", "20260912_051000")

    def test_first_item_of_each_launch_is_not_reconstructed(self):
        self.assertNotIn("warm", self.rec.values)
        self.assertNotIn("warm", self.rec.join.asr_only)
        self.assertNotIn("warm", self.rec.join.text_only)

    def test_the_warmup_value_would_have_wrecked_the_tail(self):
        """Not a hypothetical: this fixture's warm-up item carries a 50 s load.

        The median is the stratum's least affected figure -- over the 144
        archived first items it is the same +5 ms as everywhere else -- so
        what one retained warm-up item destroys is the tail, and the tail is
        what a latency target is read off.
        """
        kept = sorted(self.rec.values.values())
        with_warmup = sorted(kept + [5.0 + 50000.0 + 40.0])
        self.assertEqual(rstat.percentile(kept, 0.95), PER_ITEM_MEDIAN_MS)
        self.assertGreater(rstat.percentile(with_warmup, 0.95), 30000.0)
        self.assertEqual(max(kept), PER_ITEM_MEDIAN_MS)

    def test_exclusion_is_the_shared_one(self):
        """Imported, not redefined, so the four scripts cannot drift apart."""
        self.assertEqual(recon.creport.WARMUP_ITEMS, creport.WARMUP_ITEMS)
        run = creport.read_latency_csv(recon.Path(
            os.path.join(self.tmp, "asr", "20260912_050000", "latency_log_asr.csv")))
        self.assertEqual(run.warmup_items, ["warm"])
        self.assertNotIn("warm", recon.timed_items(run))


class JoinIntegrityTest(unittest.TestCase):
    """Zero unmatched items, and a loud failure when that is not true."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttfa-recon-join-")
        self.addCleanup(_rmtree, self.tmp)

    def test_matching_arms_join_cleanly(self):
        asr_csv(self.tmp, "20260912_050000", ANTI_CORRELATED_ENDPOINT)
        text_csv(self.tmp, "20260912_051000", ANTI_CORRELATED_TEXT)
        rec = reconstruct_from(self.tmp, "20260912_050000", "20260912_051000")
        self.assertEqual(len(rec.join.matched), 5)
        self.assertTrue(rec.join.clean)

    def test_an_item_missing_from_the_text_arm_is_named_not_dropped(self):
        asr_csv(self.tmp, "20260912_050000", ANTI_CORRELATED_ENDPOINT)
        text_csv(self.tmp, "20260912_051000", ANTI_CORRELATED_TEXT[:-1])
        rec = reconstruct_from(self.tmp, "20260912_050000", "20260912_051000")
        self.assertEqual(rec.join.asr_only, ["i5"])
        self.assertFalse(rec.join.clean)

    def test_an_item_missing_a_stage_is_reported_as_incomplete(self):
        asr_csv(self.tmp, "20260912_050000", ANTI_CORRELATED_ENDPOINT)
        partial = [row for row in ANTI_CORRELATED_TEXT]
        text_csv(self.tmp, "20260912_051000", partial)
        # Strip one item's tts_first_chunk row, the way a run that never
        # reached the synthesizer for that item would leave the log.
        path = os.path.join(self.tmp, "text", "17-gemma3", "20260912_051000",
                            "latency_log_text.csv")
        _drop_rows(path, item="i2", stage="tts_first_chunk")
        rec = reconstruct_from(self.tmp, "20260912_050000", "20260912_051000")
        self.assertEqual(rec.join.incomplete, {"i2": ["tts_first_chunk"]})
        self.assertNotIn("i2", rec.values)
        self.assertFalse(rec.join.clean)

    def test_unmatched_items_make_the_cli_exit_nonzero(self):
        asr_csv(self.tmp, "20260912_050000", ANTI_CORRELATED_ENDPOINT)
        text_csv(self.tmp, "20260912_051000", ANTI_CORRELATED_TEXT[:-1])
        report, code = _run_cli_code(["--asr-arm", os.path.join(self.tmp, "asr"),
                                      "--text-arm", os.path.join(self.tmp, "text")])
        self.assertEqual(code, recon.EXIT_UNMATCHED)
        self.assertIn("only in the ASR run: i5", report)


class PercentileConventionTest(unittest.TestCase):
    """`run_statistics.percentile`, not a truncated index."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttfa-recon-pct-")
        self.addCleanup(_rmtree, self.tmp)
        # 20 timed items whose reconstructed values jump from 100 ms to 50 s
        # between the last two order statistics - the shape of a real
        # heavy-tailed stratum, where the two conventions disagree by a
        # factor of nineteen.
        endpoint = [("warm", 5.0)] + [(f"i{k}", 10.0 + k) for k in range(18)] \
            + [("i18", 70.0), ("i19", 49970.0)]
        text = [(item, 20.0, 10.0) for item, _ in endpoint]
        asr_csv(self.tmp, "20260912_050000", endpoint)
        text_csv(self.tmp, "20260912_051000", text)
        self.values = sorted(reconstruct_from(
            self.tmp, "20260912_050000", "20260912_051000").values.values())

    def test_p95_interpolates_between_order_statistics(self):
        self.assertEqual(len(self.values), 20)
        self.assertEqual(recon.describe(self.values)["p95"],
                         rstat.percentile(self.values, 0.95))
        self.assertAlmostEqual(recon.describe(self.values)["p95"], 2595.0)

    def test_a_truncated_index_would_have_read_the_tail_instead(self):
        truncated = self.values[int(len(self.values) * 0.95)]
        self.assertEqual(truncated, 50000.0)
        self.assertNotEqual(recon.describe(self.values)["p95"], truncated)


class QueueHandoffOffsetTest(unittest.TestCase):
    """The +5 ms is carried, named and removable - never absorbed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttfa-recon-offset-")
        self.addCleanup(_rmtree, self.tmp)
        asr_csv(self.tmp, "20260912_050000", ANTI_CORRELATED_ENDPOINT)
        text_csv(self.tmp, "20260912_051000", ANTI_CORRELATED_TEXT)

    def test_default_offset_is_the_measured_constant(self):
        self.assertEqual(recon.QUEUE_HANDOFF_MS, 5.0)
        rec = reconstruct_from(self.tmp, "20260912_050000", "20260912_051000",
                               offset_ms=recon.QUEUE_HANDOFF_MS)
        self.assertEqual(rec.stage_sum["i3"], SUM_OF_MEDIANS_MS)
        self.assertEqual(rec.values["i3"], SUM_OF_MEDIANS_MS + 5.0)

    def test_the_report_states_which_offset_it_used(self):
        applied = _run_cli(["--asr-arm", os.path.join(self.tmp, "asr"),
                            "--text-arm", os.path.join(self.tmp, "text")])
        self.assertIn("queue-handoff offset applied: +5 ms", applied)
        dropped = _run_cli(["--asr-arm", os.path.join(self.tmp, "asr"),
                            "--text-arm", os.path.join(self.tmp, "text"),
                            "--offset-ms", "0"])
        self.assertIn("queue-handoff offset applied: none", dropped)
        self.assertIn("biased low", dropped)

    def test_the_per_item_table_keeps_the_bare_sum_too(self):
        out = os.path.join(self.tmp, "per_item.csv")
        _run_cli(["--asr-arm", os.path.join(self.tmp, "asr"),
                  "--text-arm", os.path.join(self.tmp, "text"),
                  "--per-item-csv", out])
        rows = {row["item"]: row for row in _read_rows(out)}
        self.assertEqual(float(rows["i3"]["stage_sum_ms"]), SUM_OF_MEDIANS_MS)
        self.assertEqual(float(rows["i3"]["offset_ms"]), 5.0)
        self.assertEqual(float(rows["i3"]["reconstructed_ttfa_ms"]),
                         SUM_OF_MEDIANS_MS + 5.0)


class CanonicalAsrRunTest(unittest.TestCase):
    """One frozen pass; the others describe the spread and nothing else."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttfa-recon-canon-")
        self.addCleanup(_rmtree, self.tmp)
        asr_csv(self.tmp, "20260912_050000", ANTI_CORRELATED_ENDPOINT)
        asr_csv(self.tmp, "20260912_050500",
                [(item, delay + 400.0) for item, delay in ANTI_CORRELATED_ENDPOINT])
        text_csv(self.tmp, "20260912_051000", ANTI_CORRELATED_TEXT)

    def test_two_runs_and_no_choice_is_refused(self):
        arm = recon.load_arm("asr", recon.Path(os.path.join(self.tmp, "asr")))
        run, problem = recon.pick_canonical(arm, None)
        self.assertIsNone(run)
        self.assertIn("--canonical", problem)

    def test_the_second_pass_changes_no_reconstructed_value(self):
        rec = reconstruct_from(self.tmp, "20260912_050000", "20260912_051000")
        self.assertEqual(sorted(rec.values.values()),
                         [46.0, 1042.0, 1042.0, 1042.0, 1042.0])

    def test_the_spread_is_reported_separately(self):
        arm = recon.load_arm("asr", recon.Path(os.path.join(self.tmp, "asr")))
        spread = recon.endpoint_spread(arm.runs)
        self.assertEqual(spread.runs, 2)
        self.assertEqual(spread.items_on_every_run, 5)
        self.assertEqual(set(spread.per_item_spread), {400.0})

    def test_the_report_says_the_spread_is_not_propagated(self):
        report = _run_cli(["--asr-arm", os.path.join(self.tmp, "asr"),
                           "--canonical", "20260912_050000",
                           "--text-arm", os.path.join(self.tmp, "text"),
                           "--offset-ms", "0"])
        self.assertIn("NOT propagated", report)
        self.assertIn("median=1042 ms", report)


class RepeatedRowCollapseTest(unittest.TestCase):
    """An item that held several utterances still counts once."""

    def test_repeats_collapse_to_their_median_before_the_sum(self):
        tmp = tempfile.mkdtemp(prefix="ttfa-recon-collapse-")
        self.addCleanup(_rmtree, tmp)
        endpoint = [("warm", 5.0), ("i1", 100.0), ("i2", 200.0), ("i3", 300.0)]
        asr_csv(tmp, "20260912_050000", endpoint)
        text_csv(tmp, "20260912_051000",
                 [(item, 10.0, 5.0) for item, _ in endpoint])
        # i2 answered three times in the text launch; the median of the three
        # is what the item contributes, not their sum and not the last one.
        path = os.path.join(tmp, "text", "17-gemma3", "20260912_051000",
                            "latency_log_text.csv")
        _append_rows(path, [("i2", "llm_ttfc", 1000.0), ("i2", "llm_ttfc", 40.0)])
        rec = reconstruct_from(tmp, "20260912_050000", "20260912_051000")
        self.assertEqual(rec.parts["i2"]["llm_ttfc"], 40.0)
        self.assertEqual(rec.stage_sum["i2"], 200.0 + 40.0 + 5.0)


class MeasuredVersusReconstructedTest(unittest.TestCase):
    """Section 3's comparison, and that it is not the within-run residual."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttfa-recon-valid-")
        self.addCleanup(_rmtree, self.tmp)
        endpoint = [("warm", 5.0), ("i1", 900.0), ("i2", 950.0), ("i3", 1000.0)]
        asr_csv(self.tmp, "20260912_050000", endpoint)
        text_csv(self.tmp, "20260912_051000",
                 [(item, 500.0, 120.0) for item, _ in endpoint])
        # The full-pipeline arm is a separate launch: its llm_ttfc runs 60 ms
        # slower, which is what the comparison has to surface.
        rows = []
        for item, delay in endpoint:
            rows += [(item, "llm_ttft", 400.0), (item, "llm_ttfc", 560.0),
                     (item, "tts_first_chunk", 120.0),
                     (item, "stt_endpoint_delay", delay),
                     (item, "ttfa", delay + 560.0 + 120.0 + 5.0)]
        write_csv(os.path.join(self.tmp, "file", "17-gemma3", "20260912_052000",
                               "latency_log_file.csv"),
                  "17-gemma3", "20260912_052000", "file", rows)

    def test_difference_is_per_item_and_signed(self):
        out = os.path.join(self.tmp, "comparison.csv")
        report = _run_cli(["--asr-arm", os.path.join(self.tmp, "asr"),
                           "--text-arm", os.path.join(self.tmp, "text"),
                           "--measured", os.path.join(self.tmp, "file"),
                           "--comparison-csv", out])
        rows = _read_rows(out)
        self.assertEqual([row["item"] for row in rows], ["i1", "i2", "i3"])
        # measured ttfa carries the same +5 handoff the reconstruction adds,
        # so the difference is the arms' 60 ms llm_ttfc gap and nothing else.
        self.assertEqual({float(row["ttfa_diff_ms"]) for row in rows}, {60.0})
        self.assertEqual({float(row["llm_ttfc_diff_ms"]) for row in rows}, {60.0})
        self.assertEqual({float(row["tts_first_chunk_diff_ms"]) for row in rows}, {0.0})
        self.assertIn("measured - reconstructed", report)

    def test_the_report_separates_it_from_the_within_run_residual(self):
        report = _run_cli(["--asr-arm", os.path.join(self.tmp, "asr"),
                           "--text-arm", os.path.join(self.tmp, "text"),
                           "--measured", os.path.join(self.tmp, "file")])
        self.assertIn("NOT the within-run additivity residual", report)

    def test_a_cell_on_only_one_side_writes_no_row(self):
        """Identity is the cell_id column, so the measured arm is re-stamped."""
        path = os.path.join(self.tmp, "file", "17-gemma3", "20260912_052000",
                            "latency_log_file.csv")
        rows = _read_rows(path)
        for row in rows:
            row["cell_id"] = "99-a-different-model"
        _write_rows(path, rows)
        out = os.path.join(self.tmp, "comparison.csv")
        report, code = _run_cli_code(["--asr-arm", os.path.join(self.tmp, "asr"),
                                      "--text-arm", os.path.join(self.tmp, "text"),
                                      "--measured", os.path.join(self.tmp, "file"),
                                      "--comparison-csv", out])
        self.assertEqual(code, recon.EXIT_OK)
        self.assertEqual(_read_rows(out), [])


# ---------- helpers ----------
def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = position
    return ranks


def _run_cli(argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = recon.main(argv)
    assert code == recon.EXIT_OK, f"exit {code}\n{buffer.getvalue()}"
    return buffer.getvalue()


def _run_cli_code(argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        code = recon.main(argv)
    return buffer.getvalue(), code


def _read_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _drop_rows(path, item, stage):
    _write_rows(path, [row for row in _read_rows(path)
                       if not (row["item"] == item and row["stage"] == stage)])


def _append_rows(path, extra):
    rows = _read_rows(path)
    template = dict(rows[0])
    for item, stage, duration in extra:
        row = dict(template)
        row.update({"item": item, "stage": stage, "duration_ms": duration})
        rows.append(row)
    _write_rows(path, rows)


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
