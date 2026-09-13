#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What the validation runner has to get right before anyone spends hours on it.

The experiment it drives costs cells x replicates x two arms of realtime
launches, so the expensive failures are the ones that only show up at the
end: a subset with no endpoint-hard item in it (the overlap the experiment
exists to size lives on exactly those recordings), a text arm pointed at the
wrong transcripts, or a measured arm reading a different corpus than the
subset. All three are decided before the first launch starts, so all three
are checked here against the command lines rather than against the runs.

The draw is seeded and asserted to be reproducible for the same reason
`run_campaign.py` seeds its launch order: a subset nobody can regenerate
makes the measurement it produced unrepeatable.

Run:  python -m unittest discover -s tests -v
"""

import csv
import io
import contextlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ttfa_validation as val              # noqa: E402

METADATA_COLUMNS = ["id", "filename", "question", "duration_ms", "speech_end_ms",
                    "max_internal_pause_ms"]


def corpus(tmp, keys, with_audio=True):
    """A metadata.csv (and optionally its WAVs) with the given hardness keys."""
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for index, (name, key) in enumerate(sorted(keys.items()), start=1):
        rows.append({"id": index, "filename": name, "question": f"q{index}",
                     "duration_ms": 7000, "speech_end_ms": 5000,
                     "max_internal_pause_ms": key})
        if with_audio:
            with open(os.path.join(tmp, name), "wb") as handle:
                handle.write(b"RIFF" + bytes(40))
    with open(os.path.join(tmp, "metadata.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# Heavy-tailed the way the real corpus is: almost everything settles at once
# and a handful of recordings carry a long internal pause. Equal-width bands
# over this would leave the top band empty most of the time.
KEYS = {f"{index:05d}.wav": key for index, key in enumerate(
    [0, 0, 0, 0, 32, 64, 96, 128, 256, 512, 1024, 1728], start=1)}
HARDEST = "00012.wav"


class StratifiedPickTest(unittest.TestCase):

    def test_every_band_is_represented(self):
        chosen, bands = val.stratified_pick(KEYS, wanted=6, strata=3, seed=1)
        self.assertEqual(len(chosen), 6)
        for band in bands:
            self.assertTrue(set(band) & set(chosen), f"band {band} unrepresented")

    def test_the_hardest_recording_is_always_in(self):
        """The case the experiment exists to cover cannot be left to the draw."""
        for seed in range(25):
            chosen, _ = val.stratified_pick(KEYS, wanted=3, strata=3, seed=seed)
            self.assertIn(HARDEST, chosen, f"seed {seed} dropped the hardest item")

    def test_the_draw_is_reproducible_and_seed_dependent(self):
        first = val.stratified_pick(KEYS, wanted=6, strata=3, seed=7)[0]
        self.assertEqual(first, val.stratified_pick(KEYS, wanted=6, strata=3, seed=7)[0])
        others = {tuple(val.stratified_pick(KEYS, wanted=6, strata=3, seed=s)[0])
                  for s in range(1, 12)}
        self.assertGreater(len(others), 1)

    def test_asking_for_more_than_there_is_takes_everything(self):
        chosen, _ = val.stratified_pick(KEYS, wanted=99, strata=3, seed=1)
        self.assertEqual(sorted(chosen), sorted(KEYS))

    def test_bands_are_equal_count_not_equal_width(self):
        _, bands = val.stratified_pick(KEYS, wanted=6, strata=3, seed=1)
        self.assertEqual([len(band) for band in bands], [4, 4, 4])


class WarmupRecordingTest(unittest.TestCase):
    """One recording is spent on the exclusion so the draw keeps all of its own."""

    def test_the_warmup_is_whatever_sorts_first(self):
        """`assistant.py` runs `sorted(args.audio)`, so sort order picks it."""
        self.assertEqual(val.pick_warmup(KEYS), sorted(KEYS)[0])

    def test_the_draw_never_spends_a_stratum_on_it(self):
        warmup = val.pick_warmup(KEYS)
        pool = {name: key for name, key in KEYS.items() if name != warmup}
        chosen, _ = val.stratified_pick(pool, wanted=3, strata=3, seed=1)
        self.assertNotIn(warmup, chosen)
        self.assertIn(HARDEST, chosen)

    def test_the_subset_folder_puts_it_ahead_of_every_chosen_item(self):
        tmp = tempfile.mkdtemp(prefix="ttfa-val-warm-")
        self.addCleanup(_rmtree, tmp)
        source = os.path.join(tmp, "audios")
        rows = corpus(source, KEYS)
        dest = os.path.join(tmp, "subset")
        warmup = val.pick_warmup(KEYS)
        chosen = [HARDEST, "00006.wav"]
        val.build_subset(source, rows, [warmup] + chosen, dest)
        present = sorted(name for name in os.listdir(dest) if name.endswith(".wav"))
        self.assertEqual(present[0], warmup)
        self.assertEqual(len(present), len(chosen) + 1)


class SubsetFolderTest(unittest.TestCase):

    def test_the_subset_is_a_corpus_of_its_own(self):
        tmp = tempfile.mkdtemp(prefix="ttfa-val-subset-")
        self.addCleanup(_rmtree, tmp)
        source = os.path.join(tmp, "audios")
        rows = corpus(source, KEYS)
        dest = os.path.join(tmp, "subset")
        chosen = [HARDEST, "00001.wav", "00006.wav"]
        kept = val.build_subset(source, rows, chosen, dest)

        self.assertEqual(sorted(os.listdir(dest)),
                         sorted(chosen + ["metadata.csv"]))
        self.assertEqual([row["filename"] for row in kept], chosen)
        with open(os.path.join(dest, "metadata.csv"), encoding="utf-8") as handle:
            written = list(csv.DictReader(handle))
        self.assertEqual([row["filename"] for row in written], chosen)
        self.assertEqual(list(written[0]), METADATA_COLUMNS)

    def test_a_chosen_recording_that_is_not_on_disk_is_fatal(self):
        tmp = tempfile.mkdtemp(prefix="ttfa-val-missing-")
        self.addCleanup(_rmtree, tmp)
        source = os.path.join(tmp, "audios")
        rows = corpus(source, KEYS, with_audio=False)
        with self.assertRaises(SystemExit):
            val.build_subset(source, rows, [HARDEST], os.path.join(tmp, "subset"))


class PlannedCommandsTest(unittest.TestCase):
    """The three arms' command lines, asserted before anything is launched."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttfa-val-plan-")
        self.addCleanup(_rmtree, self.tmp)
        self.audio = os.path.join(self.tmp, "audios")
        corpus(self.audio, KEYS)
        configs = os.path.join(self.tmp, "configs")
        os.makedirs(configs, exist_ok=True)
        for cell in ("17-small", "21-large"):
            open(os.path.join(configs, f"{cell}.yaml"), "w").close()
        self.commands = []
        real_run = val.run
        val.run = lambda cmd, dry_run=False, cwd=None: (self.commands.append(cmd) or 0)
        self.addCleanup(setattr, val, "run", real_run)

    def plan(self, *extra):
        argv = ["--out-dir", self.tmp, "--audio-dir", self.audio,
                "--cells", "17-small", "21-large",
                "--configs-dir", os.path.join(self.tmp, "configs"),
                "--items", "6", "--strata", "3", "--replicates", "3",
                "--seed", "20260912", "--dry-run"] + list(extra)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = val.main(argv)
        return code, out.getvalue()

    def test_every_phase_issues_exactly_one_command(self):
        code, _ = self.plan("--", "--llm-num-gpu", "0")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.commands), 4)
        scripts = [cmd[1] for cmd in self.commands]
        self.assertEqual(scripts, ["assistant.py", "run_campaign.py",
                                   "run_campaign.py", "ttfa_reconstruct.py"])

    def test_the_asr_pass_is_one_launch_and_builds_no_llm(self):
        self.plan()
        asr = self.commands[0]
        self.assertIn("--asr-only", asr)
        self.assertNotIn("--rounds", asr)
        self.assertEqual(asr[asr.index("--cell-id") + 1], "asr-pass")

    def test_the_text_arm_reads_the_canonical_pass_transcripts(self):
        self.plan()
        text = self.commands[1]
        self.assertIn("--input-mode", text)
        transcripts = text[text.index("--transcripts") + 1]
        self.assertTrue(transcripts.startswith(os.path.join(self.tmp, "asr")))
        self.assertTrue(transcripts.endswith("transcripts.jsonl"))
        self.assertNotIn("--audio", text)

    def test_the_measured_arm_reads_the_subset_and_nothing_else(self):
        self.plan()
        measured = self.commands[2]
        self.assertEqual(measured[measured.index("--audio") + 1],
                         os.path.join(self.tmp, "subset"))
        self.assertNotIn("--input-mode", measured)

    def test_both_arms_get_the_same_cells_and_replicates(self):
        self.plan()
        for arm in self.commands[1:3]:
            self.assertEqual(arm[arm.index("--rounds") + 1], "3")
            self.assertEqual(arm[arm.index("--seed") + 1], "20260912")
            cells = arm[arm.index("--cells") + 1:arm.index("--out-dir")]
            self.assertEqual(cells, ["17-small", "21-large"])

    def test_extra_args_reach_every_launch(self):
        self.plan("--", "--llm-num-gpu", "0")
        for cmd in self.commands[:3]:
            self.assertIn("--llm-num-gpu", cmd)

    def test_the_comparison_reads_all_three_arms(self):
        self.plan()
        compare = self.commands[3]
        for flag, arm in (("--asr-arm", "asr"), ("--text-arm", "text"),
                          ("--measured", "measured")):
            self.assertEqual(compare[compare.index(flag) + 1],
                             os.path.join(self.tmp, arm))

    def test_stop_after_runs_no_later_phase(self):
        self.plan("--stop-after", "asr")
        self.assertEqual([cmd[1] for cmd in self.commands], ["assistant.py"])

    def test_a_dry_run_writes_no_subset(self):
        self.plan()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "subset")))


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
