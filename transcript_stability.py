#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How far two or more ASR passes over the same recordings agree.

Realtime pacing does not deliver chunks on an exactly repeatable schedule, and
Vosk's endpointer can only fire on a chunk boundary, so two passes over one
folder need not produce one transcript. The effect is known rather than
suspected: identical Vosk runs flip about 4% of transcripts while the corpus
word error rate stays stable to +/-0.001. This quantifies it for a given
recording set, which is what makes the number reportable.

Two uses, and they want different columns of the same report:

  - Freezing a canonical pass. Every configuration cell must answer the same
    text or the cells stop being exactly paired, so one pass is frozen and
    reused. The per-item disagreement count says how much that choice matters.
  - Reporting launch-level ASR variability. The corpus-wide pairwise error rate
    is the figure to quote: it is stable even when individual items are not,
    and the gap between the two columns is the whole point.

`endpoint_fire_count` comes from the `stt` rows of the run's latency CSV, not
from the transcripts file, so a pass is named by its run directory and both
artefacts are read from it. An item that fired twice produced two utterances
where the pipeline expected one; a pass that disagrees with another about that
disagrees about what the recording contained, not merely about how it was
spelled.

Word error rate here is a plain S+D+I over whitespace-separated tokens, with no
numeral-reading variants and no normalization beyond case folding -- unlike
`evaluation/asr.py`, which scores a recognizer against written reference text
and must allow "2004" to have been spoken as "two thousand four". Both sides
here came out of the same recognizer under the same configuration, so there is
no convention gap to bridge, and any normalization added would hide the
differences this exists to count.

Standard library only, and no import of the pipeline's modules, for the reason
`run_statistics.py` gives for itself: this reads archived run directories on
whatever interpreter is at hand and must not need the pipeline's environment
installed. Reads `transcripts.jsonl` rather than `transcripts.yaml` for the
same reason -- the runs write both, and only one of them parses without a
third-party module.
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path, PurePath


def tokens(text):
    """Word tokens of a recognizer's output, case folded.

    Nothing else is stripped: Vosk emits lowercase words with no punctuation, so
    anything a normalizer would remove here is a real difference between the
    two passes.
    """
    return (text or "").casefold().split()


def align(reference, hypothesis):
    """Levenshtein counts of hypothesis against reference, as (S, D, I, N).

    One row of the matrix at a time, so a long utterance costs the length of the
    shorter side in memory rather than their product.
    """
    n_ref, n_hyp = len(reference), len(hypothesis)
    # Each cell holds (cost, substitutions, deletions, insertions).
    row = [(j, 0, 0, j) for j in range(n_hyp + 1)]
    for i in range(1, n_ref + 1):
        previous, row = row, [(i, 0, i, 0)] + [None] * n_hyp
        for j in range(1, n_hyp + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                row[j] = previous[j - 1]
                continue
            sub, dele, ins = previous[j - 1], previous[j], row[j - 1]
            best = min(sub, dele, ins, key=lambda cell: cell[0])
            if best is sub:
                row[j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
            elif best is dele:
                row[j] = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
            else:
                row[j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
    _, subs, dels, inss = row[n_hyp]
    return subs, dels, inss, n_ref


def error_rate(pairs):
    """Corpus word error rate over (reference, hypothesis) text pairs.

    Errors and reference words are summed before dividing, which is what makes
    this a corpus rate rather than the mean of per-item rates: a one-word item
    getting one word wrong would otherwise weigh as heavily as a twenty-word
    item doing the same.
    """
    errors = ref_words = 0
    for reference, hypothesis in pairs:
        subs, dels, inss, n_ref = align(tokens(reference), tokens(hypothesis))
        errors += subs + dels + inss
        ref_words += n_ref
    if ref_words == 0:
        return None
    return errors / ref_words


class Pass(object):
    """One ASR pass: what it transcribed, and how often its endpointer fired."""

    def __init__(self, label, path, texts, fires, csv_name):
        self.label = label
        self.path = path
        self.texts = texts
        self.fires = fires
        self.csv_name = csv_name

    @property
    def items(self):
        return set(self.texts)


def find_transcripts(path):
    """Locate the transcripts file of a pass named by directory or by file."""
    if path.is_file():
        return path
    candidate = path / "transcripts.jsonl"
    if candidate.is_file():
        return candidate
    raise ValueError(f"{path}: no transcripts.jsonl here")


def read_transcripts(path):
    """Map item name to recognized text, keyed as the latency CSV keys it.

    `PurePath(...).stem` is what `assistant.py` puts in the CSV's `item` column
    for the same recording, so the two artefacts of a pass join without a
    translation table.
    """
    texts = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON line: {exc}")
            filename = str(record.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"{path}:{line_no}: record has no filename")
            texts[PurePath(filename).stem] = record.get("stt_text") or ""
    return texts


def read_fire_counts(run_dir):
    """Read endpoint_fire_count per item from a run's latency CSV.

    Returns an empty mapping when no CSV sits beside the transcripts: a pass
    archived as transcripts alone can still be compared on its text, and saying
    so beats refusing to run.
    """
    csvs = sorted(glob.glob(os.path.join(str(run_dir), "latency_log_*.csv")))
    if not csvs:
        return {}, None
    fires = {}
    with open(csvs[-1], "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") != "stt":
                continue
            try:
                extra = json.loads(row.get("extra_json") or "{}")
            except ValueError:
                continue
            if "endpoint_fire_count" in extra:
                fires[row.get("item")] = extra["endpoint_fire_count"]
    return fires, os.path.basename(csvs[-1])


def load_pass(label, raw_path):
    path = Path(raw_path)
    if not path.exists():
        raise ValueError(f"{path}: does not exist")
    transcripts = find_transcripts(path)
    fires, csv_name = read_fire_counts(transcripts.parent)
    return Pass(label, raw_path, read_transcripts(transcripts), fires, csv_name)


def format_report(passes):
    """The comparison, as the text a run log should keep."""
    common = set.intersection(*[p.items for p in passes])
    union = set.union(*[p.items for p in passes])
    partial = sorted(union - common)

    lines = [f"Passes compared: {len(passes)}"]
    for p in passes:
        fires = (f"endpoint_fire_count from {p.csv_name}" if p.csv_name
                 else "no latency CSV beside it, so no endpoint_fire_count")
        lines.append(f"  {p.label}  {p.path}")
        lines.append(f"     {len(p.texts)} items, {fires}")

    lines.append("")
    lines.append(f"Items present in every pass: {len(common)} of {len(union)}")
    if partial:
        # A pass writes no transcript record for an item it recognized nothing
        # in, so a name missing here means one pass heard silence where another
        # heard words. That is a disagreement of the largest kind available and
        # is reported apart rather than scored.
        lines.append(f"  present in some but not all: {len(partial)} "
                     f"({', '.join(partial[:8])}{', ...' if len(partial) > 8 else ''})")
    if not common:
        lines.append("")
        lines.append("No item is in every pass; nothing can be compared.")
        return "\n".join(lines), 0

    differing = sorted(
        item for item in common
        if len({tuple(tokens(p.texts[item])) for p in passes}) > 1)
    agreeing = len(common) - len(differing)

    lines.append("")
    lines.append("Transcript agreement")
    lines.append(f"  identical across all passes : {agreeing} of {len(common)} "
                 f"({agreeing / len(common):.1%})")
    lines.append(f"  differing in at least one   : {len(differing)} of {len(common)} "
                 f"({len(differing) / len(common):.1%})")

    lines.append("")
    lines.append("Pairwise word error rate. The first pass of a pair is the reference, "
                 "the rate being")
    lines.append("asymmetric; over all common items it is the launch-level figure, and "
                 "over the")
    lines.append("differing ones alone it says how far apart the disagreements actually are.")
    lines.append(f"  {'pair':<12} {'all items':>12} {'differing only':>16}")
    for i, left in enumerate(passes):
        for right in passes[i + 1:]:
            over_all = error_rate([(left.texts[k], right.texts[k]) for k in sorted(common)])
            over_diff = error_rate([(left.texts[k], right.texts[k]) for k in differing])
            lines.append(f"  {left.label + ' -> ' + right.label:<12} "
                         f"{over_all:>12.4f} "
                         f"{(f'{over_diff:.4f}' if over_diff is not None else '-'):>16}")

    lines.append("")
    scored = sorted(item for item in common
                    if all(item in p.fires for p in passes))
    if not scored:
        lines.append("endpoint_fire_count: not recorded by every pass, so not compared")
    else:
        fire_differing = sorted(
            item for item in scored
            if len({p.fires[item] for p in passes}) > 1)
        lines.append("endpoint_fire_count")
        lines.append(f"  identical across all passes : {len(scored) - len(fire_differing)} "
                     f"of {len(scored)}")
        lines.append(f"  differing                   : {len(fire_differing)} of {len(scored)}")

    listed = sorted(set(differing) | {
        item for item in common
        if all(item in p.fires for p in passes)
        and len({p.fires[item] for p in passes}) > 1})
    if listed:
        lines.append("")
        lines.append("Items that disagree, by name:")
        for item in listed:
            fires = "/".join(str(p.fires.get(item, "-")) for p in passes)
            what = []
            if item in differing:
                what.append("text")
            if len({p.fires[item] for p in passes if item in p.fires}) > 1:
                what.append("fires")
            lines.append(f"  {item:<24} {', '.join(what):<12} fires {fires}")

    return "\n".join(lines), len(differing)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("passes", nargs="+",
                        help="Two or more ASR passes over the same recordings, each a run "
                             "directory or a transcripts.jsonl inside one.")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Also write the headline figures here, so a campaign can keep "
                             "the agreement number without re-parsing the report.")
    args = parser.parse_args()

    if len(args.passes) < 2:
        print("error: at least two passes are needed to compare any", file=sys.stderr)
        return 1

    passes = []
    for index, raw_path in enumerate(args.passes):
        try:
            # A, B, C ... rather than the paths, which are timestamped run
            # directories too long to head a table column.
            passes.append(load_pass(chr(ord("A") + index), raw_path))
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    report, differing = format_report(passes)
    print(report)

    if args.json_path:
        common = set.intersection(*[p.items for p in passes])
        summary = {
            "passes": [{"label": p.label, "path": str(p.path), "items": len(p.texts)}
                       for p in passes],
            "common_items": len(common),
            "differing_items": differing,
            "pairwise_wer": {
                f"{left.label}->{right.label}": error_rate(
                    [(left.texts[k], right.texts[k]) for k in sorted(common)])
                for i, left in enumerate(passes) for right in passes[i + 1:]
            },
        }
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"\nHeadline figures written to: {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
