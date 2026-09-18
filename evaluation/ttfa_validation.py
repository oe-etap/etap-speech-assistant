#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agreement between measured TTFA and TTFA reconstructed from split ASR + LLM arms.

The pipeline can run recognition and generation separately. Reconstructed TTFA
is the sum of the recognizer's endpoint delay, the time to the first speakable
LLM chunk, the first TTS chunk, and a small queue-handoff offset measured on
archived items. This module does not invent that reconstruction: it scores a
table the measurement campaign already wrote (`measured_vs_reconstructed.csv`)
with a method comparison that can support or reject using the reconstruction
in place of a realtime measurement.

Method: Bland and Altman (1986). Status: validated. A high Pearson correlation
between the two columns is not agreement and is not reported as such.

    python -m evaluation.ttfa_validation --csv outputs/ttfa-validation-heyval/measured_vs_reconstructed.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from . import references, reporting
from .stats import bland_altman

PathLike = Union[str, Path]

# Predeclared practical agreement bands, in milliseconds. The 100 ms band is
# the same margin the paired TTFA TOST uses; 50 ms is half of that; 5 ms is
# the engineering scatter reported from the reconstruction's own offset fit.
# Choosing them after seeing the LoA would make the "within band" rates a
# formality, so they live here, not in the report writer.
AGREEMENT_BANDS_MS = (5.0, 50.0, 100.0)

REFERENCE_KEYS = ["bland_altman"]


def load_pairs(path: PathLike,
               measured_column: str = "measured_ttfa_ms",
               reconstructed_column: str = "reconstructed_ttfa_ms",
               exclude_stems: Optional[Sequence[str]] = None
               ) -> List[Dict[str, Any]]:
    """Read one row per (cell, item) of measured vs reconstructed TTFA."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TTFA reconstruction table not found: {path}")

    blocked = {str(stem).strip().lower() for stem in (exclude_stems or [])
               if stem}
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item = str(row.get("item") or "").strip()
            if item.lower() in blocked:
                continue
            try:
                measured = float(row[measured_column])
                reconstructed = float(row[reconstructed_column])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({
                "cell_id": str(row.get("cell_id") or ""),
                "item": item,
                "reconstructed_ttfa_ms": reconstructed,
                "measured_ttfa_ms": measured,
                "diff_ms": measured - reconstructed,
            })
    return rows


def evaluate_agreement(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Bland-Altman overall and per cell.

    Bias is mean(measured - reconstructed). A positive bias means the split
    reconstruction underestimates the wait the user would have heard.
    """
    overall = bland_altman(
        [row["reconstructed_ttfa_ms"] for row in rows],
        [row["measured_ttfa_ms"] for row in rows])

    by_cell: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell_id"]].append(row)

    cells = []
    for cell_id, group in sorted(by_cell.items()):
        result = bland_altman(
            [row["reconstructed_ttfa_ms"] for row in group],
            [row["measured_ttfa_ms"] for row in group])
        cells.append({"cell_id": cell_id, **result.as_dict()})

    return {
        "n_pairs": overall.n,
        "n_cells": len(cells),
        "overall": overall.as_dict(),
        "per_cell": cells,
        "agreement_bands_ms": list(AGREEMENT_BANDS_MS),
        "method": "Bland and Altman, 1986",
        "status": "validated",
        "bias_definition": "mean(measured - reconstructed); positive = reconstruction underestimates",
    }


def render_report(payload: Dict[str, Any], source: Path) -> str:
    overall = payload["overall"]
    lines = [
        "=" * 88,
        "TTFA RECONSTRUCTION AGREEMENT".center(88),
        "=" * 88,
        f"Source                 : {source}",
        f"Paired items           : {payload['n_pairs']}",
        f"Configuration cells    : {payload['n_cells']}",
        f"Method                 : {payload['method']}  [{payload['status']}]",
        f"Bias definition        : {payload['bias_definition']}",
        "",
        "Overall",
        "-" * 88,
        f"  bias (measured - reconstructed) : {overall.get('bias')} ms",
        f"  SD of differences               : {overall.get('sd_diff')} ms",
        f"  95% limits of agreement         : {overall.get('loa_low')} to "
        f"{overall.get('loa_high')} ms",
        f"  MAE / RMSE                      : {overall.get('mae')} / "
        f"{overall.get('rmse')} ms",
        f"  Spearman rho                    : {overall.get('spearman')}",
        f"  within 5 / 50 / 100 ms          : "
        f"{_pct(overall.get('pct_within_5ms'))} / "
        f"{_pct(overall.get('pct_within_50ms'))} / "
        f"{_pct(overall.get('pct_within_100ms'))}",
        "",
        "A reconstruction whose 95% LoA lie inside +/- 100 ms (the predeclared",
        "TTFA TOST margin) can stand in for measured TTFA at the resolution of",
        "the configuration contrasts. A LoA that crosses 100 ms cannot; the",
        "split design then supports stage decomposition, not absolute wait.",
        "",
        f"{'Cell':<52} {'n':>5} {'bias':>8} {'LoA low':>9} {'LoA high':>9} "
        f"{'MAE':>8} {'<=100ms':>8}",
        "-" * 88,
    ]
    for cell in payload["per_cell"]:
        lines.append(
            f"{cell['cell_id']:<52} {cell['n']:>5} "
            f"{_fmt(cell.get('bias')):>8} {_fmt(cell.get('loa_low')):>9} "
            f"{_fmt(cell.get('loa_high')):>9} {_fmt(cell.get('mae')):>8} "
            f"{_pct(cell.get('pct_within_100ms')):>8}")
    lines += [
        "",
        "Method references",
        "-" * 88,
    ]
    lines.extend(references.bibliography_lines(REFERENCE_KEYS))
    lines.append("=" * 88)
    return "\n".join(lines)


def write_agreement(path: PathLike, out_dir: PathLike,
                    exclude_stems: Optional[Sequence[str]] = None) -> Path:
    source = Path(path)
    rows = load_pairs(source, exclude_stems=exclude_stems)
    if not rows:
        raise ValueError(f"no paired TTFA rows in {source}")
    payload = evaluate_agreement(rows)
    report = render_report(payload, source)

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    reporting.write_text(target / "ttfa_agreement_report.txt", report)
    reporting.write_json(target / "ttfa_agreement.json", payload)
    _write_cell_csv(target / "ttfa_agreement_by_cell.csv", payload["per_cell"])
    return target


def _write_cell_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.ttfa_validation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True,
                        help="measured_vs_reconstructed.csv from the validation campaign")
    parser.add_argument("--out-dir", type=Path,
                        help="Where to write the agreement report")
    parser.add_argument("--exclude-item", action="append", default=[],
                        help="Recording stem to drop. Repeatable.")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or args.csv.parent / "ttfa_agreement"
    try:
        target = write_agreement(args.csv, out_dir, args.exclude_item)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Artefacts written to: {target.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
