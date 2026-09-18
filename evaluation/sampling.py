#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified item sampling for costly evaluation tiers.

The judge round scores a subset, not the full factorial. Sampling is
proportional to recognizer-error strata (clean / mild / severe) so that the
sample has the same mix of recognition quality as the corpus. The sample is
drawn once, with a fixed seed, and reused across every configuration cell;
drawing a fresh sample per cell would confound model differences with item
differences.

Allocation uses the largest-remainder (Hamilton) method: each stratum receives
the floor of its proportional share, then leftover seats go to the largest
fractional remainders. Within a stratum, items are drawn without replacement
by ``random.Random(seed)``.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .loaders import item_stem


def proportional_counts(group_sizes: Mapping[str, int], n: int) -> Dict[str, int]:
    """Largest-remainder allocation of ``n`` seats across groups.

    Empty groups receive zero. If ``n`` exceeds the population, every item is
    taken. Group names that are missing from ``group_sizes`` are ignored.
    """
    positive = {key: int(size) for key, size in group_sizes.items() if int(size) > 0}
    if n <= 0 or not positive:
        return {key: 0 for key in group_sizes}
    population = sum(positive.values())
    n = min(int(n), population)
    raw = {key: n * size / population for key, size in positive.items()}
    floors = {key: int(math.floor(share)) for key, share in raw.items()}
    leftover = n - sum(floors.values())
    # Largest remainder, then larger group, then name: a deterministic tie-break.
    order = sorted(
        raw,
        key=lambda key: (raw[key] - floors[key], positive[key], key),
        reverse=True)
    allocated = {key: floors[key] for key in positive}
    for key in order[:leftover]:
        allocated[key] += 1
    return {key: allocated.get(key, 0) for key in group_sizes}


def stratified_sample(rows: Sequence[Mapping[str, str]],
                      n: int,
                      seed: int = 0,
                      stratum_key: str = "stt_stratum",
                      id_key: str = "item_id",
                      exclude: Optional[Sequence[str]] = None
                      ) -> List[Dict[str, str]]:
    """Draw a proportional stratified sample of ``n`` rows.

    Rows whose stem is in ``exclude`` are dropped before allocation. The
    returned rows keep their original fields and are ordered by stratum name
    then item id, so two runs with the same seed write the same file.
    """
    blocked = {item_stem(stem) for stem in (exclude or []) if stem}
    eligible: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        mapping = dict(row)
        stem = item_stem(mapping.get(id_key) or mapping.get("filename") or "")
        if not stem or stem in blocked or stem in seen:
            continue
        seen.add(stem)
        mapping["_stem"] = stem
        eligible.append(mapping)

    buckets: Dict[str, List[Dict[str, str]]] = {}
    for row in eligible:
        buckets.setdefault(str(row.get(stratum_key) or "unknown"), []).append(row)
    sizes = {name: len(members) for name, members in buckets.items()}
    quotas = proportional_counts(sizes, n)

    rng = random.Random(seed)
    picked: List[Dict[str, str]] = []
    for name in sorted(buckets):
        members = list(buckets[name])
        members.sort(key=lambda row: row["_stem"])
        rng.shuffle(members)
        take = quotas.get(name, 0)
        picked.extend(members[:take])

    picked.sort(key=lambda row: (str(row.get(stratum_key) or ""), row["_stem"]))
    for row in picked:
        row.pop("_stem", None)
    return picked


def load_reference_rows(path: Path) -> List[Dict[str, str]]:
    """Read an ``input_reference.csv`` (or any table with ``item_id``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"reference table not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sample_from_reference_csv(path: Path,
                              n: int,
                              seed: int = 0,
                              exclude: Optional[Sequence[str]] = None
                              ) -> List[Dict[str, str]]:
    """Stratified sample from a recognizer reference table."""
    return stratified_sample(load_reference_rows(path), n=n, seed=seed,
                             exclude=exclude)


def stems_of(rows: Iterable[Mapping[str, str]],
             id_key: str = "item_id") -> List[str]:
    """Recording stems in sample order, one per row."""
    stems = []
    for row in rows:
        stem = item_stem(row.get(id_key) or row.get("filename") or "")
        if stem:
            stems.append(stem)
    return stems


def write_sample_csv(path: Path, rows: Sequence[Mapping[str, str]],
                     fields: Optional[Sequence[str]] = None) -> None:
    """Write the sample table. Extra columns are kept when present."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = ["item_id", "filename", "stt_stratum", "stt_wer", "ori_text",
                 "stt_text"]
    if fields is None:
        seen: List[str] = []
        for name in preferred:
            if any(name in row for row in rows):
                seen.append(name)
        extra = []
        for row in rows:
            for name in row:
                if name not in seen and name not in extra:
                    extra.append(name)
        fields = seen + extra
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def stratum_counts(rows: Sequence[Mapping[str, str]],
                   stratum_key: str = "stt_stratum") -> Dict[str, int]:
    """How many sampled rows fall in each stratum."""
    counts: MutableMapping[str, int] = {}
    for row in rows:
        name = str(row.get(stratum_key) or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return dict(counts)
