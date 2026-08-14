#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distribution-free statistics for the latency logs.

Latency samples are bounded below, unbounded above and routinely skewed: one
item that hits a cold cache moves the mean while leaving the other hundred
exactly as they were. The mean and the standard deviation therefore describe
these samples poorly, and everything here is built instead on order statistics
and ranks, which assume nothing about the shape of the distribution.

The few figures that do concern the mean carry a bootstrap interval rather than
a normal one, for the same reason.

Standard library only, deliberately: this runs over logs long after the run
produced them, on whatever interpreter is at hand, and must not need the
pipeline's environment to be installed.
"""

from collections import Counter
from dataclasses import dataclass
import math
import random
import statistics
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Bootstrap resampling is random, and a report that changes between two runs
# over the same logs is a report nobody can cite. Every draw comes from a
# generator seeded with this unless the caller passes its own.
DEFAULT_SEED = 20260814

# Below this many non-zero differences the two-sided signed-rank test cannot
# reach p < 0.05 even when every difference points the same way, so a p-value
# is not worth printing.
MIN_PAIRS_FOR_SIGNED_RANK = 6

# The percentile bootstrap runs narrow on small samples: it can only resample
# the values it was given, and a handful of them carry no information about the
# tail they were drawn from. Measured coverage of a nominal 95% interval over a
# lognormal like the logged latencies, 1500 trials each:
#
#     n =   4     6     8    10    15    20    30    60   120
#         0.77  0.83  0.85  0.88  0.91  0.90  0.93  0.93  0.94
#
# An interval claiming 95% and delivering 77% is worse than no interval, so
# below this the mean simply goes without one. Even above it the figure runs a
# little narrow, which is why the median's exact interval is the one to quote.
MIN_N_FOR_BOOTSTRAP = 20


@dataclass
class Interval:
    """A confidence interval. A bound is None where the sample cannot place it.

    That happens with small samples: a distribution-free interval for the 95th
    percentile needs enough observations above it to have something to point
    at, and n = 20 does not have them. None says the data is silent, which is
    the honest report and not the same as a wide interval.
    """
    low: Optional[float]
    high: Optional[float]

    @property
    def bounded(self) -> bool:
        return self.low is not None and self.high is not None

    @property
    def width(self) -> Optional[float]:
        return self.high - self.low if self.bounded else None


@dataclass
class Summary:
    """What one stage's per-item durations look like."""
    n: int
    mean: float
    sd: Optional[float]
    minimum: float
    maximum: float
    quantiles: Dict[float, float]
    quantile_intervals: Dict[float, Interval]
    mean_interval: Interval

    @property
    def median(self) -> float:
        return self.quantiles[0.5]

    @property
    def iqr(self) -> Optional[float]:
        if 0.25 in self.quantiles and 0.75 in self.quantiles:
            return self.quantiles[0.75] - self.quantiles[0.25]
        return None


@dataclass
class PairedComparison:
    """A paired A/B result for one stage, plus what it took to get it."""
    n_pairs: int
    n_zero: int
    shift: Optional[float]          # Hodges-Lehmann estimate of the median shift
    shift_interval: Interval
    p_value: Optional[float]
    p_adjusted: Optional[float]
    method: str
    median_a: Optional[float]
    median_b: Optional[float]
    detectable_shift: Optional[float]
    note: str = ""

    @property
    def relative_shift(self) -> Optional[float]:
        """The shift as a fraction of the baseline median."""
        if self.shift is None or not self.median_b:
            return None
        return self.shift / self.median_b


@dataclass
class UnpairedComparison:
    """A rank-sum result, for when the two sides do not share their items."""
    n_a: int
    n_b: int
    shift: Optional[float]
    p_value: Optional[float]
    p_adjusted: Optional[float]
    effect_size: Optional[float]    # Cliff's delta
    median_a: Optional[float]
    median_b: Optional[float]
    method: str = "Mann-Whitney U (normal approximation)"
    note: str = ""


# ---------- Order statistics ----------
def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """The q-quantile by linear interpolation between order statistics.

    Matches the convention `vosk_endpoint_sweep.py` already reports its delay
    percentiles under, so figures from the two scripts are comparable.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _binomial_cdf_table(n: int, p: float) -> List[float]:
    """P(X <= k) for every k in 0..n, X ~ Binomial(n, p).

    Summed through logarithms so the individual terms cannot underflow: with
    120 items and p = 0.95 the smallest of them sits near 1e-150, and the
    straightforward product loses it entirely.
    """
    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_n_factorial = math.lgamma(n + 1)
    table = []
    total = 0.0
    for i in range(n + 1):
        log_term = (log_n_factorial - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * log_p + (n - i) * log_q)
        total = min(total + math.exp(log_term), 1.0)
        table.append(total)
    return table


def quantile_interval(values: Sequence[float], q: float,
                      confidence: float = 0.95) -> Interval:
    """A distribution-free confidence interval for the q-quantile.

    The number of observations falling below the true quantile is Binomial(n, q)
    whatever the distribution is, so a pair of order statistics brackets it with
    a coverage that can be read straight off that binomial. No normality, no
    bootstrap, and the coverage is exact rather than asymptotic - it is only
    conservative in that the discrete ranks cannot land on 95% precisely.

    Returns unbounded ends where n is too small for the quantile asked for.
    """
    n = len(values)
    if n < 2 or not 0.0 < q < 1.0:
        return Interval(None, None)

    ordered = sorted(values)
    alpha = 1.0 - confidence
    cdf = _binomial_cdf_table(n, q)

    # The widest lower rank whose exceedance probability still fits in alpha/2,
    # and the tightest upper rank that covers the rest.
    lower_rank = None
    for rank in range(1, n + 1):
        if cdf[rank - 1] <= alpha / 2:
            lower_rank = rank
        else:
            break

    upper_rank = None
    for rank in range(1, n + 1):
        if cdf[rank - 1] >= 1 - alpha / 2:
            upper_rank = rank
            break

    return Interval(ordered[lower_rank - 1] if lower_rank else None,
                    ordered[upper_rank - 1] if upper_rank else None)


def bootstrap_interval(values: Sequence[float],
                       statistic: Callable[[Sequence[float]], float] = statistics.fmean,
                       confidence: float = 0.95,
                       resamples: int = 5000,
                       seed: int = DEFAULT_SEED) -> Interval:
    """Percentile bootstrap interval for any statistic of one sample.

    Used for the mean, where the skew of a latency sample makes the textbook
    t interval lean the wrong way. Returns an empty interval when resampling
    is switched off, or when the sample is smaller than MIN_N_FOR_BOOTSTRAP
    and the interval would claim more confidence than it delivers.
    """
    n = len(values)
    if n < MIN_N_FOR_BOOTSTRAP or resamples < 100:
        return Interval(None, None)

    rng = random.Random(seed)
    pool = list(values)
    draws = [statistic(rng.choices(pool, k=n)) for _ in range(resamples)]
    draws.sort()
    alpha = 1.0 - confidence
    return Interval(percentile(draws, alpha / 2), percentile(draws, 1 - alpha / 2))


def summarize(values: Sequence[float],
              quantiles: Sequence[float] = (0.5, 0.9, 0.95),
              confidence: float = 0.95,
              resamples: int = 5000,
              seed: int = DEFAULT_SEED) -> Optional[Summary]:
    """Describe one sample: quantiles with their intervals, mean with its own.

    The quartiles are always computed whether or not they were asked for, since
    the IQR is what says how spread out the sample is without a tail deciding
    the answer.
    """
    if not values:
        return None

    wanted = sorted(set(quantiles) | {0.25, 0.5, 0.75})
    return Summary(
        n=len(values),
        mean=statistics.fmean(values),
        sd=statistics.stdev(values) if len(values) > 1 else None,
        minimum=min(values),
        maximum=max(values),
        quantiles={q: percentile(values, q) for q in wanted},
        quantile_intervals={q: quantile_interval(values, q, confidence) for q in wanted},
        mean_interval=bootstrap_interval(values, confidence=confidence,
                                         resamples=resamples, seed=seed),
    )


# ---------- Ranks ----------
def _average_ranks(values: Sequence[float]) -> List[float]:
    """Ranks 1..n in input order, tied values sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2 + 1
        for position in range(start, stop + 1):
            ranks[order[position]] = shared
        start = stop + 1
    return ranks


def _exact_signed_rank_p(w_plus: float, n: int) -> float:
    """Two-sided p from the exact null distribution of the signed-rank sum.

    Counts the sign assignments reaching each possible rank sum, which is the
    whole null distribution when no two absolute differences tie. Only worth
    the table for small n; the caller falls back to the approximation above it.
    """
    total_rank = n * (n + 1) // 2
    counts = [0] * (total_rank + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for total in range(total_rank, rank - 1, -1):
            counts[total] += counts[total - rank]

    assignments = 2 ** n
    at_or_below = sum(counts[:int(w_plus) + 1]) / assignments
    at_or_above = sum(counts[int(math.ceil(w_plus)):]) / assignments
    return min(1.0, 2 * min(at_or_below, at_or_above))


def wilcoxon_signed_rank(differences: Sequence[float]) -> Tuple[Optional[float], str, int]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Returns (p-value, the method used, how many exact zeros were dropped).

    Pairing is what makes this worth doing. Two runs over the same recordings
    differ item by item for reasons that have nothing to do with the change
    under test - one file is longer, another is noisier - and comparing each
    item only against itself removes all of that from the comparison. What is
    left is the shift, which is what the test is asked about.
    """
    nonzero = [d for d in differences if d != 0]
    dropped = len(differences) - len(nonzero)
    n = len(nonzero)
    if n < MIN_PAIRS_FOR_SIGNED_RANK:
        return None, "too few non-zero differences", dropped

    magnitudes = [abs(d) for d in nonzero]
    ranks = _average_ranks(magnitudes)
    w_plus = sum(rank for rank, d in zip(ranks, nonzero) if d > 0)

    tie_sizes = Counter(magnitudes)
    tied = any(size > 1 for size in tie_sizes.values())

    if n <= 20 and not tied:
        return _exact_signed_rank_p(w_plus, n), "Wilcoxon signed-rank (exact)", dropped

    mean_w = n * (n + 1) / 4
    tie_correction = sum(size ** 3 - size for size in tie_sizes.values()) / 48
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_correction
    if variance <= 0:
        return None, "no variation between the pairs", dropped

    deviation = w_plus - mean_w
    # Continuity correction, applied toward the mean so it can only widen p.
    corrected = deviation - math.copysign(min(0.5, abs(deviation)), deviation)
    z = corrected / math.sqrt(variance)
    return 2 * NormalDist().cdf(-abs(z)), "Wilcoxon signed-rank (normal approx.)", dropped


def _walsh_averages(differences: Sequence[float]) -> List[float]:
    """Every pairwise mean (d_i + d_j) / 2 including i == j, sorted."""
    ordered = sorted(differences)
    n = len(ordered)
    return sorted((ordered[i] + ordered[j]) / 2
                  for i in range(n) for j in range(i, n))


def hodges_lehmann(differences: Sequence[float],
                   confidence: float = 0.95) -> Tuple[Optional[float], Interval]:
    """The median shift between the paired sides, with its interval.

    The estimate is the median of the Walsh averages rather than the median of
    the differences: it is the location estimate the signed-rank test is
    actually about, so a shift whose interval clears zero and a significant
    p-value always agree. Its interval comes from the same rank distribution
    and needs no assumption about shape.
    """
    n = len(differences)
    if n < 2:
        return (float(differences[0]) if n else None), Interval(None, None)

    walsh = _walsh_averages(differences)
    estimate = percentile(walsh, 0.5)

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    span = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    k = math.floor(n * (n + 1) / 4 - z * span)
    if k < 0 or 2 * k >= len(walsh):
        return estimate, Interval(None, None)
    return estimate, Interval(walsh[k], walsh[len(walsh) - 1 - k])


def mann_whitney(sample_a: Sequence[float],
                 sample_b: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Two-sided rank-sum test. Returns (p-value, Cliff's delta).

    The fallback for sides that do not share their items, and a markedly
    weaker instrument than the paired test: everything the pairing would have
    cancelled stays in the comparison as noise.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 3 or n_b < 3:
        return None, None

    pooled = list(sample_a) + list(sample_b)
    ranks = _average_ranks(pooled)
    rank_sum_a = sum(ranks[:n_a])
    u_a = rank_sum_a - n_a * (n_a + 1) / 2

    mean_u = n_a * n_b / 2
    tie_sizes = Counter(pooled)
    n = n_a + n_b
    tie_correction = sum(size ** 3 - size for size in tie_sizes.values())
    variance = n_a * n_b / 12 * ((n + 1) - tie_correction / (n * (n - 1)))
    if variance <= 0:
        return None, None

    deviation = u_a - mean_u
    corrected = deviation - math.copysign(min(0.5, abs(deviation)), deviation)
    z = corrected / math.sqrt(variance)
    p_value = 2 * NormalDist().cdf(-abs(z))

    # Cliff's delta: how much more often an item from A beats one from B than
    # the other way round, on a -1..1 scale that no unit of measure enters.
    delta = (2 * u_a) / (n_a * n_b) - 1
    return p_value, delta


def two_sample_shift(sample_a: Sequence[float], sample_b: Sequence[float]) -> Optional[float]:
    """Hodges-Lehmann shift for unpaired sides: the median of all differences."""
    if not sample_a or not sample_b:
        return None
    if len(sample_a) * len(sample_b) > 4_000_000:
        return percentile(sample_a, 0.5) - percentile(sample_b, 0.5)
    return percentile([a - b for a in sample_a for b in sample_b], 0.5)


# ---------- Reading several stages at once ----------
def holm_adjust(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni adjustment, in the order the p-values came in.

    Eleven stages are tested per comparison, and at the 5% line one of them
    clearing it by chance is likelier than not. Holm holds the probability of
    any false positive across the family at 5% while giving up less power than
    plain Bonferroni. Stages that could not be tested carry None and do not
    count toward the family.
    """
    testable = [(i, p) for i, p in enumerate(p_values) if p is not None]
    m = len(testable)
    adjusted: List[Optional[float]] = [None] * len(p_values)
    if not m:
        return adjusted

    running = 0.0
    for step, (index, p) in enumerate(sorted(testable, key=lambda pair: pair[1])):
        running = max(running, min(1.0, (m - step) * p))
        adjusted[index] = running
    return adjusted


def minimum_detectable_shift(differences: Sequence[float],
                             alpha: float = 0.05,
                             power: float = 0.80) -> Optional[float]:
    """The smallest true shift this many pairs would catch four times in five.

    Read off the spread of the differences that were actually observed, so it
    answers "was this run long enough" in the units of the metric rather than
    in the abstract. A measured difference far below this figure is not
    evidence of no change: it is a sample that could not have shown one.

    Stated for the mean shift; the signed-rank test needs some 2% more than
    this, which is well inside the precision of the estimate itself.
    """
    n = len(differences)
    if n < 3:
        return None
    sd = statistics.stdev(differences)
    if sd == 0:
        return 0.0
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    return (z_alpha + z_power) * sd / math.sqrt(n)


def compare_paired(sample_a: Dict[str, float], sample_b: Dict[str, float],
                   confidence: float = 0.95) -> PairedComparison:
    """Compare two runs item by item, over the items they have in common.

    sample_a and sample_b map an item name to that item's duration. Only the
    items present on both sides take part; a name missing on either side is
    dropped rather than matched by position, since position means nothing once
    a file has been added or has failed.
    """
    shared = [name for name in sample_a if name in sample_b]
    differences = [sample_a[name] - sample_b[name] for name in shared]

    if not differences:
        return PairedComparison(0, 0, None, Interval(None, None), None, None,
                                "no shared items", None, None, None,
                                note="the two sides have no item names in common")

    p_value, method, dropped = wilcoxon_signed_rank(differences)
    shift, interval = hodges_lehmann(differences, confidence)
    return PairedComparison(
        n_pairs=len(differences),
        n_zero=dropped,
        shift=shift,
        shift_interval=interval,
        p_value=p_value,
        p_adjusted=None,        # filled in once the whole family is known
        method=method,
        median_a=percentile([sample_a[name] for name in shared], 0.5),
        median_b=percentile([sample_b[name] for name in shared], 0.5),
        detectable_shift=minimum_detectable_shift(differences),
    )


def compare_unpaired(values_a: Sequence[float], values_b: Sequence[float]) -> UnpairedComparison:
    """Compare two sides that share no item names."""
    p_value, delta = mann_whitney(values_a, values_b)
    return UnpairedComparison(
        n_a=len(values_a),
        n_b=len(values_b),
        shift=two_sample_shift(values_a, values_b),
        p_value=p_value,
        p_adjusted=None,
        effect_size=delta,
        median_a=percentile(values_a, 0.5),
        median_b=percentile(values_b, 0.5),
    )
