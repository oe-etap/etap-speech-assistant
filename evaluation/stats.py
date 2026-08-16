#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptive and inferential statistics for evaluation scores.

Two choices here are deliberate and worth stating, because they change what a
result is allowed to claim.

Interval estimates are computed by the percentile bootstrap (Efron and
Tibshirani, 1993) rather than from a normal approximation. Rubric scores on a
1-5 scale over a few dozen items are bounded and often skewed, and a
mean-plus-1.96-standard-errors interval on such data can extend past the end of
the scale.

Configuration contrasts report an effect size next to any p-value, using Cliff's
delta (Cliff, 1993), which is ordinal and therefore appropriate for rubric
scores; and, where the claim is that two configurations are *equivalent* rather
than merely not significantly different, an equivalence test (TOST; Lakens,
2017). This distinction is the one that matters for a quantization comparison: a
non-significant difference on a small sample is not evidence of equivalence, and
only the equivalence test can support the statement that a compressed model is
no worse than its baseline within a stated margin.
"""

from dataclasses import dataclass
import math
import statistics
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                            # pragma: no cover - optional dep
    from scipy import stats as _scipy_stats
except Exception:                               # pragma: no cover - optional dep
    _scipy_stats = None


def _clean(values: Sequence[float]) -> List[float]:
    """Drop None and NaN so that a partially measured column still summarizes."""
    out = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        out.append(number)
    return out


@dataclass
class Summary:
    """Descriptive summary of one metric over a set of items."""

    n: int
    mean: Optional[float] = None
    sd: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "n": self.n, "mean": _r(self.mean), "sd": _r(self.sd),
            "median": _r(self.median), "q1": _r(self.q1), "q3": _r(self.q3),
            "min": _r(self.minimum), "max": _r(self.maximum),
            "ci95_low": _r(self.ci_low), "ci95_high": _r(self.ci_high),
        }


def describe(values: Sequence[float], n_boot: int = 5000,
             seed: int = 0) -> Summary:
    """Summarize a metric, with a bootstrap confidence interval for the mean."""
    data = _clean(values)
    if not data:
        return Summary(n=0)

    quantiles = np.percentile(data, [25, 50, 75])
    low, high = bootstrap_ci(data, statistics.fmean, n_boot=n_boot, seed=seed)
    return Summary(
        n=len(data),
        mean=statistics.fmean(data),
        sd=statistics.stdev(data) if len(data) > 1 else 0.0,
        median=float(quantiles[1]), q1=float(quantiles[0]), q3=float(quantiles[2]),
        minimum=min(data), maximum=max(data), ci_low=low, ci_high=high)


def bootstrap_ci(values: Sequence[float],
                 statistic: Callable[[Sequence[float]], float] = statistics.fmean,
                 n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap interval for `statistic` over `values`.

    Returns (None, None) for fewer than three observations, where a resampling
    interval carries no information beyond the observations themselves.
    """
    data = _clean(values)
    if len(data) < 3:
        return (None, None)

    rng = np.random.default_rng(seed)
    array = np.asarray(data, dtype=float)
    draws = rng.integers(0, len(array), size=(n_boot, len(array)))
    estimates = np.array([statistic(array[row]) for row in draws])
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def cliffs_delta(treatment: Sequence[float],
                 baseline: Sequence[float]) -> Optional[float]:
    """Cliff's delta (Cliff, 1993): P(treatment > baseline) - P(treatment < baseline).

    Ranges from -1 to 1 and needs no distributional assumption, which is why it is
    preferred here over a standardized mean difference on ordinal rubric scores.

    Counted pairwise for small inputs and from ranks for large ones. The pairwise
    form allocates an n*m matrix, which is unusable once a run holds thousands of
    items; the rank identity gives the same number in n log n time.
    """
    a = _clean(treatment)
    b = _clean(baseline)
    if not a or not b:
        return None

    if len(a) * len(b) <= 4_000_000:
        left = np.asarray(a)[:, None]
        right = np.asarray(b)[None, :]
        greater = int(np.sum(left > right))
        lesser = int(np.sum(left < right))
        return (greater - lesser) / (len(a) * len(b))

    # With average ranks over the combined sample, the rank sum of `a` gives the
    # Mann-Whitney statistic U = #(a>b) + 0.5*#(a==b), and delta = 2U/(nm) - 1.
    ranks = _rank(list(a) + list(b))
    rank_sum_a = sum(ranks[:len(a)])
    u_statistic = rank_sum_a - len(a) * (len(a) + 1) / 2.0
    return 2.0 * u_statistic / (len(a) * len(b)) - 1.0


def interpret_cliffs_delta(delta: Optional[float]) -> str:
    """Label a delta using the conventional negligible/small/medium/large bands."""
    if delta is None:
        return "undefined"
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


@dataclass
class EquivalenceResult:
    """Outcome of a two one-sided equivalence test."""

    margin: float
    mean_difference: float
    p_lower: float
    p_upper: float
    equivalent: bool
    alpha: float
    method: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "equivalence_margin": self.margin,
            "mean_difference": _r(self.mean_difference),
            "tost_p_lower": _r(self.p_lower, 5),
            "tost_p_upper": _r(self.p_upper, 5),
            "tost_p_max": _r(max(self.p_lower, self.p_upper), 5),
            "equivalent_within_margin": self.equivalent,
            "alpha": self.alpha,
            "tost_method": self.method,
        }


def tost_equivalence(treatment: Sequence[float], baseline: Sequence[float],
                     margin: float, alpha: float = 0.05
                     ) -> Optional[EquivalenceResult]:
    """Test whether two groups are equivalent within +/- `margin`.

    Two one-sided Welch t-tests, per Lakens (2017). Equivalence is concluded when
    both one-sided tests reject, that is when the whole confidence interval for
    the difference lies inside the margin.

    `margin` must be chosen and recorded before the data are seen; it states what
    difference in rubric points would be small enough not to matter in
    deployment. Choosing it afterwards turns the test into a formality.
    """
    a = _clean(treatment)
    b = _clean(baseline)
    if len(a) < 2 or len(b) < 2:
        return None

    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    standard_error = math.sqrt(var_a / len(a) + var_b / len(b))
    if standard_error == 0:
        equivalent = abs(mean_a - mean_b) < margin
        return EquivalenceResult(margin=margin, mean_difference=mean_a - mean_b,
                                 p_lower=0.0 if equivalent else 1.0,
                                 p_upper=0.0 if equivalent else 1.0,
                                 equivalent=equivalent, alpha=alpha,
                                 method="degenerate (zero variance)")

    difference = mean_a - mean_b
    t_lower = (difference + margin) / standard_error     # H0: difference <= -margin
    t_upper = (difference - margin) / standard_error     # H0: difference >= +margin

    numerator = (var_a / len(a) + var_b / len(b)) ** 2
    denominator = ((var_a / len(a)) ** 2 / (len(a) - 1)
                   + (var_b / len(b)) ** 2 / (len(b) - 1))
    df = numerator / denominator if denominator > 0 else len(a) + len(b) - 2

    if _scipy_stats is not None:
        p_lower = float(_scipy_stats.t.sf(t_lower, df))
        p_upper = float(_scipy_stats.t.cdf(t_upper, df))
        method = f"Welch TOST, t distribution, df={df:.1f}"
    else:
        p_lower = _normal_sf(t_lower)
        p_upper = 1.0 - _normal_sf(t_upper)
        method = "Welch TOST, normal approximation (scipy unavailable)"

    return EquivalenceResult(margin=margin, mean_difference=difference,
                             p_lower=p_lower, p_upper=p_upper,
                             equivalent=(p_lower < alpha and p_upper < alpha),
                             alpha=alpha, method=method)


def paired_mean_difference_ci(differences: Sequence[float], n_boot: int = 5000,
                              alpha: float = 0.05, seed: int = 0
                              ) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap interval for the mean of paired differences.

    Resampling the differences rather than the two runs separately is what makes
    the interval paired: every item contributes its own before-and-after pair, so
    the variation between items cancels instead of being counted as noise. On a
    shared prompt set that is a large gain in power.
    """
    data = _clean(differences)
    if len(data) < 3:
        return (None, None)

    rng = np.random.default_rng(seed)
    array = np.asarray(data, dtype=float)
    # Vectorised because a comparison scores every metric on every item, and the
    # per-row Python loop of the generic bootstrap becomes the dominant cost.
    draws = rng.integers(0, array.size, size=(n_boot, array.size))
    estimates = array[draws].mean(axis=1)
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


@dataclass
class PairedTestResult:
    """Outcome of a paired significance test on before-and-after measurements."""

    n_pairs: int
    n_effective: int          # Pairs that carry information (non-zero difference)
    statistic: Optional[float]
    p_value: Optional[float]
    method: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_pairs": self.n_pairs,
            "n_effective": self.n_effective,
            "statistic": _r(self.statistic),
            "p_value": _r(self.p_value, 6),
            "test_method": self.method,
        }


def wilcoxon_signed_rank(differences: Sequence[float]) -> Optional[PairedTestResult]:
    """Wilcoxon signed-rank test (Wilcoxon, 1945) on paired differences.

    The nonparametric counterpart of a paired t-test, used here because bounded
    rubric scores and rates are not normal and a mean can be dragged by a few
    items. Zero differences are dropped, which is the standard treatment, and
    ties share average ranks with the usual variance correction.

    A normal approximation is used for the p-value. It is accurate from roughly
    twenty informative pairs, which is far below the sample sizes this is meant
    for; below that the p-value is reported but should not carry a decision.
    """
    data = _clean(differences)
    if not data:
        return None

    nonzero = [value for value in data if value != 0.0]
    n = len(nonzero)
    if n == 0:
        return PairedTestResult(n_pairs=len(data), n_effective=0, statistic=0.0,
                                p_value=1.0,
                                method="Wilcoxon signed-rank (all differences zero)")

    ranks = _rank([abs(value) for value in nonzero])
    w_plus = sum(rank for value, rank in zip(nonzero, ranks) if value > 0)
    w_minus = sum(rank for value, rank in zip(nonzero, ranks) if value < 0)
    statistic = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0

    # Tie correction: groups of equal |difference| shrink the null variance.
    counts: Dict[float, int] = {}
    for value in nonzero:
        counts[abs(value)] = counts.get(abs(value), 0) + 1
    variance -= sum(t ** 3 - t for t in counts.values()) / 48.0

    if variance <= 0:
        return PairedTestResult(n_pairs=len(data), n_effective=n,
                                statistic=statistic, p_value=1.0,
                                method="Wilcoxon signed-rank (degenerate variance)")

    # Continuity correction, applied towards the mean.
    z = (w_plus - mean_w)
    z = (z - math.copysign(0.5, z)) / math.sqrt(variance)
    p_value = min(1.0, 2.0 * _normal_sf(abs(z)))
    return PairedTestResult(n_pairs=len(data), n_effective=n,
                            statistic=statistic, p_value=p_value,
                            method="Wilcoxon signed-rank, normal approximation")


def mcnemar(baseline: Sequence[bool],
            contrast: Sequence[bool]) -> Optional[PairedTestResult]:
    """McNemar's test (McNemar, 1947) for a paired binary outcome.

    The right test for "did this configuration change turn failures into passes",
    because it looks only at the items whose verdict changed. Items that pass in
    both runs, or fail in both, carry no information about the change and are
    excluded rather than inflating the sample.

    Uses the exact binomial test when few verdicts changed, and the
    continuity-corrected chi-square otherwise.
    """
    pairs = [(bool(a), bool(b)) for a, b in zip(baseline, contrast)
             if a is not None and b is not None]
    if not pairs:
        return None

    gained = sum(1 for before, after in pairs if not before and after)
    lost = sum(1 for before, after in pairs if before and not after)
    discordant = gained + lost

    if discordant == 0:
        return PairedTestResult(n_pairs=len(pairs), n_effective=0, statistic=0.0,
                                p_value=1.0,
                                method="McNemar (no verdict changed)")

    if discordant < 25:
        # Two-sided exact binomial probability of a split this lopsided.
        tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
        method = "McNemar, exact binomial"
        statistic = float(min(gained, lost))
    else:
        statistic = (abs(gained - lost) - 1) ** 2 / discordant
        p_value = _chi2_sf_1df(statistic)
        method = "McNemar, chi-square with continuity correction"

    return PairedTestResult(n_pairs=len(pairs), n_effective=discordant,
                            statistic=statistic, p_value=p_value, method=method)


def paired_tost(differences: Sequence[float], margin: float,
                alpha: float = 0.05) -> Optional[EquivalenceResult]:
    """Equivalence test on paired differences, within +/- `margin`.

    The paired counterpart of `tost_equivalence`, and the test that answers "the
    change made no practical difference". With a large sample almost any
    difference becomes detectable, so a significance test alone stops
    distinguishing a real improvement from a negligible one; this states in
    advance how large a difference would have to be to matter.
    """
    data = _clean(differences)
    if len(data) < 2:
        return None

    mean_difference = statistics.fmean(data)
    standard_error = math.sqrt(statistics.variance(data) / len(data))
    if standard_error == 0:
        equivalent = abs(mean_difference) < margin
        return EquivalenceResult(margin=margin, mean_difference=mean_difference,
                                 p_lower=0.0 if equivalent else 1.0,
                                 p_upper=0.0 if equivalent else 1.0,
                                 equivalent=equivalent, alpha=alpha,
                                 method="paired TOST (zero variance)")

    df = len(data) - 1
    t_lower = (mean_difference + margin) / standard_error
    t_upper = (mean_difference - margin) / standard_error

    if _scipy_stats is not None:
        p_lower = float(_scipy_stats.t.sf(t_lower, df))
        p_upper = float(_scipy_stats.t.cdf(t_upper, df))
        method = f"paired TOST, t distribution, df={df}"
    else:
        p_lower = _normal_sf(t_lower)
        p_upper = 1.0 - _normal_sf(t_upper)
        method = "paired TOST, normal approximation (scipy unavailable)"

    return EquivalenceResult(margin=margin, mean_difference=mean_difference,
                             p_lower=p_lower, p_upper=p_upper,
                             equivalent=(p_lower < alpha and p_upper < alpha),
                             alpha=alpha, method=method)


def _chi2_sf_1df(value: float) -> float:
    """Upper tail of the chi-square distribution with one degree of freedom."""
    if value <= 0:
        return 1.0
    return math.erfc(math.sqrt(value / 2.0))


def _normal_sf(z: float) -> float:
    """Upper tail of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def holm_bonferroni(p_values: Sequence[float],
                    alpha: float = 0.05) -> List[Dict[str, object]]:
    """Holm (1979) step-down adjustment for a family of related tests.

    Returned in the input order, each entry carrying the adjusted p-value and the
    decision, so that a table of contrasts can report both without reordering.
    """
    values = [(index, float(p)) for index, p in enumerate(p_values)]
    ordered = sorted(values, key=lambda pair: pair[1])
    total = len(ordered)

    adjusted: Dict[int, float] = {}
    running_max = 0.0
    for rank, (index, p) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p)
        running_max = max(running_max, candidate)
        adjusted[index] = running_max

    # Returned unrounded: a decisive test on a large sample yields a p-value far
    # below any decimal rounding, and rounding here would report it as zero.
    return [{"p_raw": p, "p_holm": adjusted[index],
             "reject": adjusted[index] < alpha}
            for index, p in values]


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation between paired values, with tie handling."""
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None
             and not (math.isnan(float(a)) or math.isnan(float(b)))]
    if len(pairs) < 3:
        return None
    left = _rank([p[0] for p in pairs])
    right = _rank([p[1] for p in pairs])
    return _pearson(left, right)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Kendall's tau-b, which is less sensitive than Spearman on short scales."""
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None
             and not (math.isnan(float(a)) or math.isnan(float(b)))]
    n = len(pairs)
    if n < 3:
        return None

    concordant = discordant = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if dx == 0:
                    tie_x += 1
                if dy == 0:
                    tie_y += 1

    total = n * (n - 1) / 2
    denominator = math.sqrt((total - tie_x) * (total - tie_y))
    return (concordant - discordant) / denominator if denominator > 0 else None


def _rank(values: Sequence[float]) -> List[float]:
    """Average ranks, so that tied rubric scores do not distort the correlation."""
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position
        while (end + 1 < len(ordered)
               and values[ordered[end + 1]] == values[ordered[position]]):
            end += 1
        mean_rank = (position + end) / 2.0 + 1.0
        for k in range(position, end + 1):
            ranks[ordered[k]] = mean_rank
        position = end + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x)
                            * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator if denominator > 0 else None


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else round(number, digits)


REFERENCE_KEYS = ["bootstrap", "cliffs_delta", "tost", "holm"]

# Cited only by the run-to-run comparison, so that a single-run report does not
# carry references to tests it never performed.
PAIRED_REFERENCE_KEYS = ["bootstrap", "cliffs_delta", "tost", "holm",
                         "wilcoxon", "mcnemar"]
