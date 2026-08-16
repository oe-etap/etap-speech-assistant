#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 4: inter-rater reliability and calibration of automatic scores.

Nothing in the preceding tiers licenses a claim about response quality on its
own. The rubric tier produces judge scores over the whole set; this module is
what turns them into a defensible estimate, in two steps.

Reliability. Human ratings are only interpretable alongside the agreement
between raters, reported with statistics whose sampling behaviour is published:
Krippendorff's alpha (Krippendorff, 2018) for ordinal ratings with missing
cells, Cohen's (1960) and Fleiss's (1971) kappa for categorical decisions, and
the intraclass correlation for continuous ratings, specified and reported as
recommended by Shrout and Fleiss (1979) and Koo and Li (2016).

Gwet's AC1 (2008) is reported next to kappa, and for safety categories it is the
statistic to read. Safety items are overwhelmingly passes, and under that
skew kappa collapses towards zero even when raters agree on nearly every item -
the first of the two paradoxes described by Feinstein and Cicchetti (1990).
Reporting kappa alone on such a category would understate reliability badly
enough to invite the wrong conclusion.

Calibration. Prediction-powered inference (Angelopoulos et al., 2023; applied to
model-based evaluation by Boyeau et al., 2024) combines a small human-labelled
subsample with judge scores over the full set. It corrects the judge's bias using
the labelled pairs and returns a confidence interval that remains valid even if
the judge is systematically wrong. This is what allows an estimate computed from
mostly automatic scores to be reported as an estimate of the human quantity,
rather than as an estimate of what the judge thinks.
"""

from collections import Counter
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .stats import bootstrap_ci, kendall_tau_b, spearman


# Ratings keyed as item_id -> rater_id -> score.
RatingTable = Dict[str, Dict[str, float]]


def load_human_annotations(path: Path) -> Dict[str, RatingTable]:
    """Load human ratings from a long-format CSV.

    Expected columns: `item_id`, `rater_id`, `dimension`, `score`. Long format is
    required rather than a wide matrix because raters routinely score different
    subsets of items, and a wide sheet forces those gaps to be encoded as
    something that looks like data.

    Returns:
        dimension -> item_id -> rater_id -> score
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Human annotation file not found: {path}")

    tables: Dict[str, RatingTable] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"item_id", "rater_id", "dimension", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: missing column(s): {', '.join(sorted(missing))}. "
                f"Expected long format: item_id,rater_id,dimension,score")

        for row_no, row in enumerate(reader, 2):
            raw_score = (row.get("score") or "").strip()
            if not raw_score:
                continue
            try:
                score = float(raw_score)
            except ValueError:
                raise ValueError(f"{path}:{row_no}: non-numeric score "
                                 f"{raw_score!r}") from None
            dimension = (row["dimension"] or "").strip()
            item_id = (row["item_id"] or "").strip()
            rater_id = (row["rater_id"] or "").strip()
            tables.setdefault(dimension, {}).setdefault(item_id, {})[rater_id] = score
    return tables


# --------------------------------------------------------------------------
# Krippendorff's alpha
# --------------------------------------------------------------------------

def krippendorff_alpha(ratings: RatingTable,
                       level: str = "ordinal") -> Optional[float]:
    """Krippendorff's alpha for `ratings`, tolerating missing cells.

    Args:
        ratings: item_id -> rater_id -> score.
        level: "nominal", "ordinal" or "interval". Rubric scores are ordinal;
            using the interval metric on them assumes the distance from 1 to 2
            equals the distance from 4 to 5, which rubric anchors do not
            guarantee.

    Returns:
        Alpha, or None when fewer than two items carry at least two ratings.
    """
    units = [list(scores.values()) for scores in ratings.values()
             if len(scores) >= 2]
    if len(units) < 2:
        return None

    all_values = [value for unit in units for value in unit]
    n_total = len(all_values)
    if n_total < 2:
        return None

    counts = Counter(all_values)
    ordered_values = sorted(counts)
    delta = _distance_function(level, ordered_values, counts)

    observed = 0.0
    for unit in units:
        m = len(unit)
        pair_sum = sum(delta(unit[i], unit[j])
                       for i in range(m) for j in range(m) if i != j)
        observed += pair_sum / (m - 1)
    observed /= n_total

    expected = 0.0
    for value_a in ordered_values:
        for value_b in ordered_values:
            if value_a == value_b:
                pairs = counts[value_a] * (counts[value_a] - 1)
            else:
                pairs = counts[value_a] * counts[value_b]
            expected += pairs * delta(value_a, value_b)
    expected /= n_total * (n_total - 1)

    if expected == 0:
        # Every rating is identical, so there is no disagreement to explain and
        # no variance against which agreement could be judged.
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def _distance_function(level: str, ordered_values: Sequence[float],
                       counts: Counter):
    """Return the squared-difference function for the requested measurement level."""
    level = level.lower()

    if level == "nominal":
        return lambda a, b: 0.0 if a == b else 1.0

    if level == "interval":
        return lambda a, b: float((a - b) ** 2)

    if level != "ordinal":
        raise ValueError(f"Unsupported measurement level: {level!r}")

    index = {value: position for position, value in enumerate(ordered_values)}

    def ordinal(a: float, b: float) -> float:
        if a == b:
            return 0.0
        low, high = sorted((index[a], index[b]))
        total = sum(counts[ordered_values[g]] for g in range(low, high + 1))
        adjustment = (counts[ordered_values[low]]
                      + counts[ordered_values[high]]) / 2.0
        return float((total - adjustment) ** 2)

    return ordinal


# --------------------------------------------------------------------------
# Kappa family and Gwet's AC1
# --------------------------------------------------------------------------

def percent_agreement(ratings: RatingTable) -> Optional[float]:
    """Share of within-item rater pairs that agree exactly."""
    agree = pairs = 0
    for scores in ratings.values():
        values = list(scores.values())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pairs += 1
                agree += int(values[i] == values[j])
    return agree / pairs if pairs else None


def cohen_kappa(ratings: RatingTable) -> Optional[float]:
    """Cohen's kappa for exactly two raters over the items both rated."""
    rater_ids = sorted({rater for scores in ratings.values() for rater in scores})
    if len(rater_ids) != 2:
        return None
    first, second = rater_ids

    pairs = [(scores[first], scores[second]) for scores in ratings.values()
             if first in scores and second in scores]
    if len(pairs) < 2:
        return None

    categories = sorted({value for pair in pairs for value in pair})
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n

    marginal_a = Counter(a for a, _ in pairs)
    marginal_b = Counter(b for _, b in pairs)
    expected = sum((marginal_a[c] / n) * (marginal_b[c] / n) for c in categories)

    return (observed - expected) / (1 - expected) if expected < 1 else None


def fleiss_kappa(ratings: RatingTable) -> Optional[float]:
    """Fleiss's kappa for items rated by a fixed number of raters (at least two).

    Items are used only if they carry the modal number of ratings, since the
    statistic assumes a constant count per item. The number of items dropped is
    reported by `agreement_report` so the reduction is visible.
    """
    per_item = {item: list(scores.values()) for item, scores in ratings.items()
                if len(scores) >= 2}
    if len(per_item) < 2:
        return None

    modal_n = Counter(len(v) for v in per_item.values()).most_common(1)[0][0]
    usable = {item: values for item, values in per_item.items()
              if len(values) == modal_n}
    if len(usable) < 2 or modal_n < 2:
        return None

    categories = sorted({value for values in usable.values() for value in values})
    n_items = len(usable)

    p_j = {c: 0.0 for c in categories}
    agreement_sum = 0.0
    for values in usable.values():
        counts = Counter(values)
        for c in categories:
            p_j[c] += counts.get(c, 0)
        agreement_sum += (sum(count ** 2 for count in counts.values()) - modal_n)

    for c in categories:
        p_j[c] /= n_items * modal_n

    p_bar = agreement_sum / (n_items * modal_n * (modal_n - 1))
    p_e = sum(value ** 2 for value in p_j.values())
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else None


def gwet_ac1(ratings: RatingTable) -> Optional[float]:
    """Gwet's AC1 (2008) for any number of raters and missing cells.

    Chance agreement is estimated from the propensity to rate at random rather
    than from the observed marginals, which is why AC1 does not collapse on a
    skewed category the way kappa does.
    """
    units = [list(scores.values()) for scores in ratings.values()
             if len(scores) >= 2]
    if len(units) < 2:
        return None

    categories = sorted({value for unit in units for value in unit})
    q = len(categories)
    if q < 2:
        # A single observed category means perfect agreement with no chance
        # correction to apply.
        return 1.0

    n = len(units)
    observed = 0.0
    propensity = {c: 0.0 for c in categories}

    for unit in units:
        r = len(unit)
        counts = Counter(unit)
        observed += sum(counts[c] * (counts[c] - 1) for c in categories) / (r * (r - 1))
        for c in categories:
            propensity[c] += counts.get(c, 0) / r

    observed /= n
    for c in categories:
        propensity[c] /= n

    expected = sum(propensity[c] * (1 - propensity[c]) for c in categories) / (q - 1)
    return (observed - expected) / (1 - expected) if expected < 1 else None


# --------------------------------------------------------------------------
# Intraclass correlation
# --------------------------------------------------------------------------

@dataclass
class IccResult:
    """ICC(2,1) and ICC(2,k) with the sample they were computed on."""

    icc_single: Optional[float]
    icc_average: Optional[float]
    n_items: int
    n_raters: int
    dropped_items: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "icc_2_1": _r(self.icc_single),
            "icc_2_1_interpretation": interpret_icc(self.icc_single),
            "icc_2_k": _r(self.icc_average),
            "icc_n_items": self.n_items,
            "icc_n_raters": self.n_raters,
            "icc_dropped_items": self.dropped_items,
        }


def icc_two_way(ratings: RatingTable) -> Optional[IccResult]:
    """ICC(2,1) and ICC(2,k): two-way random effects, absolute agreement.

    Model 2 is the correct choice here because the raters are a sample of
    possible expert raters and their systematic leniency counts as disagreement,
    not as a fixed effect to be removed. ICC(2,1) describes the reliability of
    one rater; ICC(2,k) describes the mean of k raters, which is the quantity
    that matters when scores are averaged before analysis.

    Requires a complete matrix, so items not rated by every rater are dropped and
    counted.
    """
    rater_ids = sorted({rater for scores in ratings.values() for rater in scores})
    if len(rater_ids) < 2:
        return None

    rows = [[scores[rater] for rater in rater_ids] for scores in ratings.values()
            if all(rater in scores for rater in rater_ids)]
    dropped = len(ratings) - len(rows)
    if len(rows) < 2:
        return IccResult(None, None, len(rows), len(rater_ids), dropped)

    matrix = np.asarray(rows, dtype=float)
    n, k = matrix.shape
    grand_mean = matrix.mean()

    ss_total = float(((matrix - grand_mean) ** 2).sum())
    ss_rows = float(k * ((matrix.mean(axis=1) - grand_mean) ** 2).sum())
    ss_cols = float(n * ((matrix.mean(axis=0) - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    single_denominator = (ms_rows + (k - 1) * ms_error
                          + k * (ms_cols - ms_error) / n)
    average_denominator = ms_rows + (ms_cols - ms_error) / n

    single = ((ms_rows - ms_error) / single_denominator
              if single_denominator != 0 else None)
    average = ((ms_rows - ms_error) / average_denominator
               if average_denominator != 0 else None)

    return IccResult(single, average, n, k, dropped)


def interpret_icc(value: Optional[float]) -> str:
    """Label an ICC using the bands recommended by Koo and Li (2016)."""
    if value is None:
        return "undefined"
    if value < 0.5:
        return "poor"
    if value < 0.75:
        return "moderate"
    if value < 0.9:
        return "good"
    return "excellent"


def interpret_kappa(value: Optional[float]) -> str:
    """Label a kappa-family coefficient using the Landis and Koch (1977) bands."""
    if value is None:
        return "undefined"
    if value < 0:
        return "poor"
    if value <= 0.20:
        return "slight"
    if value <= 0.40:
        return "fair"
    if value <= 0.60:
        return "moderate"
    if value <= 0.80:
        return "substantial"
    return "almost perfect"


@dataclass
class AgreementReport:
    """Every reliability statistic for one rubric dimension."""

    dimension: str
    n_items: int
    n_raters: int
    n_items_multi_rated: int
    percent_agreement: Optional[float] = None
    alpha_ordinal: Optional[float] = None
    alpha_nominal: Optional[float] = None
    cohen_kappa: Optional[float] = None
    fleiss_kappa: Optional[float] = None
    gwet_ac1: Optional[float] = None
    icc: Optional[IccResult] = None

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "dimension": self.dimension,
            "n_items": self.n_items,
            "n_raters": self.n_raters,
            "n_items_multi_rated": self.n_items_multi_rated,
            "percent_agreement": _r(self.percent_agreement),
            "krippendorff_alpha_ordinal": _r(self.alpha_ordinal),
            "krippendorff_alpha_nominal": _r(self.alpha_nominal),
            "cohen_kappa": _r(self.cohen_kappa),
            "cohen_kappa_interpretation": interpret_kappa(self.cohen_kappa),
            "fleiss_kappa": _r(self.fleiss_kappa),
            "gwet_ac1": _r(self.gwet_ac1),
            "gwet_ac1_interpretation": interpret_kappa(self.gwet_ac1),
        }
        data.update(self.icc.as_dict() if self.icc else {})
        return data


def agreement_report(dimension: str, ratings: RatingTable) -> AgreementReport:
    """Compute the full reliability panel for one dimension.

    Every statistic is computed and reported rather than one being selected,
    because which one is appropriate depends on the distribution of the ratings,
    and that is visible only after the fact. The reader can then apply the
    correct statistic knowing the others; reporting only the most favourable one
    is the failure mode this is designed to prevent.
    """
    multi_rated = {item: scores for item, scores in ratings.items()
                   if len(scores) >= 2}
    rater_ids = {rater for scores in ratings.values() for rater in scores}

    return AgreementReport(
        dimension=dimension,
        n_items=len(ratings),
        n_raters=len(rater_ids),
        n_items_multi_rated=len(multi_rated),
        percent_agreement=percent_agreement(multi_rated),
        alpha_ordinal=krippendorff_alpha(multi_rated, "ordinal"),
        alpha_nominal=krippendorff_alpha(multi_rated, "nominal"),
        cohen_kappa=cohen_kappa(multi_rated),
        fleiss_kappa=fleiss_kappa(multi_rated),
        gwet_ac1=gwet_ac1(multi_rated),
        icc=icc_two_way(multi_rated))


# --------------------------------------------------------------------------
# Judge calibration and prediction-powered inference
# --------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """How well the judge tracks human ratings, and the corrected estimate."""

    dimension: str
    n_labelled: int
    n_unlabelled: int
    human_mean: Optional[float] = None
    human_ci: Tuple[Optional[float], Optional[float]] = (None, None)
    judge_mean_all: Optional[float] = None
    judge_mean_labelled: Optional[float] = None
    bias: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    spearman: Optional[float] = None
    kendall_tau: Optional[float] = None
    ppi_estimate: Optional[float] = None
    ppi_ci: Tuple[Optional[float], Optional[float]] = (None, None)
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "n_labelled": self.n_labelled,
            "n_unlabelled": self.n_unlabelled,
            "human_mean": _r(self.human_mean),
            "human_ci95_low": _r(self.human_ci[0]),
            "human_ci95_high": _r(self.human_ci[1]),
            "judge_mean_all_items": _r(self.judge_mean_all),
            "judge_mean_labelled_items": _r(self.judge_mean_labelled),
            "judge_bias_vs_human": _r(self.bias),
            "judge_mae_vs_human": _r(self.mean_absolute_error),
            "judge_human_spearman": _r(self.spearman),
            "judge_human_kendall_tau": _r(self.kendall_tau),
            "ppi_estimate": _r(self.ppi_estimate),
            "ppi_ci95_low": _r(self.ppi_ci[0]),
            "ppi_ci95_high": _r(self.ppi_ci[1]),
            "note": self.note,
        }


def calibrate_judge(dimension: str,
                    human_ratings: RatingTable,
                    judge_scores: Dict[str, float],
                    alpha: float = 0.05,
                    seed: int = 0) -> CalibrationResult:
    """Compare judge scores with human ratings and correct the judge's bias.

    Args:
        dimension: Rubric dimension being calibrated.
        human_ratings: item_id -> rater_id -> score, for the labelled subsample.
            Multiple raters per item are averaged first, since the target
            quantity is the mean expert rating.
        judge_scores: item_id -> judge score, for every item including the
            labelled ones.
        alpha: Significance level for the intervals.
        seed: Bootstrap seed for the human-only interval.

    Returns:
        A CalibrationResult carrying the rank correlation and error of the judge
        against humans, plus the prediction-powered estimate of the human mean
        over the full set with a valid interval.
    """
    human_means = {item: statistics.fmean(scores.values())
                   for item, scores in human_ratings.items() if scores}
    labelled = sorted(set(human_means) & set(judge_scores))
    unlabelled = sorted(set(judge_scores) - set(human_means))

    result = CalibrationResult(dimension=dimension,
                               n_labelled=len(labelled),
                               n_unlabelled=len(unlabelled))

    if not labelled:
        result.note = ("no item carries both a human rating and a judge score; "
                       "check that item ids match between the two sources")
        return result

    human_values = [human_means[item] for item in labelled]
    judge_labelled = [float(judge_scores[item]) for item in labelled]
    judge_all = [float(v) for v in judge_scores.values()]

    result.human_mean = statistics.fmean(human_values)
    result.human_ci = bootstrap_ci(human_values, alpha=alpha, seed=seed)
    result.judge_mean_all = statistics.fmean(judge_all) if judge_all else None
    result.judge_mean_labelled = statistics.fmean(judge_labelled)
    result.bias = result.judge_mean_labelled - result.human_mean
    result.mean_absolute_error = statistics.fmean(
        [abs(j - h) for j, h in zip(judge_labelled, human_values)])
    result.spearman = spearman(judge_labelled, human_values)
    result.kendall_tau = kendall_tau_b(judge_labelled, human_values)

    ppi = prediction_powered_mean(
        human_values=human_values,
        judge_values_labelled=judge_labelled,
        judge_values_unlabelled=[float(judge_scores[i]) for i in unlabelled],
        alpha=alpha)
    if ppi is None:
        result.note = ("too few labelled or unlabelled items for a "
                       "prediction-powered interval; report the human-only "
                       "estimate instead")
    else:
        result.ppi_estimate, result.ppi_ci = ppi
    return result


def prediction_powered_mean(human_values: Sequence[float],
                            judge_values_labelled: Sequence[float],
                            judge_values_unlabelled: Sequence[float],
                            alpha: float = 0.05
                            ) -> Optional[Tuple[float, Tuple[float, float]]]:
    """Prediction-powered estimate of the human mean over the full item set.

    Implements the rectified mean of Angelopoulos et al. (2023): the judge's mean
    over the unlabelled items, corrected by the mean judge-minus-human residual
    measured on the labelled items. The correction is what makes the estimate
    consistent for the human quantity regardless of how biased the judge is; the
    judge's contribution is only to reduce variance.

    Returns:
        (estimate, (ci_low, ci_high)) or None when either subset is too small.
    """
    n_labelled = len(human_values)
    n_unlabelled = len(judge_values_unlabelled)
    if n_labelled < 2 or n_unlabelled < 2:
        return None
    if n_labelled != len(judge_values_labelled):
        raise ValueError("labelled human and judge sequences must be aligned")

    judge_unlabelled = np.asarray(judge_values_unlabelled, dtype=float)
    residuals = (np.asarray(human_values, dtype=float)
                 - np.asarray(judge_values_labelled, dtype=float))

    estimate = float(judge_unlabelled.mean() + residuals.mean())
    variance = (judge_unlabelled.var(ddof=1) / n_unlabelled
                + residuals.var(ddof=1) / n_labelled)
    half_width = _normal_quantile(1 - alpha / 2) * math.sqrt(variance)
    return estimate, (estimate - half_width, estimate + half_width)


def _normal_quantile(p: float) -> float:
    """Standard normal quantile via bisection on the error function.

    Avoids a scipy dependency for the one quantile this module needs; the
    tolerance is far tighter than the precision at which intervals are reported.
    """
    low, high = -10.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if 0.5 * (1 + math.erf(middle / math.sqrt(2.0))) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else round(number, digits)


REFERENCE_KEYS = ["krippendorff", "cohen_kappa", "fleiss_kappa", "gwet_ac1",
                  "kappa_paradox", "icc_shrout", "icc_koo", "landis_koch",
                  "ppi", "autoeval"]
