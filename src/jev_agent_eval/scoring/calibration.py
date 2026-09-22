"""Calibration analysis (spec sections 18, 34).

ECE, adaptive ECE, reliability data, Brier, and negative log-likelihood.
Probability vectors are validated (normalized within epsilon) before use;
invalid vectors are reported, never silently normalized for the primary
score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_EPSILON = 1e-3


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    n: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float


@dataclass
class CalibrationReport:
    ece: float | None = None
    adaptive_ece: float | None = None
    brier: float | None = None
    nll: float | None = None
    reliability: list[CalibrationBin] = field(default_factory=list)
    n: int = 0
    n_invalid: int = 0


def validate_probability_vector(probs: dict[str, float], epsilon: float = DEFAULT_EPSILON) -> bool:
    """0 <= p_i <= 1 for all i and abs(sum - 1) <= epsilon (spec 34)."""
    if not probs:
        return False
    for p in probs.values():
        if not (0.0 <= p <= 1.0):
            return False
    return abs(sum(probs.values()) - 1.0) <= epsilon


def p_positive(answer: dict) -> float | None:
    """Extract p(positive) from a parsed noul answer per the pack contract.

    JEV semantics: for `noul`, the `noul` value and
    `probabilities["true"]` both represent the probability of the positive
    pole. Prefer the probability vector; fall back to the noul value.
    """
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and isinstance(probs.get("true"), (int, float)):
        return float(probs["true"])
    noul = answer.get("noul")
    if isinstance(noul, (int, float)):
        return float(noul)
    return None


def calibration_report(
    confidences: list[float],
    corrects: list[bool],
    *,
    bins: int = 10,
) -> CalibrationReport:
    """Reliability analysis over (confidence, correct) pairs.

    Works for binary (p_positive vs binary gold) and multiclass
    (max-probability vs exact-match) confidence.
    """
    report = CalibrationReport(n=len(confidences))
    if not confidences:
        return report
    if len(confidences) != len(corrects):
        raise ValueError(
            f"calibration_report: {len(confidences)} confidences but "
            f"{len(corrects)} correctness flags — pairs are desynced"
        )
    pairs = [(c, 1.0 if ok else 0.0) for c, ok in zip(confidences, corrects, strict=True)]
    # Out-of-range confidences are excluded from every bin yet would still
    # count in the denominator, deflating ECE. Exclude them from `total`
    # entirely and count them in n_invalid instead.
    valid_pairs = [pa for pa in pairs if 0.0 <= pa[0] <= 1.0]
    report.n_invalid = len(pairs) - len(valid_pairs)
    if not valid_pairs:
        return report
    pairs = valid_pairs
    total = len(pairs)
    report.n = total

    # Fixed-width ECE.
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [(c, a) for c, a in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not in_bin:
            continue
        mean_conf = sum(c for c, _ in in_bin) / len(in_bin)
        acc = sum(a for _, a in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(mean_conf - acc)
        report.reliability.append(
            CalibrationBin(
                lower=lo,
                upper=hi,
                n=len(in_bin),
                mean_confidence=round(mean_conf, 4),
                empirical_accuracy=round(acc, 4),
                calibration_gap=round(abs(mean_conf - acc), 4),
            )
        )
    report.ece = ece

    # Adaptive ECE: equal-mass bins where sample counts permit.
    if total >= 10 * bins:
        sorted_pairs = sorted(pairs)
        adaptive = 0.0
        chunk = total // bins
        for b in range(bins):
            part = (
                sorted_pairs[b * chunk : (b + 1) * chunk]
                if b < bins - 1
                else sorted_pairs[b * chunk :]
            )
            if not part:
                continue
            mean_conf = sum(c for c, _ in part) / len(part)
            acc = sum(a for _, a in part) / len(part)
            adaptive += (len(part) / total) * abs(mean_conf - acc)
        report.adaptive_ece = adaptive

    report.brier = sum((c - a) ** 2 for c, a in pairs) / total
    report.nll = (
        sum(-math.log(min(max(c if a else 1 - c, 1e-15), 1 - 1e-15)) for c, a in pairs) / total
    )
    return report
