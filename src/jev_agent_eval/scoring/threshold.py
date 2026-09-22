"""Threshold evaluation for binary decisions (spec sections 19, 52).

Thresholds belong to the consuming agent policy, not the evaluator: we sweep
0.00-1.00 in 0.01 steps and report the confusion matrix for each, so each
capability owner can justify their own operating point.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThresholdPoint:
    threshold: float
    precision: float | None
    recall: float | None
    fpr: float | None  # false positive rate
    fnr: float | None  # false negative rate
    decision_coverage: float  # fraction of cases with a probability
    predicted_positive_rate: float


def sweep_thresholds(
    y_true: list[int], p_pred: list[float | None], step: float = 0.01
) -> list[ThresholdPoint]:
    """Evaluate decision quality for thresholds from 0 to 1 inclusive."""
    points: list[ThresholdPoint] = []
    n = len(y_true)
    if n == 0:
        return points
    with_prob = [(t, p) for t, p in zip(y_true, p_pred, strict=False) if p is not None]
    coverage = len(with_prob) / n
    t = 0.0
    while t <= 1.0 + 1e-9:
        tp = fp = fn = tn = 0
        for gold, p in with_prob:
            pred = 1 if p >= t else 0
            if gold == 1 and pred == 1:
                tp += 1
            elif gold == 0 and pred == 1:
                fp += 1
            elif gold == 1 and pred == 0:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        fpr = fp / (fp + tn) if (fp + tn) else None
        fnr = fn / (tp + fn) if (tp + fn) else None
        points.append(
            ThresholdPoint(
                threshold=round(t, 2),
                precision=precision,
                recall=recall,
                fpr=fpr,
                fnr=fnr,
                decision_coverage=coverage,
                # Rate over DECIDED cases, matching the other metrics in
                # this point: cases with no probability are already covered
                # by decision_coverage.
                predicted_positive_rate=(tp + fp) / len(with_prob) if with_prob else 0.0,
            )
        )
        t += step
    return points


def policy_outcome(
    y_true: list[int], p_pred: list[float | None], threshold: float
) -> dict[str, float | None]:
    """Confusion matrix and operational consequences for one policy (spec 52)."""
    tp = fp = fn = tn = 0
    for gold, p in zip(y_true, p_pred, strict=False):
        if p is None:
            continue
        pred = 1 if p >= threshold else 0
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "unsafe_allow_rate": fn / total if total else None,
        "safe_deny_rate": fp / total if total else None,
    }
