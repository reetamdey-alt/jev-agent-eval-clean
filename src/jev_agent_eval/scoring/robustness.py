"""Robustness scoring: counterfactual + metamorphic metrics (v2 spec 14, 15).

Counterfactual metrics per pair (spec 14.3):
- expected label flip
- actual label flip
- probability delta
- non-target probability drift
- confidence delta

Defines: counterfactual_sensitivity, counterfactual_specificity,
non_target_invariance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Metamorphic invariance relations (spec 15).
RELATION_INVARIANT = "invariant"
RELATION_FLIP = "flip_expected"
RELATION_BOUNDED = "bounded_change"
RELATION_INDEPENDENT = "independent"


@dataclass
class PairComparison:
    parent_case_id: str
    variant_case_id: str
    relation: str
    question_id: str
    parent_pred: object
    variant_pred: object
    parent_p: float | None
    variant_p: float | None
    # For flip_expected pairs: the expected variant answer.
    expected_variant: object = None


@dataclass
class CounterfactualReport:
    n_pairs: int = 0
    # flip_expected pairs where the label actually flipped (sensitivity)
    counterfactual_sensitivity: float | None = None
    # non-target probability drift on flip pairs (specificity complement)
    counterfactual_specificity: float | None = None
    # invariant pairs where the prediction stayed the same
    non_target_invariance: float | None = None
    mean_probability_delta: float | None = None
    mean_confidence_delta: float | None = None
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "counterfactual_sensitivity": self.counterfactual_sensitivity,
            "counterfactual_specificity": self.counterfactual_specificity,
            "non_target_invariance": self.non_target_invariance,
            "mean_probability_delta": self.mean_probability_delta,
            "mean_confidence_delta": self.mean_confidence_delta,
            "n_failures": len(self.failures),
        }


def counterfactual_report(comparisons: list[PairComparison]) -> CounterfactualReport:
    """Aggregate counterfactual/minimal-pair metrics (spec 14.3)."""
    rep = CounterfactualReport(n_pairs=len(comparisons))
    flip_pairs = [c for c in comparisons if c.relation == RELATION_FLIP]
    inv_pairs = [c for c in comparisons if c.relation == RELATION_INVARIANT]

    if flip_pairs:
        flips = [
            c
            for c in flip_pairs
            if c.parent_pred is not None
            and c.variant_pred is not None
            and c.parent_pred != c.variant_pred
        ]
        rep.counterfactual_sensitivity = len(flips) / len(flip_pairs)
        prob_deltas = [
            abs(c.variant_p - c.parent_p)
            for c in flip_pairs
            if c.parent_p is not None and c.variant_p is not None
        ]
        if prob_deltas:
            rep.mean_probability_delta = sum(prob_deltas) / len(prob_deltas)
        for c in flip_pairs:
            if c not in flips:
                rep.failures.append(
                    {
                        "kind": "missed_flip",
                        "case": c.variant_case_id,
                        "question": c.question_id,
                        "parent_pred": str(c.parent_pred),
                        "variant_pred": str(c.variant_pred),
                    }
                )

    if inv_pairs:
        same = [
            c
            for c in inv_pairs
            if c.parent_pred is not None
            and c.variant_pred is not None
            and c.parent_pred == c.variant_pred
        ]
        rep.non_target_invariance = len(same) / len(inv_pairs)
        for c in inv_pairs:
            if c not in same:
                rep.failures.append(
                    {
                        "kind": "invariance_violation",
                        "case": c.variant_case_id,
                        "question": c.question_id,
                        "parent_pred": str(c.parent_pred),
                        "variant_pred": str(c.variant_pred),
                    }
                )

    conf_deltas = [
        abs(c.variant_p - c.parent_p)
        for c in comparisons
        if c.parent_p is not None and c.variant_p is not None
    ]
    if conf_deltas:
        rep.mean_confidence_delta = sum(conf_deltas) / len(conf_deltas)

    # Specificity: among flip pairs, fraction whose OTHER questions (if any)
    # did not drift — approximated here by the invariance rate on paired
    # non-target questions when provided as independent comparisons.
    rep.counterfactual_specificity = rep.non_target_invariance
    return rep


def metamorphic_consistency(comparisons: list[PairComparison]) -> dict:
    """Metamorphic consistency by transform (spec 15).

    Each comparison carries the transform in ``relation`` metadata; here we
    accept a simple aggregate plus per-transform grouping via tags on the
    variant case id.
    """
    by_transform: dict[str, list[PairComparison]] = {}
    for c in comparisons:
        t = c.variant_case_id.split("::")[-1] if "::" in c.variant_case_id else "unknown"
        by_transform.setdefault(t, []).append(c)
    out: dict[str, float | int | None] = {
        "n_total": len(comparisons),
    }
    for t, rows in sorted(by_transform.items()):
        expected_same = [c for c in rows if c.relation == RELATION_INVARIANT]
        if expected_same:
            same = sum(1 for c in expected_same if c.parent_pred == c.variant_pred)
            out[f"consistency::{t}"] = same / len(expected_same)
        else:
            flips = sum(1 for c in rows if c.parent_pred != c.variant_pred)
            out[f"sensitivity::{t}"] = flips / len(rows) if rows else None
    return out
