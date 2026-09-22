"""Selective prediction / risk-coverage scoring (v2 spec section 19).

JEV should not be forced into a confident decision when evidence is
insufficient. These metrics measure whether confidence can be used to
route uncertain decisions safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskCoveragePoint:
    threshold: float
    coverage: float
    selective_accuracy: float
    risk_at_coverage: float
    n_selected: int
    n_abstained: int


@dataclass
class SelectivePredictionReport:
    capability: str
    n_total: int
    abstention_rate: float
    points: list[RiskCoveragePoint] = field(default_factory=list)

    def utility_at_coverage(
        self, coverage_target: float, benefit: float = 1.0, cost_wrong: float = 1.0
    ) -> float | None:
        """Expected utility at the closest achievable coverage level."""
        best: RiskCoveragePoint | None = None
        for p in self.points:
            if p.coverage <= coverage_target:
                if best is None or p.coverage > best.coverage:
                    best = p
        if best is None:
            return None
        return best.selective_accuracy * benefit - (1 - best.selective_accuracy) * cost_wrong


def selective_prediction_report(
    capability: str,
    y_true: list[int],
    p_pred: list[float | None],
    thresholds: list[float] | None = None,
) -> SelectivePredictionReport:
    """Compute risk-coverage curve for binary decisions.

    Decisions with confidence below a threshold are 'abstained' (routed to a
    fallback). Coverage = fraction decided; selective accuracy = accuracy on
    decided items; risk = error rate on decided items.
    """
    assert len(y_true) == len(p_pred)
    n = len(y_true)
    if n == 0:
        return SelectivePredictionReport(capability=capability, n_total=0, abstention_rate=0.0)

    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(21)]  # 0.00 .. 1.00

    valid = [(t, p) for t, p in zip(y_true, p_pred, strict=False) if p is not None]
    points: list[RiskCoveragePoint] = []
    for thr in thresholds:
        selected = [(t, p) for t, p in valid if p >= thr]
        k = len(selected)
        if k == 0:
            points.append(RiskCoveragePoint(thr, 0.0, 0.0, 0.0, 0, len(valid)))
            continue
        correct = sum(1 for t, p in selected if (p >= 0.5) == bool(t))
        acc = correct / k
        points.append(
            RiskCoveragePoint(
                threshold=thr,
                coverage=k / len(valid) if valid else 0.0,
                selective_accuracy=acc,
                risk_at_coverage=1 - acc,
                n_selected=k,
                n_abstained=len(valid) - k,
            )
        )
    abstained = sum(1 for p in p_pred if p is None)
    return SelectivePredictionReport(
        capability=capability,
        n_total=n,
        abstention_rate=abstained / n,
        points=points,
    )


def risk_coverage_curves(
    scored: list[dict],
) -> dict[str, SelectivePredictionReport]:
    """Build risk-coverage curves grouped by capability (spec 19).

    ``scored`` rows must carry: capability, question_id, y_true, p_pred.
    """
    by_cap: dict[str, tuple[list[int], list[float | None]]] = {}
    for row in scored:
        cap = row.get("capability", "?")
        y = row.get("y_true")
        p = row.get("p_pred")
        if y is None:
            continue
        acc = by_cap.setdefault(cap, ([], []))
        acc[0].append(int(bool(y)))
        acc[1].append(p)
    return {
        cap: selective_prediction_report(cap, ys, ps) for cap, (ys, ps) in sorted(by_cap.items())
    }
