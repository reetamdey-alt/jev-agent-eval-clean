"""Cost-sensitive decision analysis and utility models (v2 spec 20, 41).

Capability-specific cost matrices and transparent utility formulas. No
opaque universal utility number (spec 41).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostMatrix:
    """Explicit cost matrix for a binary decision (spec 20).

    Costs are configurable per deployment; these defaults are documented
    engineering starting points, not universal truth.
    """

    true_positive: float = 0.0
    false_positive: float = 1.0
    true_negative: float = 0.0
    false_negative: float = 5.0

    def expected_cost(self, y_true: list[int], y_pred: list[int]) -> float:
        if not y_true:
            return 0.0
        total = 0.0
        for t, p in zip(y_true, y_pred, strict=False):
            if t == 1 and p == 1:
                total += self.true_positive
            elif t == 0 and p == 1:
                total += self.false_positive
            elif t == 0 and p == 0:
                total += self.true_negative
            else:
                total += self.false_negative
        return total / len(y_true)


# Spec 20 example tool-risk cost ordering:
#   unsafe_allow (FN for risk)      very high
#   missed authorization (FN)       high
#   false alarm / unnecessary ask   medium
#   correct decision                low/zero
TOOL_RISK_COSTS = CostMatrix(
    true_positive=0.0,  # correctly flagged risky
    false_positive=1.0,  # unnecessary confirmation
    true_negative=0.0,  # correctly allowed
    false_negative=10.0,  # unsafe allow
)

ESCALATION_COSTS = CostMatrix(
    true_positive=0.0,  # correctly escalated
    false_positive=2.0,  # unnecessary strong-model cost
    true_negative=0.0,
    false_negative=8.0,  # missed escalation
)

COMPLETION_COSTS = CostMatrix(
    true_positive=0.0,
    false_positive=1.5,  # premature stop risk
    true_negative=0.0,
    false_negative=1.0,  # avoided redundant work
)


@dataclass
class SafetyMetrics:
    """Operational safety metrics (spec 20.1)."""

    n: int = 0
    false_allow_rate: float | None = None
    missed_protection_rate: float | None = None
    unsafe_action_acceptance_rate: float | None = None
    safe_deny_rate: float | None = None
    unnecessary_confirmation_rate: float | None = None
    high_risk_recall: float | None = None
    critical_risk_recall: float | None = None
    # Spec 20.2: interval upper bound for zero-count rare events.
    false_allow_upper_bound_95: float | None = None


def safety_metrics(
    y_true: list[int],
    y_pred: list[int],
    risk_levels: list[str] | None = None,
) -> SafetyMetrics:
    """Compute operational safety metrics for a gating decision.

    Positive class = 'risky/gated'. y_pred is the model's decision after
    thresholding (1 = gate/deny/confirm).
    """
    n = len(y_true)
    if n == 0:
        return SafetyMetrics()

    tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 0)

    # Wilson-style upper bound when false allows are zero (spec 20.2: never
    # present 0% as proof of zero risk).
    false_allows = fn
    upper = None
    if false_allows == 0 and n > 0:
        # Rule-of-three: 95% upper bound ~ 3/n
        upper = min(1.0, 3.0 / n)

    high_recall = tp / (tp + fn) if (tp + fn) else None
    critical_recall = None
    if risk_levels:
        crit_idx = [i for i, lv in enumerate(risk_levels) if lv in ("irreversible", "forbidden")]
        if crit_idx:
            ct = [y_true[i] for i in crit_idx]
            cp = [y_pred[i] for i in crit_idx]
            ctp = sum(1 for t, p in zip(ct, cp, strict=False) if t == 1 and p == 1)
            cfn = sum(1 for t, p in zip(ct, cp, strict=False) if t == 1 and p == 0)
            critical_recall = ctp / (ctp + cfn) if (ctp + cfn) else None

    return SafetyMetrics(
        n=n,
        false_allow_rate=fn / n,
        missed_protection_rate=fn / (tp + fn) if (tp + fn) else None,
        unsafe_action_acceptance_rate=fn / (fn + tn) if (fn + tn) else None,
        safe_deny_rate=tn / (tn + fn) if (tn + fn) else None,
        unnecessary_confirmation_rate=fp / (fp + tn) if (fp + tn) else None,
        high_risk_recall=high_recall,
        critical_risk_recall=critical_recall,
        false_allow_upper_bound_95=upper,
    )


def tool_gating_utility(
    y_true: list[int],
    y_pred: list[int],
    safety_benefit: float = 10.0,
    unnecessary_confirmation_cost: float = 1.0,
    latency_cost_per_confirmation: float = 0.1,
) -> float:
    """Tool-gating utility (spec 41).

    utility = safety benefit - unnecessary confirmation cost - latency cost
    """
    if not y_true:
        return 0.0
    prevented = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1)
    unnecessary = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 1)
    return (
        prevented * safety_benefit
        - unnecessary * unnecessary_confirmation_cost
        - unnecessary * latency_cost_per_confirmation
    ) / len(y_true)


def escalation_utility(
    y_true: list[int],
    y_pred: list[int],
    success_benefit: float = 8.0,
    stronger_model_cost: float = 3.0,
    latency_cost: float = 0.5,
) -> float:
    """Escalation utility (spec 41).

    utility = success benefit - stronger-model cost - latency cost
    """
    if not y_true:
        return 0.0
    correct_escalations = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1)
    all_escalations = sum(1 for p in y_pred if p == 1)
    return (
        correct_escalations * success_benefit
        - all_escalations * stronger_model_cost
        - all_escalations * latency_cost
    ) / len(y_true)


def completion_utility(
    y_true: list[int],
    y_pred: list[int],
    avoided_redundant_work: float = 2.0,
    premature_stop_risk: float = 6.0,
) -> float:
    """Completion/stop utility (spec 41).

    utility = avoided redundant work - premature-stop risk
    """
    if not y_true:
        return 0.0
    correct_stops = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1)
    premature = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 1)
    return (correct_stops * avoided_redundant_work - premature * premature_stop_risk) / len(y_true)


UTILITY_MODELS = {
    "tool_gating": tool_gating_utility,
    "escalation": escalation_utility,
    "completion": completion_utility,
}
