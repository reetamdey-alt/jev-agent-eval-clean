"""Unit tests for v2 scoring modules: selective, utility, robustness."""

from jev_agent_eval.scoring.robustness import (
    PairComparison,
    counterfactual_report,
    metamorphic_consistency,
)
from jev_agent_eval.scoring.selective import (
    risk_coverage_curves,
    selective_prediction_report,
)
from jev_agent_eval.scoring.utility import (
    CostMatrix,
    completion_utility,
    escalation_utility,
    safety_metrics,
    tool_gating_utility,
)


class TestSelectivePrediction:
    def test_perfect_model_full_coverage(self):
        y = [1, 0, 1, 0]
        p = [0.99, 0.01, 0.95, 0.05]
        rep = selective_prediction_report("tool_risk", y, p)
        # At threshold 0.0 all selected with perfect accuracy
        p0 = rep.points[0]
        assert p0.coverage == 1.0
        assert p0.selective_accuracy == 1.0

    def test_abstention_improves_selective_accuracy(self):
        y = [1, 0, 1, 0, 1]
        # one unconfident-correct and one confident item
        p = [0.9, 0.1, 0.9, 0.99, 0.9]
        rep = selective_prediction_report("cap", y, p)
        acc_05 = next(pt for pt in rep.points if pt.threshold == 0.5)
        # Low-confidence correct abstention keeps accuracy while lowering coverage
        assert acc_05.selective_accuracy == 0.75  # 1 wrong of 4 decided
        # A model whose errors are all low-confidence shows monotone benefit:
        y2 = [1, 1, 0, 0]
        p2 = [0.9, 0.9, 0.2, 0.2]
        rep2 = selective_prediction_report("cap", y2, p2)
        acc_lo = next(pt for pt in rep2.points if pt.threshold == 0.05)
        acc_hi = next(pt for pt in rep2.points if pt.threshold == 0.5)
        assert acc_lo.selective_accuracy == acc_hi.selective_accuracy == 1.0
        assert acc_hi.coverage == 0.5

    def test_none_probabilities_count_as_abstained(self):
        y = [1, 0]
        p = [None, 0.8]
        rep = selective_prediction_report("cap", y, p)
        assert rep.abstention_rate == 0.5

    def test_utility_at_coverage(self):
        y = [1, 1, 0, 0]
        p = [0.9, 0.8, 0.2, 0.1]
        rep = selective_prediction_report("cap", y, p)
        u = rep.utility_at_coverage(0.5)
        assert u is not None

    def test_risk_coverage_curves_grouping(self):
        rows = [
            {"capability": "a", "y_true": 1, "p_pred": 0.9},
            {"capability": "a", "y_true": 0, "p_pred": 0.1},
            {"capability": "b", "y_true": 1, "p_pred": 0.8},
        ]
        curves = risk_coverage_curves(rows)
        assert set(curves) == {"a", "b"}

    def test_empty(self):
        rep = selective_prediction_report("cap", [], [])
        assert rep.n_total == 0


class TestUtilityAndCosts:
    def test_expected_cost_ordering(self):
        # Unsafe allows must cost more than false alarms (spec 20)
        m = CostMatrix(false_positive=1.0, false_negative=10.0)
        wrong_fn = m.expected_cost([1], [0])  # missed risk
        wrong_fp = m.expected_cost([0], [1])  # unnecessary alarm
        assert wrong_fn > wrong_fp

    def test_safety_metrics_zero_false_allow_upper_bound(self):
        m = safety_metrics([1, 1, 0, 0], [1, 1, 0, 0])
        assert m.false_allow_rate == 0.0
        # rule of three: 3/n upper bound, never raw 0% claim
        assert m.false_allow_upper_bound_95 == 3.0 / 4

    def test_safety_metrics_rates(self):
        y = [1, 1, 0, 0, 1]
        p = [1, 0, 0, 1, 1]
        m = safety_metrics(y, p)
        assert m.false_allow_rate == 1 / 5
        assert m.unnecessary_confirmation_rate == 1 / 2

    def test_critical_risk_recall(self):
        y = [1, 1, 0]
        p = [1, 0, 0]
        levels = ["irreversible", "forbidden", "safe"]
        m = safety_metrics(y, p, risk_levels=levels)
        assert m.critical_risk_recall == 0.5

    def test_tool_gating_utility_rewards_prevention(self):
        u_good = tool_gating_utility([1, 0], [1, 0])
        u_bad = tool_gating_utility([1, 0], [0, 0])
        assert u_good > u_bad

    def test_escalation_utility_penalizes_overuse(self):
        u_needed = escalation_utility([1, 0], [1, 0])
        u_always = escalation_utility([1, 0], [1, 1])
        assert u_needed > u_always

    def test_completion_utility(self):
        u = completion_utility([1, 0], [1, 1])
        assert u < 0  # premature stop is penalized harder


class TestRobustness:
    def _pair(self, relation, parent_pred, variant_pred, pp=0.9, vp=0.9):
        return PairComparison(
            parent_case_id="p1",
            variant_case_id="v1::x" if relation == "invariant" else "v1",
            relation=relation,
            question_id="q",
            parent_pred=parent_pred,
            variant_pred=variant_pred,
            parent_p=pp,
            variant_p=vp,
        )

    def test_sensitivity_counts_flips(self):
        rep = counterfactual_report(
            [
                self._pair("flip_expected", 1, 0),
                self._pair("flip_expected", 1, 1),  # missed flip
            ]
        )
        assert rep.counterfactual_sensitivity == 0.5
        assert len(rep.failures) == 1

    def test_invariance(self):
        rep = counterfactual_report(
            [
                self._pair("invariant", 0, 0),
                self._pair("invariant", 1, 1),
                self._pair("invariant", 0, 1),  # violation
            ]
        )
        assert rep.non_target_invariance == 2 / 3
        assert rep.counterfactual_specificity == 2 / 3

    def test_probability_delta(self):
        rep = counterfactual_report(
            [
                self._pair("flip_expected", 1, 0, pp=0.9, vp=0.2),
            ]
        )
        assert rep.mean_probability_delta == 0.7

    def test_metamorphic_consistency_groups(self):
        rows = [
            self._pair("invariant", 0, 0),
            self._pair("invariant", 1, 0),  # violation
        ]
        out = metamorphic_consistency(rows)
        assert out["n_total"] == 2
        assert out["consistency::x"] == 0.5

    def test_empty(self):
        rep = counterfactual_report([])
        assert rep.n_pairs == 0
        assert rep.counterfactual_sensitivity is None
