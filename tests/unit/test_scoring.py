"""Unit tests for scoring modules (spec 17, 18, 19, 31)."""

from __future__ import annotations

import math

import pytest

from jev_agent_eval.scoring import score_capability
from jev_agent_eval.scoring.calibration import calibration_report, validate_probability_vector
from jev_agent_eval.scoring.categorical import binary_scores, multiclass_scores
from jev_agent_eval.scoring.ordinal import score_scores
from jev_agent_eval.scoring.threshold import policy_outcome, sweep_thresholds
from jev_agent_eval.scoring.uncertainty import (
    bootstrap_proportion_ci,
    cluster_bootstrap_proportion_ci,
    paired_bootstrap_delta_ci,
    paired_cluster_bootstrap_delta_ci,
    wilson_interval,
)


class TestBinaryScoring:
    def test_perfect(self):
        m = binary_scores([1, 0, 1, 0], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        assert m.accuracy == 1.0
        assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
        assert m.brier is not None and m.brier < 0.1

    def test_all_wrong(self):
        m = binary_scores([1, 0], [0, 1], [0.9, 0.1])
        assert m.accuracy == 0.0

    def test_empty(self):
        # Nothing scored is undefined (None), not a fabricated 0.0.
        m = binary_scores([], [], [])
        assert m.n == 0 and m.accuracy is None and m.base_rate is None

    def test_no_probabilities(self):
        m = binary_scores([1, 0], [1, 0], [None, None])
        assert m.accuracy == 1.0
        assert m.brier is None and m.roc_auc is None

    def test_log_loss(self):
        m = binary_scores([1], [1], [0.999999])
        assert m.log_loss is not None and m.log_loss < 0.001

    def test_auc_perfect_ranking(self):
        m = binary_scores([1, 1, 0, 0], [1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1])
        assert m.roc_auc == pytest.approx(1.0)

    def test_auc_inverted(self):
        m = binary_scores([1, 0], [0, 1], [0.1, 0.9])
        assert m.roc_auc == pytest.approx(0.0)

    def test_class_imbalance_visible(self):
        # 90% negatives: accuracy can be high while recall is terrible.
        y_true = [0] * 90 + [1] * 10
        y_pred = [0] * 100
        m = binary_scores(y_true, y_pred, [None] * 100)
        assert m.accuracy == pytest.approx(0.9)
        assert m.recall == 0.0
        assert m.base_rate == pytest.approx(0.1)


class TestMulticlassScoring:
    def test_exact_accuracy(self):
        m = multiclass_scores(
            ["a", "b", "c"],
            ["a", "b", "c"],
            [{"a": 0.7, "b": 0.2, "c": 0.1}] * 3,
        )
        assert m.accuracy == 1.0
        assert m.macro_f1 == pytest.approx(1.0)

    def test_confusion_matrix(self):
        m = multiclass_scores(["a", "a", "b"], ["a", "b", "b"], [None] * 3)
        assert m.confusion["a"]["b"] == 1
        assert m.confusion["b"]["b"] == 1

    def test_top2(self):
        m = multiclass_scores(
            ["a", "b", "c"],
            ["a", "b", "c"],
            [
                {"a": 0.5, "b": 0.3, "c": 0.2},
                {"a": 0.2, "b": 0.5, "c": 0.3},
                {"a": 0.1, "b": 0.45, "c": 0.45},
            ],
        )
        assert m.top2_accuracy is not None
        assert m.top2_accuracy == pytest.approx(1.0)  # gold always in top-2

    def test_per_class_metrics(self):
        m = multiclass_scores(["a", "b", "a"], ["a", "b", "b"], [None] * 3)
        assert m.per_class["a"]["recall"] == pytest.approx(0.5)


class TestOrdinal:
    def test_mae_rmse(self):
        m = score_scores([1.0, 2.0], [2.0, 4.0])
        assert m.mae == pytest.approx(1.5)
        assert m.rmse == pytest.approx(math.sqrt(2.5))

    def test_spearman_perfect(self):
        m = score_scores([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
        assert m.spearman == pytest.approx(1.0)

    def test_within_one_level(self):
        # Spacing is derived from distinct gold levels (unit-spaced here).
        m = score_scores([1.0, 2.0], [1.4, 3.6])
        assert m.within_one_level == pytest.approx(0.5)

    def test_within_one_level_scale_aware(self):
        # On levels {0, 5, 10}, an error of 5 is within one level: the
        # hardcoded 1.0 threshold would have scored 0.0 here.
        assert score_scores([10.0, 5.0], [5.0, 5.0]).within_one_level == pytest.approx(1.0)
        assert score_scores([0.0, 5.0], [6.0, 11.0]).within_one_level == pytest.approx(0.0)

    def test_within_one_level_undefined_single_level(self):
        # With a single distinct gold level the spacing is undefined.
        assert score_scores([1.0, 1.0], [1.4, 2.6]).within_one_level is None


class TestCalibration:
    def test_perfectly_calibrated(self):
        # 10 predictions at 0.8 confidence, 8 correct -> gap ~0.
        report = calibration_report([0.8] * 10, [True] * 8 + [False] * 2, bins=10)
        assert report.ece is not None and report.ece < 0.01
        assert report.brier == pytest.approx(0.16, abs=0.01)

    def test_overconfident(self):
        # Confident but mostly wrong -> large ECE.
        report = calibration_report([0.95] * 10, [False] * 9 + [True], bins=10)
        assert report.ece is not None and report.ece > 0.7

    def test_reliability_structure(self):
        report = calibration_report(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
            [False, True, False, True, False, True, False, True, False, True],
            bins=10,
        )
        assert len(report.reliability) == 10
        for b in report.reliability:
            assert b.n == 1
            assert 0.0 <= b.calibration_gap <= 1.0

    def test_probability_vector_validation(self):
        assert validate_probability_vector({"a": 0.5, "b": 0.5})
        assert not validate_probability_vector({"a": 0.5, "b": 0.4})  # sums 0.9
        assert not validate_probability_vector({"a": 1.5})  # out of range
        assert not validate_probability_vector({})  # empty

    def test_epsilon_configurable(self):
        assert validate_probability_vector({"a": 0.5, "b": 0.505}, epsilon=0.01)
        assert not validate_probability_vector({"a": 0.5, "b": 0.505}, epsilon=0.001)


class TestThresholds:
    def test_sweep_length(self):
        points = sweep_thresholds([1, 0], [0.9, 0.1])
        assert len(points) == 101  # 0.00 to 1.00 inclusive

    def test_low_threshold_catches_all(self):
        points = {p.threshold: p for p in sweep_thresholds([1, 0], [0.9, 0.1])}
        assert points[0.0].recall == 1.0

    def test_policy_outcome(self):
        out = policy_outcome([1, 0, 1], [0.99, 0.01, 0.5], 0.9)
        assert out["tp"] == 1 and out["fn"] == 1
        assert out["unsafe_allow_rate"] == pytest.approx(1 / 3)


class TestUncertainty:
    def test_bootstrap_ci_covers_point(self):
        values = [True] * 7 + [False] * 3
        ci = bootstrap_proportion_ci(values, samples=500, seed=42)
        assert ci is not None
        assert ci.low <= 0.7 <= ci.high

    def test_bootstrap_deterministic(self):
        values = [True, False, True, True]
        ci1 = bootstrap_proportion_ci(values, samples=200, seed=7)
        ci2 = bootstrap_proportion_ci(values, samples=200, seed=7)
        assert ci1.low == ci2.low and ci1.high == ci2.high

    def test_paired_bootstrap(self):
        # Delta is B - A (positive = run B improved), matching the
        # accuracy_delta convention in run comparison reports.
        a = [True] * 4 + [False] * 6
        b = [True] * 8 + [False] * 2
        result = paired_bootstrap_delta_ci(a, b, samples=500, seed=1)
        assert result is not None
        delta, ci = result
        assert delta == pytest.approx(0.4)
        assert ci.low < delta < ci.high

    def test_wilson(self):
        ci = wilson_interval(8, 10)
        assert ci is not None
        assert ci.low < 0.8 < ci.high
        assert wilson_interval(0, 0) is None

    def test_cluster_bootstrap_ci_contains_per_row_point(self):
        # 24 units all-correct + 1 unit with 2/3 correct repeats: the
        # per-row point estimate is 74/75. A collapsed-majority CI of
        # [1.0, 1.0] would exclude the reported accuracy — the cluster
        # bootstrap must contain it.
        clusters = [[True] * 3] * 24 + [[True, True, False]]
        ci = cluster_bootstrap_proportion_ci(clusters, samples=2000, seed=42)
        assert ci is not None
        assert ci.low <= 74 / 75 <= ci.high
        assert ci.method == "cluster_bootstrap"

    def test_cluster_bootstrap_repeats_do_not_shrink_ci(self):
        # Triplicating every observation (repeats=3 of a deterministic
        # model) must NOT tighten the CI relative to the single-run CI:
        # repeats are not independent samples.
        units = [[True]] * 6 + [[False]] * 4
        ci_single = cluster_bootstrap_proportion_ci(units, samples=2000, seed=7)
        # same 10 units, each observed 3 times with identical outcomes
        units3 = [c * 3 for c in units]
        ci_triple = cluster_bootstrap_proportion_ci(units3, samples=2000, seed=7)
        assert ci_single is not None and ci_triple is not None
        # Resampled cluster counts grow 3x per draw; the resulting CI
        # width must be within ~1% of the single-run width (identical
        # distribution over units -> essentially the same CI).
        w_single = ci_single.high - ci_single.low
        w_triple = ci_triple.high - ci_triple.low
        assert w_triple <= w_single * 1.01 + 1e-9

    def test_paired_cluster_bootstrap_delta(self):
        # 10 paired units, each observed twice; run B is better on 3 units.
        ca = [[True, True]] * 7 + [[False, False]] * 3
        cb = [[True, True]] * 10
        result = paired_cluster_bootstrap_delta_ci(ca, cb, samples=1000, seed=3)
        assert result is not None
        delta, ci = result
        assert delta == pytest.approx(0.3)
        assert ci.low <= 0.3 <= ci.high

    def test_paired_cluster_bootstrap_mismatched_units(self):
        ca = [[True]] * 4
        cb = [[True]] * 5
        assert paired_cluster_bootstrap_delta_ci(ca, cb, samples=10, seed=1) is None


class TestAuditRegressions:
    """Regression tests for audit-confirmed scoring bugs."""

    def test_multiclass_none_prediction_no_crash(self):
        # A choice answer of null must not crash the report build; it
        # counts as an incorrect prediction excluded from class tallies.
        m = multiclass_scores(["a"], [None], [None])
        assert m.n == 1 and m.accuracy == 0.0 and m.classes == ["a"]

    def test_f1_zero_when_tp_zero(self):
        # tp=0 with defined precision/recall is F1=0.0, not None.
        b = binary_scores([1, 1, 0], [0, 0, 1], [0.2, 0.3, 0.6])
        assert b.precision == 0.0 and b.recall == 0.0 and b.f1 == 0.0

    def test_macro_f1_includes_never_correct_classes(self):
        # A class the model never gets right contributes 0.0 to macro-F1
        # instead of being dropped (which would inflate the score).
        m = multiclass_scores(["a", "b", "b"], ["b", "b", "b"], [None, None, None])
        assert m.macro_f1 == pytest.approx(0.4)

    def test_pr_auc_tie_invariance(self):
        from jev_agent_eval.scoring.categorical import _pr_auc

        # Tied scores must yield the same AUC regardless of input order.
        assert _pr_auc([1, 0], [0.5, 0.5]) == _pr_auc([0, 1], [0.5, 0.5]) == 0.5

    def test_wilson_level(self):
        ci90 = wilson_interval(8, 10, level=0.90)
        ci95 = wilson_interval(8, 10, level=0.95)
        # The 90% interval must be narrower than 95%, and match level 0.90.
        assert ci90 is not None and ci95 is not None
        assert (ci90.high - ci90.low) < (ci95.high - ci95.low)
        assert ci90.low == pytest.approx(0.541, abs=0.01)

    def test_calibration_desync_raises(self):
        # Mismatched confidence/correct list lengths must fail loudly.
        with pytest.raises(ValueError):
            calibration_report([0.8, 0.9], [True])

    def test_calibration_out_of_range_counted_invalid(self):
        report = calibration_report([0.8, 1.5], [True, True])
        assert report.n == 1
        assert report.n_invalid == 1
        assert report.ece == pytest.approx(0.2)

    def test_predicted_positive_rate_over_decided(self):
        # Cases without a probability are excluded from the rate.
        points = sweep_thresholds([1, 0], [0.9, None])
        assert points[0].decision_coverage == pytest.approx(0.5)
        assert points[0].predicted_positive_rate == pytest.approx(1.0)

    def test_mixed_type_question_id_split_into_separate_entries(self):
        # Regression (live nightly run): "claim_supported" is a noul question
        # in the synthetic claim family and a choice question in the
        # contradiction family. Pooling both under one slot mixed int golds
        # with str golds and crashed multiclass_scores' sorted() at report
        # time. The two families must produce separate metric entries.
        r_noul = _make_case_result(
            "claim_supported",
            {"type": "noul", "noul": 1, "probabilities": {"true": 0.9, "false": 0.1}},
            1,
            capability="claim_evidence",
        )
        r_choice = _make_case_result(
            "claim_supported",
            {"type": "choice", "choice": "contradicted", "probabilities": None},
            "contradicted",
            capability="claim_evidence",
        )
        m = score_capability([r_noul, r_choice])
        qs = m["questions"]
        assert set(qs) == {"claim_supported:noul", "claim_supported:choice"}
        assert qs["claim_supported:noul"]["type"] == "noul"
        assert qs["claim_supported:noul"]["accuracy"] == 1.0
        assert qs["claim_supported:choice"]["type"] == "choice"
        assert qs["claim_supported:choice"]["accuracy"] == 1.0
        # Both scored questions count toward the pooled accuracy.
        assert m["n_scored_questions"] == 2
        assert m["accuracy"] == 1.0

    def test_single_type_question_keeps_bare_id(self):
        # A question id used with one type only must keep its bare id in
        # the metrics (existing consumers depend on questions["q"]).
        r = _make_case_result("q", {"type": "noul", "noul": 1}, 1)
        m = score_capability([r])
        assert set(m["questions"]) == {"q"}


def _make_case_result(
    qid: str, pred: dict, gold_value, capability: str = "cap", gold_type: str | None = None
):
    from jev_agent_eval.schemas.case import GoldAnswer, GoldProvenance
    from jev_agent_eval.schemas.result import CaseResult, CaseStatus

    cr = CaseResult(
        case_id=f"case-{qid}",
        dataset="test",
        dataset_version="1",
        capability=capability,
        status=CaseStatus.SUCCESS,
        question_ids=[qid],
        predictions={qid: pred},
    )
    object.__setattr__(
        cr,
        "gold_snapshot",
        {
            qid: GoldAnswer(
                type=gold_type or pred.get("type"),
                value=gold_value,
                provenance=GoldProvenance.SYNTHETIC_DETERMINISTIC,
            )
        },
    )
    return cr


class TestScoreCapabilityAlignment:
    def test_unanswerable_noul_excluded_not_misaligned(self):
        # A noul answer with neither a value nor probabilities must not
        # enter binary metrics: previously y_true grew without y_pred and
        # every subsequent pairing was off by one.
        r1 = _make_case_result("q", {"type": "noul", "noul": None, "probabilities": None}, 0)
        r2 = _make_case_result(
            "q", {"type": "noul", "noul": 1, "probabilities": {"true": 1.0, "false": 0.0}}, 1
        )
        m = score_capability([r1, r2])
        q = m["questions"]["q"]
        assert q["n"] == 1
        assert q["accuracy"] == 1.0
        assert q["base_rate"] == 1.0

    def test_null_choice_labeled_incorrect_not_crash(self):
        # choice=None counts in n as an incorrect prediction; the report
        # must still build (previously TypeError in sorted()).
        r1 = _make_case_result("q", {"type": "choice", "choice": None}, "a")
        r2 = _make_case_result(
            "q", {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.2}}, "a"
        )
        m = score_capability([r1, r2])
        q = m["questions"]["q"]
        assert q["n"] == 2
        assert q["accuracy"] == 0.5
        assert None not in q["confusion"]

    def test_noul_out_of_range_probability_counted(self):
        # p_positive=1.5 corrupts Brier/NLL; it must be excluded from
        # calibration and counted in n_invalid_probability.
        r = _make_case_result(
            "q", {"type": "noul", "noul": 1, "probabilities": {"true": 1.5, "false": -0.5}}, 1
        )
        m = score_capability([r])
        q = m["questions"]["q"]
        assert q["accuracy"] == 1.0  # still scored via the label
        assert q["brier"] is None
        assert m["calibration"]["n_invalid_probability"] == 1
        assert m["calibration"]["n"] == 0

    def test_calibration_pairs_stay_aligned(self):
        # Confidences and correctness are appended atomically; a case
        # without a valid probability vector must not shift subsequent
        # pairs (previously choice confidence lists desynced).
        r1 = _make_case_result("q", {"type": "choice", "choice": "b"}, "b")  # no vector
        r2 = _make_case_result(
            "q", {"type": "choice", "choice": "b", "probabilities": {"a": 0.05, "b": 0.95}}, "b"
        )  # wrong gold? no: gold b, pred b
        m = score_capability([r1, r2])
        assert m["calibration"]["n"] == 1
        assert m["calibration"]["reliability"], "one bin must exist"

    def test_noul_label_probability_agree(self):
        # When noul and probabilities["true"] disagree, the probability
        # wins for both label and score — one consistent prediction.
        r = _make_case_result(
            "q", {"type": "noul", "noul": 0, "probabilities": {"true": 0.9, "false": 0.1}}, 1
        )
        m = score_capability([r])
        q = m["questions"]["q"]
        assert q["accuracy"] == 1.0
        assert q["brier"] == pytest.approx(0.01, abs=1e-9)

    def test_unknown_answer_type_categorized_not_dropped(self):
        # A garbage answer type must land in the failure taxonomy with no
        # scored questions — never a clean zero-question 'success'.
        r = _make_case_result("q", {"type": "weird", "noul": 0.5}, 1)
        m = score_capability([r])
        assert m["failures"]["unknown_answer_type"] == 1
        assert m["n_scored_questions"] == 0
        assert m["accuracy"] is None

    def test_answer_type_mismatch_categorized(self):
        # A 'score'-typed answer to a noul question must not be scored as
        # ordinal against the coerced binary gold.
        r = _make_case_result("q", {"type": "score", "score": 4.0}, 1, gold_type="noul")
        m = score_capability([r])
        assert m["failures"]["answer_type_mismatch"] == 1
        assert m["n_scored_questions"] == 0

    def test_unexpected_answer_qid_categorized(self):
        # A corrupted or adversarial response answering a question the case
        # never asked must not create phantom metric entries (previously it
        # produced questions.ghost-question with its own accuracy row).
        from jev_agent_eval.schemas.case import GoldAnswer, GoldProvenance
        from jev_agent_eval.schemas.result import CaseResult, CaseStatus

        cr = CaseResult(
            case_id="case-x",
            dataset="t",
            dataset_version="1",
            capability="cap",
            status=CaseStatus.SUCCESS,
            question_ids=["q"],
            predictions={
                "q": {"type": "noul", "noul": 1, "probabilities": {"true": 0.9, "false": 0.1}},
                "ghost-question": {
                    "type": "noul",
                    "noul": 1,
                    "probabilities": {"true": 1.0, "false": 0.0},
                },
            },
        )
        object.__setattr__(
            cr,
            "gold_snapshot",
            {
                "q": GoldAnswer(
                    type="noul", value=1, provenance=GoldProvenance.SYNTHETIC_DETERMINISTIC
                )
            },
        )
        m = score_capability([cr])
        assert "ghost-question" not in m["questions"], "phantom question leaked into metrics"
        assert m["failures"]["unexpected_answer"] == 1
        # The real question is still scored normally.
        assert m["questions"]["q"]["n"] == 1

    def test_scoring_categories_propagate_to_case_result(self):
        # Metrics failures and errors.jsonl categories must agree: the
        # taxonomy found at scoring time is written back onto the
        # CaseResult so triage artifacts carry it (previously errors.jsonl
        # showed categories: [] next to failures.unknown_answer_type = 1).
        r = _make_case_result("q", {"type": "weird", "noul": 0.5}, 1)
        score_capability([r])
        assert "schema_error" in [c.value for c in r.failure_categories]

    def test_missing_answer_categorized_when_no_label_no_probability(self):
        # Neither probability nor label on a noul answer: the question must
        # be counted as missing_answer, not silently vanish.
        r = _make_case_result("q", {"type": "noul", "noul": None, "probabilities": None}, 1)
        m = score_capability([r])
        assert m["failures"]["missing_answer"] == 1
        assert m["n_scored_questions"] == 0
        assert "missing_answer" in [c.value for c in r.failure_categories]


class TestGarbageTypeDoesNotZeroQuestion:
    """Regression: one garbage-typed answer recorded its type into
    slot["types"] before validation, and the aggregation pass picked
    first-seen type to route the scorer — so a single corrupt answer
    routed the whole question into NO scorer, zeroing accuracy for every
    clean answer to that question (23 valid answers silently unscored)."""

    def test_garbage_first_does_not_zero_clean_answers(self):
        # A 'wat'-typed answer arrives FIRST; 3 clean noul answers follow.
        rs = [
            _make_case_result("q", {"type": "wat", "noul": 0.5}, 1),
            _make_case_result(
                "q", {"type": "noul", "noul": 1.0, "probabilities": {"true": 1.0, "false": 0.0}}, 1
            ),
            _make_case_result(
                "q", {"type": "noul", "noul": 1.0, "probabilities": {"true": 1.0, "false": 0.0}}, 1
            ),
            _make_case_result(
                "q", {"type": "noul", "noul": 0.0, "probabilities": {"true": 0.0, "false": 1.0}}, 0
            ),
        ]
        for i, r in enumerate(rs):
            object.__setattr__(r, "case_id", f"case-{i}")
        m = score_capability(rs)
        assert m["failures"]["unknown_answer_type"] == 1
        # The 3 clean answers are still scored.
        assert m["n_scored_questions"] == 3
        assert m["questions"]["q"]["n"] == 3
        assert m["accuracy"] == 1.0

    def test_mismatched_type_does_not_hijack_routing(self):
        # A 'score' answer to a noul question (declared-mismatch) must not
        # route the question away from the noul scorer for later cases.
        rs = [
            _make_case_result("q", {"type": "score", "score": 0.5}, 1, gold_type="noul"),
            _make_case_result(
                "q",
                {"type": "noul", "noul": 1.0, "probabilities": {"true": 1.0, "false": 0.0}},
                1,
                gold_type="noul",
            ),
        ]
        for i, r in enumerate(rs):
            object.__setattr__(r, "case_id", f"case-{i}")
        m = score_capability(rs)
        assert m["failures"]["answer_type_mismatch"] == 1
        assert m["n_scored_questions"] == 1
        assert m["accuracy"] == 1.0

    def test_type_routing_is_order_independent(self):
        # Same set, reversed order: identical metrics (majority routing,
        # not first-seen).
        def build(reversed_order):
            rs = [
                _make_case_result(
                    "q",
                    {"type": "noul", "noul": 1.0, "probabilities": {"true": 1.0, "false": 0.0}},
                    1,
                ),
                _make_case_result(
                    "q",
                    {"type": "noul", "noul": 1.0, "probabilities": {"true": 1.0, "false": 0.0}},
                    1,
                ),
                _make_case_result("q", {"type": "score", "score": 2.0}, 1, gold_type="noul"),
            ]
            for i, r in enumerate(rs):
                object.__setattr__(r, "case_id", f"case-{i}")
            return list(reversed(rs)) if reversed_order else rs

        m1 = score_capability(build(False))
        m2 = score_capability(build(True))
        assert m1["n_scored_questions"] == m2["n_scored_questions"] == 2
        assert m1["accuracy"] == m2["accuracy"]


class TestFailureTaxonomyCounts:
    """Categories appended to a result DURING scoring (invalid_probability,
    missing_answer, annotation_error, ...) must reach the failures taxonomy:
    the count used to happen at the top of the result's loop iteration,
    before scoring appended anything, so the taxonomy reported {} while
    n_invalid_probability > 0."""

    def test_invalid_probability_reaches_taxonomy(self):
        r = _make_case_result(
            "q",
            {"type": "noul", "noul": 1.0, "probabilities": {"true": 1.2, "false": 1.2}},
            1,
        )
        m = score_capability([r])
        assert m["calibration"]["n_invalid_probability"] == 1
        assert m["failures"].get("invalid_probability") == 1

    def test_annotation_error_reaches_taxonomy(self):
        # Gold that cannot be coerced to binary must be categorized.
        r = _make_case_result(
            "q",
            {"type": "noul", "noul": 1.0, "probabilities": {"true": 1.0, "false": 0.0}},
            "not-a-number",
        )
        m = score_capability([r])
        assert m["failures"].get("annotation_error") == 1

    def test_preexisting_categories_still_counted(self):
        # Categories the RUNNER attached before scoring (transport/schema
        # failures) must be counted too, exactly once.
        r = _make_case_result("q", {"type": "noul", "noul": None, "probabilities": None}, 1)
        from jev_agent_eval.schemas.result import CaseStatus, FailureCategory

        r.status = CaseStatus.SCHEMA_ERROR
        r.failure_categories.append(FailureCategory.SCHEMA_ERROR)
        m = score_capability([r])
        assert m["failures"].get("schema_error") == 1

    def test_no_double_count_per_case(self):
        # One case with one missing answer: taxonomy counts the category
        # once (per case), not twice (direct increment + category count).
        r = _make_case_result("q", {"type": "noul", "noul": None, "probabilities": None}, 1)
        m = score_capability([r])
        assert m["failures"].get("missing_answer") == 1
