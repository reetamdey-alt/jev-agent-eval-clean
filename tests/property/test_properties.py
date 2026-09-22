"""Property-based tests with Hypothesis (spec section: Reliability)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from jev_agent_eval.scoring.calibration import calibration_report
from jev_agent_eval.scoring.categorical import binary_scores, multiclass_scores
from jev_agent_eval.scoring.ordinal import score_scores
from jev_agent_eval.utils.hashing import canonical_json, request_hash


class TestScoringProperties:
    @settings(max_examples=100)
    @given(
        y_true=st.lists(st.sampled_from([0, 1]), min_size=1, max_size=50),
        seed=st.integers(),
    )
    def test_accuracy_bounds(self, y_true, seed):
        import random

        rng = random.Random(seed)
        y_pred = [rng.choice([0, 1]) for _ in y_true]
        m = binary_scores(y_true, y_pred, [None] * len(y_true))
        assert 0.0 <= m.accuracy <= 1.0
        assert m.n == len(y_true)

    @settings(max_examples=100)
    @given(
        values=st.lists(st.booleans(), min_size=1, max_size=100),
    )
    def test_calibration_ece_bounds(self, values):
        report = calibration_report([0.5] * len(values), values, bins=5)
        assert report.ece is not None and 0.0 <= report.ece <= 1.0
        assert report.n == len(values)

    @settings(max_examples=50)
    @given(
        pairs=st.lists(st.tuples(st.floats(0, 10), st.floats(0, 10)), min_size=1, max_size=30),
    )
    def test_mae_nonnegative(self, pairs):
        y_true = [p[0] for p in pairs]
        y_pred = [p[1] for p in pairs]
        m = score_scores(y_true, y_pred)
        assert m.mae >= 0
        assert m.rmse >= m.mae - 1e-9  # RMSE >= MAE always

    @settings(max_examples=50)
    @given(
        labels=st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=40),
    )
    def test_multiclass_accuracy_bounds(self, labels):
        m = multiclass_scores(labels, labels, [None] * len(labels))
        assert m.accuracy == 1.0  # identical labels -> perfect
        assert m.n == len(labels)


class TestHashingProperties:
    @settings(max_examples=100)
    @given(
        state=st.text(min_size=0, max_size=200),
        model=st.sampled_from(["jev-latest", "jev-v2"]),
    )
    def test_request_hash_deterministic(self, state, model):
        payload = {"state": state, "model": model, "questions": {}}
        assert request_hash(payload) == request_hash(dict(payload))

    @settings(max_examples=100)
    @given(
        a=st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), max_size=10),
    )
    def test_canonical_json_key_order_irrelevant(self, a):
        reordered = dict(reversed(list(a.items())))
        assert canonical_json(a) == canonical_json(reordered)

    @settings(max_examples=50)
    @given(
        a=st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), max_size=10),
        b=st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), max_size=10),
    )
    def test_request_hash_collision_resistance(self, a, b):
        # Different payloads (after secret stripping) hash differently;
        # identical content regardless of key order hashes identically.
        if canonical_json(a) != canonical_json(b):
            assert request_hash(a) != request_hash(b)
