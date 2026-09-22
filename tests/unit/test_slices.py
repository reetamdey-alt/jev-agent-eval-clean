"""Slice analysis tests (spec 32, 60): bucket semantics and dimensions."""

from __future__ import annotations

from jev_agent_eval.schemas.result import CaseResult
from jev_agent_eval.scoring.slices import (
    build_slices,
    length_bucket,
    worst_slices,
)


def _result(cap="tool_risk", state="x" * 100, correct=True, **kw):
    return CaseResult(
        case_id=kw.pop("case_id", "c1"),
        capability=cap,
        dataset=kw.pop("dataset", "synthetic"),
        dataset_version="v1",
        status=kw.pop("status", "success"),
        state=state,
        state_chars=len(state),
        predictions={"q": {"noul": 1.0}},
        correct={"q": correct},
        **kw,
    )


class TestLengthBucket:
    """Spec 60 buckets are defined in tokens; the evaluator measures chars."""

    def test_buckets_are_token_based(self):
        # 3,900 chars is under 1K tokens (chars/4) -> "<1K", not "1-4K".
        assert length_bucket(3_900) == "<1K"
        assert length_bucket(4_000) == "1-4K"
        assert length_bucket(4_100) == "1-4K"

    def test_bucket_boundaries_half_open(self):
        assert length_bucket(0) == "<1K"
        assert length_bucket(3_999) == "<1K"  # 999 tokens
        assert length_bucket(4_000) == "1-4K"  # 1,000 tokens
        assert length_bucket(15_999) == "1-4K"  # 3,999 tokens
        assert length_bucket(16_000) == "4-16K"  # 4,000 tokens
        assert length_bucket(64_000) == "16-32K"  # 16,000 tokens
        assert length_bucket(256_000) == "64K+"  # 64,000 tokens
        assert length_bucket(10**9) == "64K+"

    def test_bucket_matches_spec_names(self):
        seen = {length_bucket(n) for n in (0, 5_000, 20_000, 80_000, 200_000, 300_000)}
        assert seen <= {"<1K", "1-4K", "4-16K", "16-32K", "32-64K", "64K+"}


class TestBuildSlices:
    def test_capability_dimension_groups(self):
        rs = [_result("a"), _result("a", case_id="c2", correct=False), _result("b", case_id="c3")]
        slices = build_slices(
            rs, lambda g: {"accuracy": sum(1 for r in g if all(r.correct.values())) / len(g)}
        )
        assert slices["capability"]["a"]["accuracy"] == 0.5
        assert slices["capability"]["a"]["n"] == 2
        assert slices["capability"]["b"]["n"] == 1

    def test_length_dimension_uses_token_estimate(self):
        rs = [_result(state="x" * 4_100, case_id="c1"), _result(state="x" * 500, case_id="c2")]
        slices = build_slices(rs, lambda g: {"accuracy": 1.0})
        assert set(slices["input_length_bucket"]) == {"1-4K", "<1K"}

    def test_worst_slices_respects_min_n(self):
        rs = [_result("a"), _result("b", case_id="c2")]
        slices = build_slices(rs, lambda g: {"accuracy": 0.0})
        assert worst_slices(slices, min_n=5) == []
        cap_rows = [r for r in worst_slices(slices, min_n=1) if r[0] == "capability"]
        assert len(cap_rows) == 2
