"""dev_split isolation tests (spec 7 / invariant: holdout never leaks into a
public suite; public cases never leak into the holdout suite)."""

from __future__ import annotations

import pytest

from jev_agent_eval.runners.offline import OfflineRunner
from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer, GoldProvenance


def _case(case_id: str, dev_split: str | None) -> CanonicalCase:
    return CanonicalCase(
        case_id=case_id,
        dataset="test",
        dataset_version="1.0.0",
        capability="tool_risk",
        state=f"state for {case_id}",
        questions={"is_high_risk": {"type": "noul", "instructions": "q"}},
        gold={
            "is_high_risk": GoldAnswer(type="noul", value=0, provenance=GoldProvenance.RULE_EXACT)
        },
        dev_split=dev_split,
    )


class TestDevSplitIsolation:
    def _runner(self, suite: str):
        # Build a minimal runner without a real manifest; load_cases is the
        # unit under test, so we seed direct_cases and call it directly.
        runner = OfflineRunner.__new__(OfflineRunner)
        runner.suite = suite
        runner.capability_filter = None
        runner.tag_filter = None
        runner.max_cases = None
        runner.shuffle = False
        runner.seed = 0
        runner.direct_cases = [
            _case("pub-1", "release_public"),
            _case("hold-1", "private_holdout"),
            _case("hold-2", "private_holdout"),
            _case("legacy-1", "legacy"),
            _case("nosplit-1", None),
        ]
        return runner

    def test_release_suite_excludes_private_holdout(self):
        runner = self._runner("release")
        cases = OfflineRunner.load_cases(runner)
        ids = {c.case_id for c in cases}
        assert "hold-1" not in ids
        assert "hold-2" not in ids
        assert {"pub-1", "legacy-1", "nosplit-1"} <= ids

    def test_smoke_suite_excludes_private_holdout(self):
        runner = self._runner("smoke")
        cases = OfflineRunner.load_cases(runner)
        assert all(str(c.dev_split or "") != "private_holdout" for c in cases)

    def test_holdout_suite_keeps_only_private_holdout(self):
        runner = self._runner("holdout")
        cases = OfflineRunner.load_cases(runner)
        ids = {c.case_id for c in cases}
        assert ids == {"hold-1", "hold-2"}

    def test_redteam_suite_uses_private_split(self):
        runner = self._runner("redteam")
        cases = OfflineRunner.load_cases(runner)
        assert {c.case_id for c in cases} == {"hold-1", "hold-2"}


class TestReleaseDatasetSplit:
    """Integration with the real built dataset files (skipped if absent)."""

    def test_release_cases_file_split_counts(self):
        import json
        from pathlib import Path

        path = Path("datasets/release-v2/cases.jsonl")
        if not path.exists():
            pytest.skip("release-v2 dataset not built")
        splits: dict[str, int] = {}
        for line in path.open():
            if not line.strip():
                continue
            c = json.loads(line)
            splits[str(c.get("dev_split"))] = splits.get(str(c.get("dev_split")), 0) + 1
        # The combined store holds both, but holdout rows must be present
        # only under the private_holdout split value.
        assert splits.get("private_holdout") == 3000
        assert splits.get("release_public", 0) + splits.get("legacy", 0) >= 12000
