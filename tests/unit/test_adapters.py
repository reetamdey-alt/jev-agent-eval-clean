"""Smoke tests for all six dataset adapters (spec sections 4.1-4.6, 57).

These were previously untested. Each test feeds a plausible upstream record
through enumerate_cases/transform and checks the invariants that matter for
eval validity:

- gold provenance matches how the label was actually derived
- gold values/keys never leak into the JEV-visible state
- question ids and gold keys are aligned
- derived case ids are unique
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_agent_eval.datasets.bfcl import BFCLAdapter
from jev_agent_eval.datasets.ifbench import IFBenchAdapter
from jev_agent_eval.datasets.ifeval import IFEvalAdapter
from jev_agent_eval.datasets.swebench import SWEBenchAdapter
from jev_agent_eval.datasets.tau2 import Tau2Adapter
from jev_agent_eval.datasets.tuabench import TUABenchAdapter
from jev_agent_eval.schemas.case import GoldProvenance


def _write_jsonl(tmp_path: Path, records: list[dict]) -> str:
    p = tmp_path / "data.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(p)


def _adapt(adapter, tmp_path: Path, records: list[dict]):
    path = _write_jsonl(tmp_path, records)
    a = adapter(path)
    out = []
    for raw in a.enumerate_cases():
        out.extend(a.transform(raw))
    return out


def _check_invariants(cases):
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in cases:
        assert set(c.questions) == set(c.gold), f"{c.case_id}: question/gold key mismatch"
        for qid, gold in c.gold.items():
            # Gold must never be derivable from the state itself in a way that
            # trivially answers the question (structural labels like patch
            # contents are the whole point of keeping gold out of state).
            assert gold.provenance in {p for p in GoldProvenance}
            assert gold.type == c.questions[qid]["type"], (
                f"{c.case_id}/{qid}: gold type != question type"
            )
            if gold.type == "noul":
                assert gold.value in (0, 1)
        assert c.state.strip(), f"{c.case_id}: empty state"


class TestBFCL:
    def test_ground_truth_authoritative(self, tmp_path):
        # Irrelevance-style record: functions present, ground truth says no call.
        cases = _adapt(
            BFCLAdapter,
            tmp_path,
            [
                {
                    "question": "Write me a poem.",
                    "functions": [{"name": "get_weather"}],
                    "ground_truth": "",
                }
            ],
        )
        assert len(cases) == 1
        g = cases[0].gold["tool_relevant"]
        assert g.value == 0, "ground_truth=''/no-call must yield gold=0"
        assert g.provenance == GoldProvenance.BENCHMARK_GOLD

    def test_heuristic_fallback_is_rule_derived(self, tmp_path):
        cases = _adapt(
            BFCLAdapter,
            tmp_path,
            [{"question": "What is the weather?", "functions": [{"name": "get_weather"}]}],
        )
        g = cases[0].gold["tool_relevant"]
        assert g.value == 1
        assert g.provenance == GoldProvenance.RULE_DERIVED

    def test_execution_result_not_in_state(self, tmp_path):
        cases = _adapt(
            BFCLAdapter,
            tmp_path,
            [
                {
                    "question": "Book a flight.",
                    "functions": [{"name": "book_flight"}],
                    "execution_result": "TypeError: argument 'date' missing",
                }
            ],
        )
        _check_invariants(cases)
        # The recoverable case may expose the error (that is its premise), but
        # the tool-needed case must not reveal execution outcome.
        tool_case = next(c for c in cases if c.case_id.endswith("-tool-needed"))
        assert "TypeError" not in tool_case.state

    def test_recoverable_derived_from_error_text(self, tmp_path):
        cases = _adapt(
            BFCLAdapter,
            tmp_path,
            [
                {
                    "question": "Book a flight.",
                    "functions": [{"name": "book_flight"}],
                    "execution_result": "TypeError: argument 'date' missing",
                }
            ],
        )
        rec = next(c for c in cases if c.case_id.endswith("-recoverable"))
        assert rec.gold["recoverable"].value == 1
        assert rec.gold["recoverable"].provenance == GoldProvenance.RULE_DERIVED


class TestIFEval:
    def test_violation_gold(self, tmp_path):
        cases = _adapt(
            IFEvalAdapter,
            tmp_path,
            [
                {
                    "prompt": "Write exactly 3 sentences.",
                    "response": "One sentence.",
                    "violations": ["sentence_count"],
                },
                {"prompt": "Write exactly 3 sentences.", "response": "A. B. C.", "violations": []},
            ],
        )
        assert [c.gold["violates_constraint"].value for c in cases] == [1, 0]
        assert all(
            c.gold["violates_constraint"].provenance == GoldProvenance.BENCHMARK_GOLD for c in cases
        )
        _check_invariants(cases)

    def test_state_contains_prompt_and_response(self, tmp_path):
        cases = _adapt(
            IFEvalAdapter,
            tmp_path,
            [{"prompt": "Use no commas", "response": "Fine prose here", "violations": []}],
        )
        assert "Use no commas" in cases[0].state
        assert "Fine prose here" in cases[0].state
        assert "violations" not in cases[0].state


class TestIFBench:
    def test_violation_gold(self, tmp_path):
        cases = _adapt(
            IFBenchAdapter,
            tmp_path,
            [
                {
                    "prompt": "Use at most 10 words.",
                    "response": "way way way way way way way way way way way too long",
                    "violations": ["word_limit"],
                }
            ],
        )
        assert cases[0].gold["violates_constraint"].value == 1
        _check_invariants(cases)


class TestSWEBench:
    def test_code_change_and_multi_file(self, tmp_path):
        patch = "--- a/x.py\n+++ b/x.py\n@@\n--- a/y.py\n+++ b/y.py\n@@\n"
        cases = _adapt(
            SWEBenchAdapter,
            tmp_path,
            [
                {
                    "instance_id": "org__repo-1234",
                    "repo": "org/repo",
                    "problem_statement": "Fix the crash on empty input.",
                    "patch": patch,
                }
            ],
        )
        _check_invariants(cases)
        by_q = {c.case_id: c for c in cases}
        code_change = by_q["org__repo-1234-needs-code-change"]
        multi = by_q["org__repo-1234-multi-file"]
        assert code_change.gold["requires_code_modification"].value == 1
        assert multi.gold["likely_multi_file"].value == 1
        assert multi.gold["likely_multi_file"].provenance == GoldProvenance.EXECUTION_DERIVED
        # The patch (gold evidence) must never enter the state.
        assert "--- a/x.py" not in code_change.state
        assert "--- a/x.py" not in multi.state

    def test_single_file_patch(self, tmp_path):
        patch = "--- a/x.py\n+++ b/x.py\n@@\n"
        cases = _adapt(
            SWEBenchAdapter,
            tmp_path,
            [{"instance_id": "i-1", "repo": "r", "problem_statement": "s", "patch": patch}],
        )
        multi = next(c for c in cases if c.case_id.endswith("-multi-file"))
        assert multi.gold["likely_multi_file"].value == 0


class TestTau2:
    def test_conversation_list_rendered_readably(self, tmp_path):
        cases = _adapt(
            Tau2Adapter,
            tmp_path,
            [
                {
                    "conversation": [
                        {"role": "user", "content": "I want a refund."},
                        {"role": "agent", "content": "I can help with that."},
                    ],
                    "policy": "Refunds allowed within 30 days.",
                    "action": "issue refund",
                    "policy_compliant": True,
                }
            ],
        )
        state = cases[0].state
        assert "user: I want a refund." in state
        assert "agent: I can help with that." in state
        assert "{'role'" not in state, "list repr leaked into state"

    def test_policy_compliance_gold(self, tmp_path):
        cases = _adapt(
            Tau2Adapter,
            tmp_path,
            [{"conversation": "c", "policy": "p", "action": "a", "policy_compliant": False}],
        )
        assert cases[0].gold["policy_compliant"].value == 0
        assert cases[0].gold["policy_compliant"].provenance == GoldProvenance.BENCHMARK_GOLD
        _check_invariants(cases)


class TestTUABench:
    def test_action_appropriate_gold(self, tmp_path):
        cases = _adapt(
            TUABenchAdapter,
            tmp_path,
            [
                {"task": "Count files in /tmp", "command": "ls /tmp | wc -l", "success": True},
                {"task": "Count files in /tmp", "command": "rm -rf /", "success": False},
            ],
        )
        assert [c.gold["action_appropriate"].value for c in cases] == [1, 0]
        assert all(
            c.gold["action_appropriate"].provenance == GoldProvenance.EXECUTION_DERIVED
            for c in cases
        )
        _check_invariants(cases)

    def test_succeeded_alias(self, tmp_path):
        cases = _adapt(
            TUABenchAdapter,
            tmp_path,
            [{"task": "t", "command": "true", "succeeded": True}],
        )
        assert cases[0].gold["action_appropriate"].value == 1


def test_all_adapters_expose_metadata():
    for cls in (
        BFCLAdapter,
        IFEvalAdapter,
        IFBenchAdapter,
        SWEBenchAdapter,
        Tau2Adapter,
        TUABenchAdapter,
    ):
        with pytest.raises(NotImplementedError):
            cls("/nonexistent").download()
        # metadata() itself must not touch the filesystem.
        m = cls("/nonexistent").metadata()
        assert m.name
        assert m.version
        assert m.upstream_repo
