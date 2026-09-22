"""Contract tests: full pipeline behavior against a mock provider (spec 42, live=False)."""

from __future__ import annotations

import json

import pytest

from jev_agent_eval.client.systemone import MockJEVProvider
from jev_agent_eval.schemas.request import JEVQuestion, JEVRequest
from jev_agent_eval.schemas.response import JEVResponse


class TestMockProviderContract:
    """The mock provider must honor the same contract as the live one."""

    def test_auth_not_required_for_mock(self):
        prov = MockJEVProvider()
        assert prov.name == "mock"

    def test_model_acceptance(self):
        prov = MockJEVProvider(model="jev-latest")
        req = JEVRequest(state="s", model="jev-latest", questions={})
        result = prov.infer(req)
        assert result.response is not None
        assert result.response.model == "jev-latest"

    def test_noul_supported(self):
        prov = MockJEVProvider()
        req = JEVRequest(
            state="anything",
            model="jev-latest",
            questions={"q": JEVQuestion(type="noul", instructions="?")},
        )
        resp = prov.infer(req).response
        assert resp is not None and "q" in resp.answers
        assert resp.answers["q"].type == "noul"

    def test_choice_supported(self):
        prov = MockJEVProvider()
        req = JEVRequest(
            state="anything",
            model="jev-latest",
            questions={
                "q": JEVQuestion(
                    type="choice",
                    instructions="?",
                    criteria={"a": "A", "b": "B"},
                )
            },
        )
        resp = prov.infer(req).response
        assert resp is not None
        ans = resp.answers["q"]
        assert ans.choice in ("a", "b")
        assert ans.probabilities is not None
        assert sum(ans.probabilities.values()) == pytest.approx(1.0, abs=1e-3)

    def test_score_supported(self):
        prov = MockJEVProvider()
        req = JEVRequest(
            state="anything",
            model="jev-latest",
            questions={"q": JEVQuestion(type="score", instructions="?")},
        )
        resp = prov.infer(req).response
        assert resp is not None
        assert resp.answers["q"].score is not None

    def test_multiple_questions_per_request(self):
        prov = MockJEVProvider()
        req = JEVRequest(
            state="anything",
            model="jev-latest",
            questions={
                "q1": JEVQuestion(type="noul", instructions="?"),
                "q2": JEVQuestion(type="choice", instructions="?", criteria={"x": "X", "y": "Y"}),
                "q3": JEVQuestion(type="score", instructions="?"),
            },
        )
        resp = prov.infer(req).response
        assert resp is not None
        assert set(resp.answers.keys()) == {"q1", "q2", "q3"}

    def test_usage_fields_present(self):
        prov = MockJEVProvider()
        req = JEVRequest(
            state="some state text",
            model="jev-latest",
            questions={"q": JEVQuestion(type="noul", instructions="?")},
        )
        resp = prov.infer(req).response
        assert resp is not None and resp.usage is not None
        assert resp.usage.input_tokens and resp.usage.input_tokens > 0
        assert resp.usage.output_tokens and resp.usage.output_tokens > 0

    def test_response_schema_stability(self):
        prov = MockJEVProvider()
        req = JEVRequest(
            state="x",
            model="jev-latest",
            questions={"q": JEVQuestion(type="noul", instructions="?")},
        )
        result = prov.infer(req)
        # Raw body must round-trip through the schema.
        parsed = JEVResponse.model_validate(json.loads(result.raw_body))
        assert parsed.answers["q"].type == "noul"

    def test_deterministic(self):
        prov = MockJEVProvider()
        req = JEVRequest(
            state="determinism check state",
            model="jev-latest",
            questions={"q": JEVQuestion(type="noul", instructions="?")},
        )
        r1 = prov.infer(req)
        r2 = prov.infer(req)
        assert r1.raw_body == r2.raw_body


class TestResponseValidationCLI:
    def test_validate_response_command(self, tmp_path):
        from typer.testing import CliRunner

        from jev_agent_eval.cli import app

        runner = CliRunner()
        good = tmp_path / "good.json"
        good.write_text(
            json.dumps(
                {
                    "model": "m",
                    "answers": {"q": {"type": "noul", "noul": 0.5}},
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            ),
            encoding="utf-8",
        )
        result = runner.invoke(app, ["validate-response", str(good)])
        assert result.exit_code == 0

        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["validate-response", str(bad)])
        assert result.exit_code == 6  # schema failure

        # Semantically invalid: probability vector sums to 2.4. The schema
        # tolerates it so runs can record and categorize bad responses, but
        # this command checks the full provider contract (spec 34).
        semantically_bad = tmp_path / "semantic.json"
        semantically_bad.write_text(
            json.dumps(
                {
                    "model": "m",
                    "answers": {
                        "q": {
                            "type": "noul",
                            "noul": 1,
                            "probabilities": {"true": 1.5, "false": 0.9},
                        }
                    },
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            ),
            encoding="utf-8",
        )
        result = runner.invoke(app, ["validate-response", str(semantically_bad)])
        assert result.exit_code == 6

        # Missing usage: contract requires the usage field to be present.
        no_usage = tmp_path / "no_usage.json"
        no_usage.write_text(
            json.dumps({"model": "m", "answers": {"q": {"type": "noul", "noul": 0.5}}}),
            encoding="utf-8",
        )
        result = runner.invoke(app, ["validate-response", str(no_usage)])
        assert result.exit_code == 6


class TestMockHeuristicEvidenceScope:
    """Regression tests for mock-provider evidence scoping (loop 6)."""

    def test_completion_evidence_not_taken_from_user_request(self):
        # "make the suite green" is the user's GOAL statement, not evidence
        # of completion; the agent has only started work.
        prov = MockJEVProvider()
        req = JEVRequest(
            state=(
                "User request: Fix the failing auth test and make the suite green.\n\n"
                "Agent message: I'll start by running the test suite to see what fails."
            ),
            model="jev-latest",
            questions={
                "goal_complete": JEVQuestion(type="noul", instructions="Is the goal complete?")
            },
        )
        assert prov.infer(req).response.answers["goal_complete"].noul == 0.0

    def test_zero_failure_count_is_not_contradiction(self):
        # "failed': 0" in a passing test summary must not read as failure.
        prov = MockJEVProvider()
        req = JEVRequest(
            state=(
                "Tool result: ok\nPASS auth.test.ts 42/42\nAll 60 tests passing.\n\n"
                "Tests: {'total': 60, 'passed': 60, 'failed': 0}"
            ),
            model="jev-latest",
            questions={
                "goal_complete": JEVQuestion(type="noul", instructions="Is the goal complete?")
            },
        )
        assert prov.infer(req).response.answers["goal_complete"].noul == 1.0

    def test_repeated_failing_command_is_stuck(self):
        # Retry language + an error result = stuck (3rd identical attempt).
        prov = MockJEVProvider()
        req = JEVRequest(
            state=(
                "Agent message: Running it a third time; maybe it will pass now.\n\n"
                "Tool result: error\nFailed to connect"
            ),
            model="jev-latest",
            questions={"stuck": JEVQuestion(type="noul", instructions="Is the agent stuck?")},
        )
        assert prov.infer(req).response.answers["stuck"].noul == 1.0

    def test_single_failure_without_retry_is_not_stuck(self):
        prov = MockJEVProvider()
        req = JEVRequest(
            state="Tool result: error\nFailed to connect",
            model="jev-latest",
            questions={"stuck": JEVQuestion(type="noul", instructions="Is the agent stuck?")},
        )
        assert prov.infer(req).response.answers["stuck"].noul == 0.0

    def test_risk_scoping_survives_paraphrase_markers(self):
        # The mock must scope risk to the proposed command under ANY of the
        # accepted proposal markers; a paraphrased marker ("intends to run:")
        # previously fell back to whole-text scanning, so "verify credentials"
        # in the request made a benign read-only command look risky.
        prov = MockJEVProvider()
        for marker in ("Proposed tool call:", "The agent intends to run:"):
            req = JEVRequest(
                state=f"User asked to verify credentials. {marker} aws sts get-caller-identity",
                model="jev-latest",
                questions={
                    "is_high_risk": JEVQuestion(type="noul", instructions="Is this high risk?")
                },
            )
            assert prov.infer(req).response.answers["is_high_risk"].noul == 0.0, marker
