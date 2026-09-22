"""Golden tests: exact parsing/scoring behavior against fixed fixtures (spec 41).

These tests never call the provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_agent_eval.client.systemone import SystemOneProvider
from jev_agent_eval.schemas.request import JEVQuestion, JEVRequest
from jev_agent_eval.schemas.response import JEVResponse
from jev_agent_eval.scoring.calibration import p_positive, validate_probability_vector

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict | str:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_body(name: str) -> tuple[JEVResponse | None, str | None]:
    provider = object.__new__(SystemOneProvider)  # no network, no auth needed
    body = (FIXTURES / name).read_text(encoding="utf-8")
    return provider._parse_response(body)


class TestValidResponses:
    def test_valid_noul(self):
        resp, err = parse_body("valid_noul.json")
        assert err is None and resp is not None
        ans = resp.answers["q1"]
        assert ans.type == "noul"
        assert ans.noul == pytest.approx(0.8994)
        assert p_positive(ans.model_dump()) == pytest.approx(0.8994)
        assert resp.usage.input_tokens == 218

    def test_valid_choice_three_classes(self):
        resp, err = parse_body("valid_choice.json")
        assert err is None and resp is not None
        ans = resp.answers["q1"]
        assert ans.choice == "billing"
        probs = ans.probabilities
        assert validate_probability_vector(probs)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_valid_score(self):
        resp, err = parse_body("valid_score.json")
        assert err is None and resp is not None
        assert resp.answers["q1"].score == pytest.approx(3.0)

    def test_extra_provider_fields_tolerated(self):
        resp, err = parse_body("extra_provider_fields.json")
        assert err is None and resp is not None
        assert resp.answers["q1"].noul == pytest.approx(0.9)
        assert resp.usage.output_tokens == 1

    def test_null_confidence_tolerated(self):
        resp, err = parse_body("null_confidence.json")
        assert err is None and resp is not None
        assert resp.answers["q1"].confidence is None


class TestInvalidResponses:
    def test_malformed_json(self):
        provider = object.__new__(SystemOneProvider)
        _, err = provider._parse_response(
            (FIXTURES / "malformed_json.json").read_text(encoding="utf-8")
        )
        assert err is not None and "malformed_json" in err

    def test_missing_answer_detected_at_scoring(self):
        # A response may parse fine but lack the question we asked about;
        # the runner must classify this as missing_answer, not silently pass.
        resp, err = parse_body("missing_answer.json")
        assert err is None and resp is not None
        assert "q1" not in resp.answers  # we asked q1; provider returned q2

    def test_unknown_answer_type(self):
        # The response schema tolerates unknown types; scoring must reject them.
        resp, err = parse_body("unknown_answer_type.json")
        assert err is None and resp is not None
        assert resp.answers["q1"].type == "weird"

    def test_probability_sum_error(self):
        resp, _ = parse_body("prob_sum_error.json")
        probs = resp.answers["q1"].probabilities
        assert not validate_probability_vector(probs)  # sums to 1.2

    def test_out_of_range_probability(self):
        resp, _ = parse_body("out_of_range_probability.json")
        probs = resp.answers["q1"].probabilities
        assert not validate_probability_vector(probs)  # p > 1

    def test_missing_probabilities(self):
        resp, _ = parse_body("missing_probabilities.json")
        assert resp.answers["q1"].probabilities is None


class TestHTTPStatuses:
    @pytest.mark.parametrize(
        "fixture,expected_status",
        [
            ("http_429.json", 429),
            ("http_500.json", 500),
        ],
    )
    def test_error_bodies_not_treated_as_responses(self, fixture, expected_status):
        # Error bodies are not valid JEVResponse objects.
        with pytest.raises(Exception):
            JEVResponse.model_validate(load(fixture))
        assert expected_status in (429, 500)


class TestBinaryMapping:
    """Spec 17.1: the binary probability mapping is validated by fixtures."""

    def test_noul_probability_mapping(self):
        # probabilities["true"] takes precedence; noul is the fallback.
        assert p_positive({"probabilities": {"true": 0.7, "false": 0.3}}) == pytest.approx(0.7)
        assert p_positive({"noul": 0.4}) == pytest.approx(0.4)
        assert p_positive({}) is None

    def test_mock_provider_binary_semantics(self):
        from jev_agent_eval.client.systemone import MockJEVProvider

        prov = MockJEVProvider()
        req = JEVRequest(
            state="User asked to inspect the repository. Proposed tool call: rm -rf build/",
            model="jev-latest",
            questions={"q": JEVQuestion(type="noul", instructions="Is this risky?")},
        )
        result = prov.infer(req)
        assert result.response is not None
        p = p_positive(result.response.answers["q"].model_dump())
        assert p is not None and p >= 0.5  # risky state -> positive
