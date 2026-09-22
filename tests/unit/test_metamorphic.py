"""Tests for the metamorphic (spec 27) and consistency (spec 28) suites."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jev_agent_eval.cli import app
from jev_agent_eval.consistency import consistency_report
from jev_agent_eval.metamorphic import (
    append_irrelevant,
    build_variants,
    duplicate_fact,
    formatting_variation,
    paraphrase,
    position_shift,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "datasets" / "manifests" / "core-v1.yaml"

runner = CliRunner()


def _cases():
    from jev_agent_eval.datasets.registry import load_manifest, load_manifest_cases

    m = load_manifest(MANIFEST)
    return load_manifest_cases(m, MANIFEST.parent)


class TestTransforms:
    def test_paraphrase_changes_wording(self):
        s = "User asked to fix the bug. Proposed tool call: rm -rf /"
        out = paraphrase(s)
        assert "The user requested" in out
        assert "The agent intends to run:" in out
        assert "rm -rf /" in out  # content preserved

    def test_irrelevant_context_appends(self):
        out = append_irrelevant("state", "filler text")
        assert out.startswith("state")
        assert "filler text" in out

    def test_formatting_preserves_content(self):
        s = "- item one\n- item two\nplain line"
        out = formatting_variation(s)
        assert "item one" in out and "item two" in out and "plain line" in out

    def test_duplicate_fact(self):
        out = duplicate_fact("state", "fact")
        assert out.count("fact") == 1

    def test_position_shift_cyclic(self):
        s = "line1\nline2\nline3"
        out = position_shift(s)
        assert out.splitlines()[0] == "line2"
        assert out.splitlines()[-1] == "line1"

    def test_position_shift_short_state_unchanged(self):
        assert position_shift("one\ntwo") == "one\ntwo"


class TestBuildVariants:
    def test_variant_count_and_ids(self):
        cases = _cases()[:3]
        variants = build_variants(cases, max_originals=3)
        # 5 invariance transforms + negation when noul gold exists
        assert len(variants) >= 3 * 5
        ids = {v.case.case_id for v in variants}
        assert len(ids) == len(variants)  # all unique
        for v in variants:
            assert v.case.case_id.endswith(f"::mm-{v.transform}")
            assert "metamorphic" in v.case.tags

    def test_variants_keep_gold_and_questions(self):
        cases = _cases()[:1]
        variants = build_variants(cases, max_originals=1)
        for v in variants:
            assert v.case.gold == cases[0].gold
            assert v.case.questions == cases[0].questions


class TestConsistencyReport:
    def test_mock_provider_is_fully_stable(self):
        from jev_agent_eval.client.systemone import MockJEVProvider

        cases = _cases()
        case = next(c for c in cases if c.capability == "tool_risk")
        report = consistency_report(
            MockJEVProvider(), case, case.questions, "jev-latest", repeats=5
        )
        for qid, q in report["questions"].items():
            assert q["label_stability"] == 1.0
            # Probability stability applies to noul questions (choice
            # questions carry a class distribution, not p_positive).
            if qid == "is_high_risk":
                assert q["std_probability"] == 0.0
                assert q["max_probability_delta"] == 0.0


class TestCli:
    def test_metamorphic_command(self, tmp_path):
        out = tmp_path / "mm.json"
        result = runner.invoke(
            app,
            [
                "metamorphic",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--max-originals",
                "3",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        import json

        data = json.loads(out.read_text())
        assert data["n_originals"] == 3
        assert "paraphrase" in data["transforms"]
        # semantic_negation must be satisfiable: the mock provider honors
        # the marked contradiction and negation is restricted to noul
        # questions (previously pass_rate was a permanent 0.0 because
        # choice labels were included and no flip ever occurred).
        neg = data["transforms"]["semantic_negation"]
        assert neg["pass_rate"] == 1.0, neg

    def test_consistency_command(self, tmp_path):
        out = tmp_path / "consistency.json"
        result = runner.invoke(
            app,
            [
                "consistency",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--max-cases",
                "3",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
