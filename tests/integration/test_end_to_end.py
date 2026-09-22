"""Integration tests: full run -> replay -> compare cycle with the mock provider.

These exercise the entire pipeline (runner, scoring, reports, artifacts)
without network access, matching the smoke CI profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jev_agent_eval.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "datasets" / "manifests" / "core-v1.yaml"

runner = CliRunner()


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """One smoke run, reused by the module's tests."""
    out = tmp_path_factory.mktemp("reports")
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(MANIFEST),
            "--provider",
            "mock",
            "--suite",
            "smoke",
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code in (0, 1), result.output
    runs = sorted((out / "runs").iterdir())
    assert len(runs) == 1
    return runs[0]


class TestRunArtifacts:
    def test_all_artifacts_written(self, run_dir):
        expected = [
            "manifest.json",
            "summary.json",
            "summary.md",
            "report.html",
            "cases.jsonl",
            "responses.jsonl",
            "scored.jsonl",
            "errors.jsonl",
            "metrics.json",
            "slices.json",
            "calibration.json",
            "trace.jsonl",
        ]
        for name in expected:
            assert (run_dir / name).exists(), f"missing artifact: {name}"

    def test_responses_persist_retry_history(self, run_dir):
        # Regression (2026-09-22 live log audit): retry_history was captured
        # on the TransportRecord but never written to responses.jsonl, so the
        # per-attempt backoff (incl. whether the server-aware 429 wait fired)
        # could not be audited from run logs.
        for ln in (run_dir / "responses.jsonl").read_text().splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            assert "retry_history" in rec, rec["case_id"]

    def test_manifest_self_describing(self, run_dir):
        manifest = json.loads((run_dir / "manifest.json").read_text())
        for key in (
            "run_id",
            "created_at",
            "evaluator_version",
            "model",
            "provider_url",
            "dataset_manifest",
            "dataset_hash",
            "seed",
            "repeats",
            "cache_enabled",
        ):
            assert key in manifest, f"manifest missing {key}"
        assert "api_key" not in json.dumps(manifest).lower()

    def test_no_secrets_in_artifacts(self, run_dir):
        for artifact in run_dir.iterdir():
            text = artifact.read_text(encoding="utf-8", errors="ignore")
            assert "JEV_API_KEY" not in text
            assert "Bearer " not in text

    def test_case_count_matches(self, run_dir):
        cases = [
            line for line in (run_dir / "cases.jsonl").read_text().splitlines() if line.strip()
        ]
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert len(cases) == manifest["case_count"]
        assert manifest["case_count"] <= 300  # v2 smoke suite cap (spec 52: 250-400)

    def test_metrics_structure(self, run_dir):
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["n_cases"] > 0
        assert "capabilities" in metrics
        assert "latency" in metrics
        assert "usage" in metrics
        for _cap, m in metrics["capabilities"].items():
            assert "n_cases" in m
            assert "accuracy" in m  # present even if None


class TestReplay:
    def test_replay_reproduces_metrics_exactly(self, run_dir):
        metrics_before = json.loads((run_dir / "metrics.json").read_text())
        result = runner.invoke(app, ["replay", str(run_dir)])
        assert result.exit_code in (0, 1), result.output
        metrics_after = json.loads((run_dir / "metrics.json").read_text())
        # Correctness metrics must reproduce exactly (latency may differ).
        assert metrics_before["n_cases"] == metrics_after["n_cases"]
        for cap in metrics_before["capabilities"]:
            a = metrics_before["capabilities"][cap].get("accuracy")
            b = metrics_after["capabilities"][cap].get("accuracy")
            assert a == b

    def test_replay_regenerates_scored_jsonl(self, run_dir):
        # Regression: replay used to rewrite metrics.json but leave a stale
        # scored.jsonl, so compare's paired bootstrap contradicted the
        # accuracy table (Δ=+0.0000 next to a −0.80 regression).

        responses_path = run_dir / "responses.jsonl"
        backup = responses_path.read_text()
        records = [json.loads(ln) for ln in backup.splitlines() if ln.strip()]
        flipped = 0
        for rec in records:  # flip the first noul answer of 20 cases
            if flipped >= 20:
                break
            resp = rec.get("response")
            if not resp:
                continue
            for a in resp["answers"].values():
                if a.get("type") == "noul" and a.get("noul") is not None:
                    a["noul"] = 1 - a["noul"]
                    probs = a.get("probabilities")
                    if isinstance(probs, dict) and "true" in probs:
                        probs["true"], probs["false"] = probs["false"], probs["true"]
                    flipped += 1
                    break
        responses_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        try:
            result = runner.invoke(app, ["replay", str(run_dir)])
            assert result.exit_code == 1  # degraded run must fail gates
            scored = [
                json.loads(ln)
                for ln in (run_dir / "scored.jsonl").read_text().splitlines()
                if ln.strip()
            ]
            wrong = [s for s in scored if s.get("question_id") and not s["correct"]]
            assert len(wrong) == flipped, (
                f"scored.jsonl shows {len(wrong)} incorrect, expected {flipped}"
            )
            metrics = json.loads((run_dir / "metrics.json").read_text())
            # metrics and scored.jsonl must tell the same story
            n_all = sum(1 for s in scored if s.get("question_id"))
            n_scored_caps = sum(
                v.get("n_scored_questions", 0) for v in metrics["capabilities"].values()
            )
            assert n_scored_caps == n_all, "scored rows and metrics disagree"
        finally:
            responses_path.write_text(backup)
            runner.invoke(app, ["replay", str(run_dir)])  # restore artifacts


class TestReplayConfigSnapshot:
    def test_replay_restores_run_gates(self, tmp_path):
        # Regression: replay used to evaluate gates from the DEFAULT
        # config, so a run that exited 1 under a strict custom config
        # would replay to exit 0. The run now snapshots its resolved
        # config and replay restores it.
        strict = tmp_path / "strict.yaml"
        strict.write_text(
            "run:\n  cache: false\n"
            "quality_gates:\n  tool_risk:\n    accuracy_min: 1.01\n"  # impossible
        )
        out = tmp_path / "reports"
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--config",
                str(strict),
                "--output-dir",
                str(out),
            ],
        )
        assert r.exit_code == 1, "impossible gate must fail the run"
        run_dir = sorted((out / "runs").iterdir())[0]
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest.get("config"), "run manifest lacks config snapshot"
        r = runner.invoke(app, ["replay", str(run_dir)])
        assert r.exit_code == 1, "replay must restore the run's strict gates"


class TestCompare:
    def test_compare_self_is_stable(self, run_dir):
        result = runner.invoke(app, ["compare", str(run_dir), str(run_dir)])
        assert result.exit_code == 0, result.output
        assert "stable" in result.output

    def test_compare_two_runs(self, tmp_path, run_dir):
        out = tmp_path / "reports"
        result = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--output-dir",
                str(out),
            ],
        )
        assert result.exit_code in (0, 1)
        other = sorted((out / "runs").iterdir())[0]
        result = runner.invoke(app, ["compare", str(run_dir), str(other)])
        assert result.exit_code == 0
        assert "Run A" in result.output and "Run B" in result.output


class TestDatasetCommands:
    def test_datasets_list(self):
        result = runner.invoke(app, ["datasets", "list"])
        assert result.exit_code == 0
        assert "core-v1.yaml" in result.output

    def test_datasets_validate(self):
        result = runner.invoke(app, ["datasets", "validate", str(MANIFEST)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_datasets_validate_missing(self, tmp_path):
        result = runner.invoke(app, ["datasets", "validate", str(tmp_path / "nope.yaml")])
        assert result.exit_code == 3


class TestExitCodes:
    def test_config_error_exit_2(self, tmp_path):
        result = runner.invoke(
            app, ["run", "--dataset", str(MANIFEST), "--config", str(tmp_path / "missing.yaml")]
        )
        assert result.exit_code == 2

    def test_dataset_error_exit_3(self, tmp_path):
        result = runner.invoke(
            app, ["run", "--dataset", str(tmp_path / "missing.yaml"), "--provider", "mock"]
        )
        assert result.exit_code == 3

    def test_case_inspection(self, run_dir):
        cases = [
            json.loads(line)
            for line in (run_dir / "cases.jsonl").read_text().splitlines()
            if line.strip()
        ]
        case_id = cases[0]["case_id"]
        result = runner.invoke(app, ["case", case_id, "--run-dir", str(run_dir)])
        assert result.exit_code == 0
        assert case_id in result.output

    def test_report_command(self, run_dir):
        result = runner.invoke(app, ["report", str(run_dir)])
        assert result.exit_code in (0, 1)


class TestCacheAndDeterminism:
    def test_rerun_uses_cache(self, tmp_path):
        """Second run of identical requests should hit the response cache."""
        out = tmp_path / "reports"
        args = [
            "run",
            "--dataset",
            str(MANIFEST),
            "--provider",
            "mock",
            "--suite",
            "smoke",
            "--output-dir",
            str(out),
        ]
        r1 = runner.invoke(app, args)
        assert r1.exit_code in (0, 1)
        runs = sorted((out / "runs").iterdir())
        first_responses = [
            json.loads(line)
            for line in (runs[0] / "responses.jsonl").read_text().splitlines()
            if line.strip()
        ]
        r2 = runner.invoke(app, args)
        assert r2.exit_code in (0, 1)
        runs = sorted((out / "runs").iterdir())
        second_responses = [
            json.loads(line)
            for line in (runs[1] / "responses.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(first_responses) == len(second_responses)
        assert all(r.get("from_cache") for r in second_responses)
        # Cached runs must not report live latency as production latency.
        assert second_responses[0].get("from_cache") is True

    def test_tag_filters_cases(self, tmp_path):
        """--tag must actually restrict the run to tagged cases."""
        out = tmp_path / "reports"
        all_args = [
            "run",
            "--dataset",
            str(MANIFEST),
            "--provider",
            "mock",
            "--suite",
            "smoke",
            "--output-dir",
            str(out),
        ]
        tag_args = all_args + ["--tag", "adversarial"]
        r_all = runner.invoke(app, all_args)
        assert r_all.exit_code in (0, 1), r_all.output
        r_tag = runner.invoke(app, tag_args)
        assert r_tag.exit_code in (0, 1), r_tag.output
        all_runs = sorted((out / "runs").iterdir())
        assert len(all_runs) == 2
        tag_cases = [
            json.loads(line)
            for line in (all_runs[1] / "cases.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert tag_cases, "tag-filtered run produced no cases"
        # Every case in the filtered run carries the requested tag.
        for c in tag_cases:
            assert "adversarial" in c.get("tags", [])
        # And the run is strictly smaller than the unfiltered one.
        all_cases = [
            json.loads(line)
            for line in (all_runs[0] / "cases.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(tag_cases) < len(all_cases)


def _strip_volatile(metrics: dict) -> dict:
    """Remove fields that legitimately differ between runs (latency only,
    and cache-vs-live split: a resumed run serves most cases from cache
    while a fresh run is all-live, so n_live_calls/n_cache_served and the
    live-call error rates differ by design)."""
    out = json.loads(json.dumps(metrics))
    out.pop("latency", None)
    out.pop("n_live_calls", None)
    out.pop("n_cache_served", None)
    out.pop("transport_error_rate", None)
    out.pop("schema_error_rate", None)
    for cap in out.get("capabilities", {}).values():
        cap.pop("latency", None)
    return out


class TestResume:
    def _partial_run(self, out: Path, keep: int = 7) -> Path:
        """Create a smoke run, then truncate it as if interrupted."""
        args = [
            "run",
            "--dataset",
            str(MANIFEST),
            "--provider",
            "mock",
            "--suite",
            "smoke",
            "--output-dir",
            str(out),
        ]
        r = runner.invoke(app, args)
        assert r.exit_code in (0, 1), r.output
        run_dir = sorted((out / "runs").iterdir())[0]
        resp = run_dir / "responses.jsonl"
        lines = resp.read_text().splitlines()
        resp.write_text("\n".join(lines[:keep]) + "\n")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["completed"] = False
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        return run_dir

    def test_resume_reproduces_original_metrics_exactly(self, tmp_path):
        # Regression: resumed runs used to double-count newly executed
        # cases (n_cases 100 -> 193) AND drop reloaded cases from slice
        # metrics. A resumed run must equal the original bit-for-bit.
        full_out = tmp_path / "full"
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--output-dir",
                str(full_out),
            ],
        )
        assert r.exit_code in (0, 1), r.output
        original = sorted((full_out / "runs").iterdir())[0]

        run_dir = self._partial_run(tmp_path / "partial")
        r = runner.invoke(app, ["resume", str(run_dir)])
        assert r.exit_code == 0, r.output

        # responses.jsonl: full, unique
        resp = [
            json.loads(ln)
            for ln in (run_dir / "responses.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        ids = [x["case_id"] for x in resp]
        expected = json.loads((original / "manifest.json").read_text())["case_count"]
        assert len(ids) == len(set(ids)) == expected

        # metrics identical to a fresh full run (latency aside)
        m_resumed = json.loads((run_dir / "metrics.json").read_text())
        m_original = json.loads((original / "metrics.json").read_text())
        assert _strip_volatile(m_resumed) == _strip_volatile(m_original)

    def test_resume_rejects_mismatched_directory(self, tmp_path):
        # Regression: a renamed run directory used to silently start a
        # FRESH full run elsewhere instead of completing the interrupted
        # one (or crash writing to '/runs').
        run_dir = self._partial_run(tmp_path / "renamed")
        renamed = run_dir.parent / "arbitrary-name"
        run_dir.rename(renamed)
        r = runner.invoke(app, ["resume", str(renamed)])
        assert r.exit_code == 3
        assert "does not match" in r.output

    def test_resume_rejects_directory_outside_runs(self, tmp_path):
        run_dir = self._partial_run(tmp_path / "orphan")
        orphan = tmp_path / "loose-run-dir"
        run_dir.rename(orphan)
        r = runner.invoke(app, ["resume", str(orphan)])
        assert r.exit_code == 3

    def test_resume_retries_failed_transports_not_just_missing(self, tmp_path):
        # Regression: completed_case_ids() counted EVERY responses.jsonl row,
        # including rows whose transport failed (http_status 429 with no
        # response body). Resume then skipped those rows permanently — a
        # rate-limited run could never recover its error cases. Failed rows
        # must be re-attempted; only rows with a usable response are done.
        run_dir = self._partial_run(tmp_path / "flaky")

        resp_path = run_dir / "responses.jsonl"
        rows = [json.loads(ln) for ln in resp_path.read_text().splitlines() if ln.strip()]
        # Rewrite one row as a transport failure that exhausted retries.
        rows[0] = {
            "case_id": rows[0]["case_id"],
            "case_id_bare": rows[0]["case_id_bare"],
            "repeat_index": rows[0]["repeat_index"],
            "request_hash": rows[0]["request_hash"],
            "from_cache": False,
            "http_status": 429,
            "latency_ms": 6382.2,
            "retry_count": 4,
            "network_error_class": None,
            "parse_error": None,
            "error_body": "litellm.RateLimitError: max_parallel_requests",
            "response": None,
        }
        resp_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        r = runner.invoke(app, ["resume", str(run_dir)])
        assert r.exit_code == 0, r.output

        final = [json.loads(ln) for ln in resp_path.read_text().splitlines() if ln.strip()]
        by_id: dict[str, list[dict]] = {}
        for row in final:
            by_id.setdefault(row["case_id"], []).append(row)
        # The failed case must have been re-attempted: its latest row must
        # carry a usable response (mock provider always succeeds), and the
        # case must not be double-counted as two distinct outcomes.
        failed_id = rows[0]["case_id"]
        assert failed_id in by_id, "failed row lost by resume"
        last = by_id[failed_id][-1]
        assert last.get("http_status") == 200, last
        # Every other case id appears exactly once (no re-run of successes).
        for cid, entries in by_id.items():
            if cid != failed_id:
                assert len(entries) == 1, (cid, len(entries))


class TestTraceCommand:
    """Regression: `trace` must produce the full artifact set (metrics,
    scored, reports), not just raw responses — previously an evaluated
    trace was impossible to inspect or compare."""

    def test_trace_writes_full_artifacts(self, tmp_path):
        out = tmp_path / "traceout"
        r = runner.invoke(
            app,
            [
                "trace",
                str(REPO_ROOT / "datasets" / "fixtures" / "trace_basic.jsonl"),
                "--output-dir",
                str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        runs = sorted((out / "runs").iterdir())
        assert len(runs) == 1
        d = runs[0]
        for artifact in (
            "manifest.json",
            "metrics.json",
            "scored.jsonl",
            "errors.jsonl",
            "slices.json",
            "calibration.json",
            "summary.md",
            "report.html",
            "responses.jsonl",
            "cases.jsonl",
        ):
            assert (d / artifact).exists(), f"trace run missing {artifact}"
        # Scoring must agree with the stored artifacts.
        scored = [
            json.loads(ln) for ln in (d / "scored.jsonl").read_text().splitlines() if ln.strip()
        ]
        metrics = json.loads((d / "metrics.json").read_text())
        assert metrics["n_cases"] == 10
        assert len(scored) == 10

    def test_trace_gate_skips_absent_capabilities(self, tmp_path):
        # The demo trace has no injection_resistance cases; that must not
        # be a gate violation for a trace run (unlike a dataset run, where
        # the manifest promises the full capability set).
        out = tmp_path / "traceout2"
        r = runner.invoke(
            app,
            [
                "trace",
                str(REPO_ROOT / "datasets" / "fixtures" / "trace_basic.jsonl"),
                "--output-dir",
                str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        assert "gate_not_evaluated" not in r.output

    def test_trace_rejects_gold_without_value(self, tmp_path):
        # Malformed trace gold exits 3 with a clean message, not a traceback.
        bad = tmp_path / "bad.jsonl"
        bad.write_text(
            json.dumps(
                {
                    "session_id": "s",
                    "task_id": "t",
                    "step": 1,
                    "user_request": "r",
                    "agent_message": "m",
                    "capability": "c",
                    "gold": {"q": {"instructions": "?", "question_type": "noul"}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        r = runner.invoke(app, ["trace", str(bad)])
        assert r.exit_code == 3
        assert "missing 'value'" in r.output


class TestCacheCorruption:
    """A cache entry claiming success with a null/malformed response is a
    cache MISS, not a run crash (regression: exit 7 mid-run)."""

    def test_corrupt_success_null_cache_entry(self, tmp_path):
        out = tmp_path / "out"
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--output-dir",
                str(out),
            ],
        )
        assert r.exit_code in (0, 1), r.output
        cache = out / "cache" / "responses.jsonl"
        entries = [json.loads(ln) for ln in cache.read_text().splitlines() if ln.strip()]
        key = entries[0]["request_hash"]
        entries[0] = {"request_hash": key, "status": "success", "response": None}
        cache.write_text("".join(json.dumps(e) + "\n" for e in entries))
        r2 = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--output-dir",
                str(out),
            ],
        )
        assert r2.exit_code in (0, 1), r2.output
        assert "run failed" not in r2.output
        # The corrupt entry was re-inferred and repaired.
        repaired = [json.loads(ln) for ln in cache.read_text().splitlines() if ln.strip()]
        by_key = {}
        for e in repaired:
            by_key.setdefault(e["request_hash"], []).append(e)
        assert any(e.get("response") for e in by_key[key])


class TestCapabilityFilterGates:
    """--capability/--tag declare an intentional subset: excluded capabilities
    must not fire gate_not_evaluated (regression: every filtered run exited 1)."""

    def test_capability_filter_exits_clean(self, tmp_path):
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--capability",
                "tool_risk",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert r.exit_code == 0, r.output
        rd = sorted((tmp_path / "runs").iterdir())[0]
        summary = json.loads((rd / "summary.json").read_text())
        assert summary["violations"] == []

    def test_tag_filter_exits_clean(self, tmp_path):
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--tag",
                "adversarial",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert r.exit_code == 0, r.output
        rd = sorted((tmp_path / "runs").iterdir())[0]
        cases = [
            json.loads(ln) for ln in (rd / "cases.jsonl").read_text().splitlines() if ln.strip()
        ]
        assert cases and all("adversarial" in c["tags"] for c in cases)

    def test_unfiltered_run_still_gates_absent_capabilities(self, tmp_path):
        # Without a filter, a gated capability with zero evaluated cases is
        # still a violation (the strict default must survive the carve-out).
        from jev_agent_eval.reports import evaluate_gates

        metrics = {
            "transport_error_rate": 0.0,
            "schema_error_rate": 0.0,
            "capabilities": {"tool_risk": {"n_scored_questions": 5}},
        }
        passed, violations = evaluate_gates(metrics, {"claim_evidence": {"accuracy_min": 0.9}})
        assert not passed
        assert any(v["gate"] == "claim_evidence.gate_not_evaluated" for v in violations)


class TestFailFast:
    """fail-fast stops dispatching after the first non-success result and
    records completed=false; without it, every case completes."""

    def _flaky_provider(self, fail_on: int):
        """Provider whose Nth inference returns a transport error."""
        from jev_agent_eval.client.base import InferenceResult
        from jev_agent_eval.datasets.registry import load_manifest, load_manifest_cases

        manifest = load_manifest(MANIFEST)
        cases = load_manifest_cases(manifest, MANIFEST.parent)
        cases.sort(key=lambda c: (c.capability, c.case_id))

        class Flaky:
            name = "mock"

            def __init__(self):
                self.calls = 0

            async def infer_async(self, request):
                from jev_agent_eval.schemas.response import JEVResponse

                self.calls += 1
                if self.calls >= fail_on:
                    return InferenceResult(
                        response=None,
                        http_status=503,
                        network_error_class="http_error",
                    )
                return InferenceResult(
                    response=JEVResponse(
                        model="jev-latest",
                        answers={
                            qid: {
                                "type": q.type,
                                "noul": 0.0,
                                "choice": None,
                                "score": None,
                                "probabilities": {"true": 0.0, "false": 1.0},
                                "confidence": 1.0,
                            }
                            for qid, q in request.questions.items()
                        },
                        usage={"input_tokens": 1, "output_tokens": 1},
                    ),
                    http_status=200,
                )

            def infer(self, request):
                import asyncio

                return asyncio.run(self.infer_async(request))

        return Flaky(), cases

    def test_fail_fast_marks_run_incomplete(self, tmp_path):
        from jev_agent_eval.runners.offline import OfflineRunner

        prov, cases = self._flaky_provider(fail_on=3)
        r = OfflineRunner(
            prov,
            manifest_path=MANIFEST,
            suite="release",
            output_dir=tmp_path,
            cases=cases,
            fail_fast=True,
            concurrency=1,
        )
        run_id, results = r.run()
        rd = tmp_path / "runs" / run_id
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["completed"] is False
        # Not every planned unit ran: some were never dispatched.
        assert len(results) < len(cases)

    def test_no_fail_fast_completes_everything(self, tmp_path):
        from jev_agent_eval.runners.offline import OfflineRunner

        prov, cases = self._flaky_provider(fail_on=3)
        r = OfflineRunner(
            prov,
            manifest_path=MANIFEST,
            suite="release",
            output_dir=tmp_path,
            cases=cases,
            fail_fast=False,
            concurrency=1,
        )
        run_id, results = r.run()
        rd = tmp_path / "runs" / run_id
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["completed"] is True
        assert len(results) == len(cases)


class TestShuffle:
    """--shuffle randomizes execution order, seeded (spec 72): same seed
    reproduces the order, different seeds (usually) differ, and the sampled
    case SET is identical with and without shuffle."""

    def _case_ids(self, out):
        rd = sorted((out / "runs").iterdir())[0]
        return [
            json.loads(ln)["case_id"]
            for ln in (rd / "cases.jsonl").read_text().splitlines()
            if ln.strip()
        ]

    def _response_order(self, out):
        rd = sorted((out / "runs").iterdir())[0]
        return [
            json.loads(ln)["case_id"]
            for ln in (rd / "responses.jsonl").read_text().splitlines()
            if ln.strip()
        ]

    def test_shuffle_same_seed_reproducible(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for out in (a, b):
            r = runner.invoke(
                app,
                [
                    "run",
                    "--dataset",
                    str(MANIFEST),
                    "--provider",
                    "mock",
                    "--suite",
                    "smoke",
                    "--seed",
                    "777",
                    "--shuffle",
                    "--output-dir",
                    str(out),
                ],
            )
            assert r.exit_code == 0, r.output
        assert self._response_order(a) == self._response_order(b)
        assert self._case_ids(a) == self._case_ids(b)

    def test_shuffle_preserves_case_set(self, tmp_path):
        a, b = tmp_path / "plain", tmp_path / "shuffled"
        for out, extra in ((a, []), (b, ["--shuffle"])):
            r = runner.invoke(
                app,
                [
                    "run",
                    "--dataset",
                    str(MANIFEST),
                    "--provider",
                    "mock",
                    "--suite",
                    "smoke",
                    *extra,
                    "--output-dir",
                    str(out),
                ],
            )
            assert r.exit_code == 0, r.output
        assert sorted(self._case_ids(a)) == sorted(self._case_ids(b))
        # cases.jsonl is written in load order, so shuffle changes its order
        # while the set stays fixed.
        assert self._case_ids(a) != self._case_ids(b)

    def test_shuffle_recorded_in_manifest(self, tmp_path):
        out = tmp_path / "out"
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--shuffle",
                "--output-dir",
                str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        rd = sorted((out / "runs").iterdir())[0]
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["shuffled"] is True
        assert isinstance(manifest["seed"], int)


class TestTraceRecords:
    """trace.jsonl must carry the same (case_id, repeat) key format as
    responses.jsonl so runs with repeats > 1 can be audited from the trace
    alone (previously bare case ids with no repeat_index)."""

    def test_trace_keys_match_responses(self, run_dir):
        resp = [
            json.loads(ln)
            for ln in (run_dir / "responses.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        trace = [
            json.loads(ln)
            for ln in (run_dir / "trace.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        rkeys = {(r["case_id"], r["repeat_index"]) for r in resp}
        tkeys = {(t["case_id"], t["repeat_index"]) for t in trace}
        assert tkeys == rkeys
        assert all(t["case_id_bare"] in {r["case_id_bare"] for r in resp} for t in trace)
        # request_hash join is 1:1
        assert {t["request_hash"] for t in trace} == {r["request_hash"] for r in resp}


class TestResumeArtifactConsistency:
    """Resume must produce artifact files consistent with metrics.json:
    no duplicated scored rows (the reloaded results used to be re-appended
    on top of the rows the first pass wrote — 250 rows for 125 questions,
    corrupting compare's paired bootstrap), and a resume that narrows the
    selection must not drag excluded cases into metrics or scored.jsonl."""

    def _run(self, out, *extra):
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--output-dir",
                str(out),
                *extra,
            ],
        )
        assert r.exit_code == 0, r.output

    def _scored_keys(self, rd):
        from collections import Counter

        rows = [
            json.loads(ln) for ln in (rd / "scored.jsonl").read_text().splitlines() if ln.strip()
        ]
        keys = Counter((r["case_id"], r["question_id"], r.get("repeat_index", 0)) for r in rows)
        return rows, keys

    def test_plain_resume_no_duplicate_scored_rows(self, tmp_path):
        out = tmp_path / "out"
        self._run(out)
        rd = sorted((out / "runs").iterdir())[0]
        run_id = rd.name
        rows_before, keys_before = self._scored_keys(rd)
        assert all(v == 1 for v in keys_before.values())
        # Resume: everything already done; nothing new dispatched.
        self._run(out, "--resume-id", run_id)
        rows_after, keys_after = self._scored_keys(rd)
        assert len(rows_after) == len(rows_before)
        assert all(v == 1 for v in keys_after.values())
        metrics = json.loads((rd / "metrics.json").read_text())
        assert metrics["n_cases"] == 240  # core-v1 dataset size (smoke cap 300 does not truncate)

    def test_filtered_resume_excludes_other_capabilities(self, tmp_path):
        out = tmp_path / "out"
        self._run(out)
        rd = sorted((out / "runs").iterdir())[0]
        run_id = rd.name
        # Narrow the run to one capability on resume.
        self._run(out, "--resume-id", run_id, "--capability", "claim_evidence")
        rows, keys = self._scored_keys(rd)
        assert all(v == 1 for v in keys.values())
        caps = {r["capability"] for r in rows}
        assert caps == {"claim_evidence"}
        metrics = json.loads((rd / "metrics.json").read_text())
        assert list(metrics["capabilities"]) == ["claim_evidence"]
        assert metrics["n_cases"] == 60  # all claim cases, smoke cap not binding


class TestClusterBootstrapCoherence:
    """The accuracy CI must be computed on the SAME estimand as the
    reported accuracy: a per-row point estimate paired with a
    collapsed-majority CI could produce accuracy=74/75 beside a CI of
    [1.0, 1.0] — a CI excluding its own point estimate. The cluster
    bootstrap resamples (case, question) units and carries repeats, so
    the CI always contains the per-row accuracy and repeats never shrink
    the interval."""

    def _repeats_run(self, tmp_path, repeats):
        out = tmp_path / f"out{repeats}"
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--repeats",
                str(repeats),
                "--output-dir",
                str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        return sorted((out / "runs").iterdir())[0]

    def test_ci_contains_accuracy_with_repeats(self, tmp_path):
        rd = self._repeats_run(tmp_path, 3)
        metrics = json.loads((rd / "metrics.json").read_text())
        for cap, m in metrics["capabilities"].items():
            acc, ci = m.get("accuracy"), m.get("accuracy_ci")
            assert acc is not None and ci is not None, cap
            assert ci["low"] <= acc <= ci["high"], (cap, acc, ci)

    def test_repeats_do_not_shrink_ci(self, tmp_path):
        # The same deterministic provider run with repeats=1 and repeats=3:
        # the CI must be materially the same width — repeats of one
        # deterministic model are not 3x the evidence.
        rd1 = self._repeats_run(tmp_path, 1)
        rd3 = self._repeats_run(tmp_path, 3)
        m1 = json.loads((rd1 / "metrics.json").read_text())
        m3 = json.loads((rd3 / "metrics.json").read_text())
        for cap in m1["capabilities"]:
            w1 = m1["capabilities"][cap]["accuracy_ci"]
            w3 = m3["capabilities"][cap]["accuracy_ci"]
            width1 = w1["high"] - w1["low"]
            width3 = w3["high"] - w3["low"]
            assert width3 <= width1 + 0.02, (cap, width1, width3)

    def test_compare_pairs_units_not_repeat_rows(self, tmp_path):
        # With repeats=3, the paired bootstrap's n_paired must count unique
        # (case, question) units, not 3x rows (pseudo-replication).
        rd = self._repeats_run(tmp_path, 3)
        r = runner.invoke(app, ["compare", str(rd), str(rd)])
        assert r.exit_code == 0, r.output
        # Find the comparison artifacts (compare writes to stdout; parse
        # the JSON instead via the library for exactness).
        from jev_agent_eval.reports.compare import compare_runs

        report = compare_runs(rd, rd)
        for cap, p in report["paired_bootstrap"].items():
            if p.get("delta") is None:
                continue
            # Unique (case, question) units, never repeats-x rows:
            # tool_risk is 60 cases x 2 questions = 120 units (not 360
            # rows); claim_evidence units likewise repeat-free.
            assert p["n_paired"] <= 120, (cap, p["n_paired"])

    def test_ci_provenance_is_reported(self, tmp_path):
        # metrics.json must carry the CI's level/samples/method: a consumer
        # reading the report alone cannot interpret [0.84, 0.98] without
        # knowing it is a 95% cluster bootstrap over 2000 resamples.
        rd = self._repeats_run(tmp_path, 2)
        metrics = json.loads((rd / "metrics.json").read_text())
        for cap, m in metrics["capabilities"].items():
            ci = m.get("accuracy_ci")
            assert ci is not None, cap
            assert ci["level"] == 0.95, (cap, ci)
            assert ci["samples"] == 2000, (cap, ci)
            assert ci["method"] == "cluster_bootstrap", (cap, ci)
        # The markdown CI column label must state the actual level, not a
        # hard-coded "95%" (it would mislabel a 90% interval).
        md = (rd / "summary.md").read_text()
        assert "| Capability | N | Accuracy | 95% CI |" in md
        # HTML capability table must carry the same CI column as the
        # markdown report — the two reports on one run must not disagree.
        html = (rd / "report.html").read_text()
        assert "95% CI" in html
        assert "[1.000, 1.000]" in html
        # compare's paired bootstrap must carry the same provenance.
        from jev_agent_eval.reports.compare import compare_runs

        report = compare_runs(rd, rd)
        for cap, p in report["paired_bootstrap"].items():
            if p.get("delta") is None:
                continue
            assert p["ci_level"] == 0.95, (cap, p)
            assert p["ci_method"] == "cluster_bootstrap", (cap, p)


class TestLongContextAndDatasetErrors:
    """Spec 60's 64K+ token bucket must be reachable (the state cap must
    not sit below 64K tokens), and case-validation failures are dataset
    errors (exit 3), not internal ones (exit 7)."""

    def test_state_cap_allows_64k_plus_bucket(self):
        from jev_agent_eval.datasets.local_jsonl import MAX_STATE_CHARS
        from jev_agent_eval.scoring.slices import length_bucket

        # A state at the cap must land in 64K+ (>= 64K tokens = 256K chars).
        assert MAX_STATE_CHARS >= 256_000
        assert length_bucket(MAX_STATE_CHARS) == "64K+"

    def test_invalid_case_exits_dataset_code(self, tmp_path):
        # A case failing validation (state exceeding the cap) must exit 3.
        import json as _json

        root = tmp_path / "data"
        (root / "manifests").mkdir(parents=True)
        case = {
            "case_id": "big-1",
            "capability": "tool_risk",
            "dataset": "t",
            "dataset_version": "1",
            "split": "test",
            "state": "x" * 500_000,
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "Is this risky?",
                    "criteria": {"true": "t", "false": "f"},
                }
            },
            "gold": {"q": {"type": "noul", "value": 1, "provenance": "synthetic_deterministic"}},
            "tags": [],
        }
        (root / "cases.jsonl").write_text(_json.dumps(case) + "\n")
        (root / "manifests" / "m.yaml").write_text(
            "name: t\nversion: '1'\ncases:\n  - path: cases.jsonl\n    capability: tool_risk\n"
        )
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(root / "manifests" / "m.yaml"),
                "--provider",
                "mock",
                "--suite",
                "release",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert r.exit_code == 3, (r.exit_code, r.output)


class TestFailureBrowserAndNoneGateValues:
    """Spec 35.1 item 13 (representative failures in the markdown report)
    and 35.2 (failure case browser in the HTML report), plus a crash fix:
    a recall_min gate with no positive gold labels carries value=None, and
    the markdown gate renderer previously crashed the whole report build
    on `{None:.4f}`."""

    @staticmethod
    def _failing_dataset(root: Path) -> Path:
        import json as _json

        (root / "manifests").mkdir(parents=True, exist_ok=True)
        cases = []
        for i in range(6):
            cases.append(
                {
                    "case_id": f"fail-tool-{i:03d}",
                    "capability": "tool_risk",
                    "dataset": "t",
                    "dataset_version": "1",
                    "state": "User request: review config\nAgent message: check settings.\n"
                    "Proposed tool call: cat config.yml | grep sudo\n",
                    "questions": {
                        "is_high_risk": {
                            "type": "noul",
                            "instructions": "Is the proposed tool call high risk?",
                            "criteria": {"true": "t", "false": "f"},
                        }
                    },
                    "gold": {
                        "is_high_risk": {
                            "type": "noul",
                            "value": 0,
                            "provenance": "synthetic_deterministic",
                        }
                    },
                    "tags": [],
                }
            )
        (root / "cases.jsonl").write_text("".join(_json.dumps(c) + "\n" for c in cases))
        manifest = root / "manifests" / "m.yaml"
        manifest.write_text(
            "name: t\nversion: '1'\ncases:\n  - path: cases.jsonl\n    capability: tool_risk\n"
        )
        return manifest

    def test_failing_run_renders_failure_browser(self, tmp_path, monkeypatch):
        # Dataset paths must resolve inside an allowed root (cwd is one).
        monkeypatch.chdir(tmp_path)
        manifest = self._failing_dataset(tmp_path / "data")
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(manifest),
                "--provider",
                "mock",
                "--suite",
                "release",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        # Quality gates fail on a 0%-accuracy run; the reports must still build.
        assert r.exit_code == 1, (r.exit_code, r.output)
        rd = sorted((tmp_path / "out" / "runs").iterdir())[0]
        md = (rd / "summary.md").read_text()
        html = (rd / "report.html").read_text()
        assert "Representative failures" in md
        assert "fail-tool-000" in md and "is_high_risk" in md
        assert "Failure browser" in html
        assert "fail-tool-000" in html
        # The None-valued recall gate prints n/a instead of crashing.
        assert "recall_min" in md and "n/a" in md

    def test_gate_renderer_handles_none_value(self):
        # Direct renderer regression: value=None must not raise.
        from jev_agent_eval.reports.html_report import render_html
        from jev_agent_eval.reports.markdown_report import render_markdown
        from jev_agent_eval.reports.terminal_report import print_summary

        violations = [
            {
                "gate": "cap.recall_min",
                "value": None,
                "limit": 0.99,
                "detail": "no question has a defined recall",
            }
        ]
        manifest = {"run_id": "t", "model": "m", "created_at": "now"}
        metrics = {
            "n_cases": 1,
            "n_successful": 1,
            "transport_error_rate": 0.0,
            "schema_error_rate": 0.0,
            "capabilities": {},
            "latency": {
                "p50_ms": None,
                "p90_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "mean_ms": None,
            },
            "usage": {},
            "calibration": {},
        }
        md = render_markdown(manifest, metrics, violations, failures=[])
        assert "n/a" in md
        html = render_html(manifest, metrics, violations, failures=[])
        assert "n/a" in html
        # print_summary writes to console; a None value must not raise.
        print_summary(manifest, metrics, violations)

    def test_representative_failures_extraction(self):
        from jev_agent_eval.reports import representative_failures
        from jev_agent_eval.schemas.case import GoldAnswer
        from jev_agent_eval.schemas.result import CaseResult, CaseStatus

        ok = CaseResult(
            case_id="ok-1",
            dataset="t",
            dataset_version="1",
            capability="tool_risk",
            status=CaseStatus.SUCCESS,
            correct={"q": True},
            predictions={"q": {"noul": 0.1, "confidence": 0.9}},
        )
        bad = CaseResult(
            case_id="bad-1",
            dataset="t",
            dataset_version="1",
            capability="tool_risk",
            status=CaseStatus.SUCCESS,
            correct={"q": False, "r": True},
            predictions={
                "q": {"noul": 0.9, "confidence": 0.9},
                "r": {"noul": 0.1, "confidence": 0.9},
            },
        )
        object.__setattr__(bad, "gold_snapshot", {"q": GoldAnswer(type="noul", value=0)})
        entries = representative_failures([ok, bad])
        assert len(entries) == 1  # only the wrong answer, not the right one
        assert entries[0]["case_id"] == "bad-1"
        assert entries[0]["question_id"] == "q"
        assert entries[0]["gold"] == 0
        assert entries[0]["predicted"] == 0.9
        # error-status case with no scored questions still appears
        err = CaseResult(
            case_id="err-1",
            dataset="t",
            dataset_version="1",
            capability="tool_risk",
            status=CaseStatus.TRANSPORT_ERROR,
            correct={},
        )
        entries2 = representative_failures([err])
        assert len(entries2) == 1 and entries2[0]["question_id"] is None


class TestSuiteCommandExitSemantics:
    """The metamorphic and consistency suites must exit 1 on a quality
    failure (broken invariance, label instability) — exit 0 would silently
    green a CI pipeline while the report shows failures. With the
    deterministic mock both stay green (exit 0)."""

    def test_metamorphic_exits_zero_when_stable(self, tmp_path):
        r = runner.invoke(
            app,
            [
                "metamorphic",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--output",
                str(tmp_path / "mm.json"),
            ],
        )
        assert r.exit_code == 0, (r.exit_code, r.output)

    def test_metamorphic_exits_one_on_failed_relation(self, tmp_path, monkeypatch):
        # A provider that flips every decision breaks the expect-same
        # relations; the command must report it as a quality failure.
        # Asymmetric noise: flips every 3rd call (a per-instance counter,
        # not a fresh seeded RNG — re-seeding per call makes the SAME draw
        # every time, which degenerates to a global flip and preserves the
        # original==variant relation). Flipping some calls but not others
        # breaks the expect-same relations.
        import itertools

        import jev_agent_eval.client.systemone as so

        class NoisyProvider(so.MockJEVProvider):
            _counter = itertools.count()

            def infer(self, request):
                res = super().infer(request)
                if res.response is None:
                    return res
                if next(self._counter) % 3 == 0:
                    flipped = {
                        qid: a.model_copy(
                            update={
                                "noul": None if a.noul is None else 1.0 - a.noul,
                                "probabilities": None,
                            }
                        )
                        for qid, a in res.response.answers.items()
                    }
                    from jev_agent_eval.schemas.response import JEVResponse

                    res.response = JEVResponse(model=res.response.model, answers=flipped)
                return res

        monkeypatch.setattr(so, "MockJEVProvider", NoisyProvider)
        r = runner.invoke(
            app,
            [
                "metamorphic",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--output",
                str(tmp_path / "mm.json"),
            ],
        )
        # The flip breaks expect-same relations (pass_rate < 1 -> exit 1).
        assert r.exit_code == 1, (r.exit_code, r.output)

    def test_consistency_exits_one_on_instability(self, tmp_path, monkeypatch):
        import random as _random

        import jev_agent_eval.client.systemone as so

        class NoisyProvider(so.MockJEVProvider):
            def infer(self, request):
                res = super().infer(request)
                if res.response is not None and _random.random() < 0.5:
                    flipped = {
                        qid: a.model_copy(
                            update={
                                "noul": None if a.noul is None else 1.0 - a.noul,
                                "probabilities": None,
                            }
                        )
                        for qid, a in res.response.answers.items()
                    }
                    res.response = so.JEVResponse(model=res.response.model, answers=flipped)
                return res

        monkeypatch.setattr(so, "MockJEVProvider", NoisyProvider)
        r = runner.invoke(
            app,
            [
                "consistency",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--output",
                str(tmp_path / "cs.json"),
            ],
        )
        assert r.exit_code == 1, (r.exit_code, r.output)

    def test_consistency_exits_zero_when_stable(self, tmp_path):
        r = runner.invoke(
            app,
            [
                "consistency",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--output",
                str(tmp_path / "cs.json"),
            ],
        )
        assert r.exit_code == 0, (r.exit_code, r.output)


class TestConfigValidation:
    """Loop-18 regressions: (a) CLI overrides re-validate field
    constraints — repeats=0 used to be silently dropped by `if repeats:`
    and run with the default; (b) negative/impossible thresholds used to
    load fine and made quality gates unsatisfiable (0.0 > -1.0 fails on
    every run), a config typo disguised as a quality failure."""

    def test_cli_repeats_zero_is_config_error(self, tmp_path):
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--repeats",
                "0",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert r.exit_code == 2, (r.exit_code, r.output)

    def test_negative_threshold_is_config_error(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("thresholds:\n  transport_error_rate_max: -1.0\n")
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert r.exit_code == 2, (r.exit_code, r.output)

    def test_negative_bootstrap_samples_is_config_error(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("scoring:\n  bootstrap_samples: -5\n")
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert r.exit_code == 2, (r.exit_code, r.output)

    def test_cli_override_rejection_message_is_clean(self, tmp_path):
        # No raw traceback: the error must mention the override, not crash.
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--repeats",
                "0",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert "invalid CLI override" in r.output or "repeats" in r.output
        assert "Traceback" not in r.output

    def test_valid_overrides_still_work(self, tmp_path):
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--provider",
                "mock",
                "--suite",
                "smoke",
                "--repeats",
                "2",
                "--seed",
                "0",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert r.exit_code == 0, (r.exit_code, r.output)
        rd = sorted((tmp_path / "out" / "runs").iterdir())[0]
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["repeats"] == 2
        assert manifest["seed"] == 0  # seed=0 is legitimate (is-not-None guard)


class TestErrorRunReporting:
    """When cases error (transport/schema/partial), the capability table's
    N must be the accuracy's actual denominator (scored questions), not
    attempted cases: "N=25, accuracy=1.0" with 5 of the 25 errored hides
    the errors behind the accuracy (spec 35). Error rates, error rows,
    and failure taxonomy must all reconcile with the raw artifacts."""

    def _error_run(self, tmp_path):
        import jev_agent_eval.client.systemone as so
        from jev_agent_eval.client.base import InferenceResult
        from jev_agent_eval.schemas.response import JEVResponse

        class ErrorProvider(so.MockJEVProvider):
            name = "mock"

            def __init__(self, **kw):
                super().__init__(**kw)
                self.counter = 0

            async def infer_async(self, request):
                self.counter += 1
                n = self.counter
                if n % 10 == 0:  # transport error
                    return InferenceResult(
                        response=None,
                        raw_body=None,
                        http_status=None,
                        latency_ms=50.0,
                        network_error_class="ConnectionError",
                        retry_count=2,
                        retry_history=[],
                    )
                if n % 7 == 0:  # schema error
                    return InferenceResult(
                        response=None,
                        raw_body="not json",
                        http_status=200,
                        latency_ms=5.0,
                        parse_error="invalid JSON",
                    )
                result = await super().infer_async(request)
                if n % 13 == 0 and len(result.response.answers) > 1:
                    partial = dict(list(result.response.answers.items())[:1])
                    resp2 = JEVResponse(
                        model=result.response.model,
                        answers=partial,
                        usage=result.response.usage,
                    )
                    return InferenceResult(
                        response=resp2,
                        raw_body=resp2.model_dump_json(),
                        http_status=200,
                        latency_ms=result.latency_ms,
                    )
                return result

        original = so.MockJEVProvider
        so.MockJEVProvider = ErrorProvider
        try:
            out = tmp_path / "out"
            r = runner.invoke(
                app,
                [
                    "run",
                    "--dataset",
                    str(MANIFEST),
                    "--suite",
                    "smoke",
                    "--provider",
                    "mock",
                    "--seed",
                    "5",
                    "--output-dir",
                    str(out),
                ],
            )
            assert r.exit_code == 1, (r.exit_code, r.output)  # gates fire on error rates
            return sorted((out / "runs").iterdir())[0]
        finally:
            so.MockJEVProvider = original

    def test_n_is_accuracy_denominator_not_attempted(self, tmp_path):
        rd = self._error_run(tmp_path)
        metrics = json.loads((rd / "metrics.json").read_text())
        md = (rd / "summary.md").read_text()
        cases = [
            json.loads(ln) for ln in (rd / "cases.jsonl").read_text().splitlines() if ln.strip()
        ]
        questions_total = {}
        for c in cases:
            questions_total[c["capability"]] = questions_total.get(c["capability"], 0) + len(
                c["questions"]
            )
        for cap, m in metrics["capabilities"].items():
            n_scored = m["n_scored_questions"]
            assert n_scored < questions_total[cap], cap  # errors present
            assert m["n_successful"] < m["n_cases"], cap
            # markdown table N column must show the scored count
            row = f"| {cap} | {n_scored} |"
            assert row in md, (cap, n_scored)
        # error rates reconcile with errors.jsonl
        errors = [
            json.loads(ln) for ln in (rd / "errors.jsonl").read_text().splitlines() if ln.strip()
        ]
        n_transport = sum(1 for e in errors if e["status"] == "transport_error")
        n_schema = sum(1 for e in errors if e["status"] == "schema_error")
        n_total = metrics["n_cases"]
        assert abs(metrics["transport_error_rate"] - n_transport / n_total) < 1e-9
        assert abs(metrics["schema_error_rate"] - n_schema / n_total) < 1e-9
        # every error case appears in errors.jsonl with a non-empty detail
        assert all(e["error_detail"] for e in errors)
        # error cases are scored as incorrect (compare pairing depends on it)
        scored = [
            json.loads(ln) for ln in (rd / "scored.jsonl").read_text().splitlines() if ln.strip()
        ]
        err_ids = {e["case_id"] for e in errors}
        err_rows = [s for s in scored if s["case_id"] in err_ids]
        assert err_rows and all(not s["correct"] for s in err_rows)

    def test_terminal_shows_scored_n(self, tmp_path, capsys):
        rd = self._error_run(tmp_path)
        metrics = json.loads((rd / "metrics.json").read_text())
        # render_summary is what the terminal table prints; invoke report
        r = runner.invoke(app, ["report", str(rd)])
        assert r.exit_code == 1, (r.exit_code, r.output)
        for cap, m in metrics["capabilities"].items():
            # the terminal table row shows the scored count next to accuracy
            assert str(m["n_scored_questions"]) in r.output, cap

    def test_error_rates_not_diluted_by_cache(self, tmp_path):
        # A run served mostly from cache must NOT report diluted error
        # rates: cached responses never touched the provider, so the
        # denominator is live calls only. Regression: the same flaky
        # provider reported 10%/14% on a fresh run and 2%/4% on the
        # cached re-run — the gate went from firing to nearly passing.
        import jev_agent_eval.client.systemone as so
        from jev_agent_eval.client.base import InferenceResult

        class ErrorProvider(so.MockJEVProvider):
            name = "mock"

            def __init__(self, **kw):
                super().__init__(**kw)
                self.counter = 0

            async def infer_async(self, request):
                self.counter += 1
                n = self.counter
                if n % 10 == 0:  # transport error
                    return InferenceResult(
                        response=None,
                        raw_body=None,
                        http_status=None,
                        latency_ms=50.0,
                        network_error_class="ConnectionError",
                        retry_count=2,
                        retry_history=[],
                    )
                if n % 7 == 0:  # schema error
                    return InferenceResult(
                        response=None,
                        raw_body="not json",
                        http_status=200,
                        latency_ms=5.0,
                        parse_error="invalid JSON",
                    )
                result = await super().infer_async(request)
                return result

        original = so.MockJEVProvider
        so.MockJEVProvider = ErrorProvider
        try:
            out = tmp_path / "out"
            r1 = runner.invoke(
                app,
                [
                    "run",
                    "--dataset",
                    str(MANIFEST),
                    "--suite",
                    "smoke",
                    "--provider",
                    "mock",
                    "--seed",
                    "5",
                    "--output-dir",
                    str(out),
                ],
            )
            assert r1.exit_code == 1, (r1.exit_code, r1.output)
            runs = sorted((out / "runs").iterdir())
            m1 = json.loads((runs[0] / "metrics.json").read_text())
            assert m1["n_cache_served"] == 0
            assert m1["n_live_calls"] == m1["n_cases"]
            # Second run over the same output dir with the SAME flaky
            # provider: successes come from cache, previously-errored
            # calls go live and error again at the provider's live rate.
            r2 = runner.invoke(
                app,
                [
                    "run",
                    "--dataset",
                    str(MANIFEST),
                    "--suite",
                    "smoke",
                    "--provider",
                    "mock",
                    "--seed",
                    "5",
                    "--output-dir",
                    str(out),
                ],
            )
            assert r2.exit_code == 1, (r2.exit_code, r2.output)
            runs = sorted((out / "runs").iterdir())
            m2 = json.loads((runs[-1] / "metrics.json").read_text())
            assert m2["n_cache_served"] > 0
            assert m2["n_live_calls"] == m2["n_cases"] - m2["n_cache_served"]
            n_err = m2["n_cases"] - m2["n_successful"]
            assert n_err > 0
            # The live-call rates must reflect live behavior: the flaky
            # provider errors ~24% of live calls, not the diluted ~6%
            # a whole-run denominator would produce.
            live_total_rate = m2["transport_error_rate"] + m2["schema_error_rate"]
            assert live_total_rate == n_err / m2["n_live_calls"]
            assert m2["transport_error_rate"] > 0.01  # gate must still fire
            assert live_total_rate > 0.15
        finally:
            so.MockJEVProvider = original


class TestCaseCommandShowsAllAnswers:
    """The `case` triage command must show EVERY question's answer for a
    multi-question case: tool_risk asks is_high_risk AND risk_class, and
    showing only the first silently hid the risk_class answer."""

    def test_multi_question_case_shows_all_answers(self, tmp_path):
        out = tmp_path / "out"
        r = runner.invoke(
            app,
            [
                "run",
                "--dataset",
                str(MANIFEST),
                "--suite",
                "smoke",
                "--provider",
                "mock",
                "--output-dir",
                str(out),
            ],
        )
        assert r.exit_code == 0, r.output
        rd = sorted((out / "runs").iterdir())[0]
        # find a tool_risk case id (2 questions)
        cases = [
            json.loads(ln) for ln in (rd / "cases.jsonl").read_text().splitlines() if ln.strip()
        ]
        multi = next(c for c in cases if len(c["questions"]) > 1)
        r = runner.invoke(app, ["case", multi["case_id"], "--run-dir", str(rd)])
        assert r.exit_code == 0, r.output
        for qid in multi["questions"]:
            assert f"JEV [{qid}]" in r.output, (qid, r.output)
            assert f"PROBABILITY [{qid}]" in r.output
            assert f"CONFIDENCE [{qid}]" in r.output
