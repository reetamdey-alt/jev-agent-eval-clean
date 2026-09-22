"""Run artifact writer (spec section 35).

Layout per run:
    reports/runs/<run_id>/
    ├── manifest.json, summary.json, summary.md, report.html
    ├── cases.jsonl, responses.jsonl, scored.jsonl, errors.jsonl
    ├── metrics.json, slices.json, calibration.json, trace.jsonl
"""

from __future__ import annotations

import itertools
import json
import secrets
from pathlib import Path
from typing import Any

from jev_agent_eval.utils.timing import utc_now_iso


class ArtifactStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Ensure every documented artifact exists even when empty (spec 35/54):
        # errors.jsonl, trace.jsonl and episodes.jsonl are part of the run
        # layout contract.
        for name in ("errors.jsonl", "episodes.jsonl"):
            path = self.run_dir / name
            path.touch(exist_ok=True)

    # -- JSONL append streams -------------------------------------------------

    def append_case(self, case_dict: dict[str, Any]) -> None:
        self._append("cases.jsonl", case_dict)

    def append_response(self, response_record: dict[str, Any]) -> None:
        self._append("responses.jsonl", response_record)

    def append_scored(self, scored_record: dict[str, Any]) -> None:
        self._append("scored.jsonl", scored_record)

    def write_scored(self, records: list[dict[str, Any]]) -> None:
        """Replace scored.jsonl wholesale (replay recomputes it)."""
        self._write_jsonl("scored.jsonl", records)

    def write_errors(self, records: list[dict[str, Any]]) -> None:
        """Replace errors.jsonl wholesale (replay recomputes it)."""
        self._write_jsonl("errors.jsonl", records)

    def append_error(self, error_record: dict[str, Any]) -> None:
        self._append("errors.jsonl", error_record)

    def append_trace(self, trace_record: dict[str, Any]) -> None:
        self._append("trace.jsonl", trace_record)

    # -- Whole-file JSON ------------------------------------------------------

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self._write_json("manifest.json", manifest)

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json("summary.json", summary)

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        self._write_json("metrics.json", metrics)

    def write_slices(self, slices: dict[str, Any]) -> None:
        self._write_json("slices.json", slices)

    def write_calibration(self, calibration: dict[str, Any]) -> None:
        self._write_json("calibration.json", calibration)

    # -- v2 report artifacts (spec 54) ----------------------------------------

    def write_thresholds(self, thresholds: dict[str, Any]) -> None:
        self._write_json("thresholds.json", thresholds)

    def write_robustness(self, robustness: dict[str, Any]) -> None:
        self._write_json("robustness.json", robustness)

    def write_performance(self, performance: dict[str, Any]) -> None:
        self._write_json("performance.json", performance)

    def write_provenance(self, provenance: dict[str, Any]) -> None:
        self._write_json("provenance.json", provenance)

    def write_security(self, security: dict[str, Any]) -> None:
        self._write_json("security.json", security)

    def write_scorecard(self, scorecard: dict[str, Any]) -> None:
        self._write_json("scorecard.json", scorecard)

    def write_baselines(self, baselines: dict[str, Any]) -> None:
        self._write_json("baselines.json", baselines)

    def write_episodes(self, records: list[dict[str, Any]]) -> None:
        """Replace episodes.jsonl wholesale."""
        self._write_jsonl("episodes.jsonl", records)

    def write_risk_coverage(self, risk_coverage: dict[str, Any]) -> None:
        self._write_json("risk_coverage.json", risk_coverage)

    def write_markdown(self, text: str) -> None:
        (self.run_dir / "summary.md").write_text(text, encoding="utf-8")

    def write_html(self, html: str) -> None:
        (self.run_dir / "report.html").write_text(html, encoding="utf-8")

    # -- Reading (for replay/resume) -------------------------------------------

    def read_manifest(self) -> dict[str, Any]:
        return self._read_json("manifest.json")

    def iter_cases(self):
        yield from self._iter("cases.jsonl")

    def iter_responses(self):
        yield from self._iter("responses.jsonl")

    def iter_scored(self):
        yield from self._iter("scored.jsonl")

    def completed_case_ids(self) -> set[str]:
        """Case ids already recorded in responses.jsonl (for resume).

        Only rows that ended in a usable response count as done: a row whose
        transport failed (http_status != 200, e.g. a 429 that exhausted its
        retry budget) must be re-attempted on resume, not skipped — otherwise
        resume would permanently freeze the error rows of a rate-limited run.
        """
        done: set[str] = set()
        for rec in self._iter("responses.jsonl"):
            if "case_id" not in rec:
                continue
            # Rows carry either a parsed response body or a transport error.
            # A missing/failed http status means the record predates the
            # status field; treat it as done only if a body is present.
            status = rec.get("http_status")
            if status is not None and status != 200:
                continue
            if rec.get("error_body") and not rec.get("response"):
                continue
            done.add(rec["case_id"])
        return done

    # -- Internals ---------------------------------------------------------------

    def _append(self, name: str, obj: dict[str, Any]) -> None:
        with open(self.run_dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    def _write_json(self, name: str, obj: dict[str, Any]) -> None:
        (self.run_dir / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    def _write_jsonl(self, name: str, records: list[dict[str, Any]]) -> None:
        with open(self.run_dir / name, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _read_json(self, name: str) -> dict[str, Any]:
        with open(self.run_dir / name, encoding="utf-8") as f:
            return json.load(f)

    def _iter(self, name: str):
        path = self.run_dir / name
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # A hard kill mid-write can leave one truncated final line;
                # dropping it keeps resume working (the case is simply
                # re-executed) instead of crashing the resumed run.
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def new_run_id() -> str:
    """Timestamp + per-process monotonic counter + random suffix.

    The counter makes ids sort chronologically even for runs created within
    the same second, so `runs[0]` in a sorted listing is always the earliest
    run. Example: 2026-09-20T02-00-00Z_0000_8f3a1c.
    """
    stamp = utc_now_iso().replace(":", "-")
    return f"{stamp}_{next(_RUN_COUNTER):04d}_{secrets.token_hex(3)}"


_RUN_COUNTER = itertools.count()
