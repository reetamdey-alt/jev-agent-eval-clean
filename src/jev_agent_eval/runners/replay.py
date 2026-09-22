"""Replay runner (spec sections 13.2, 46).

Recomputes scoring and reports from a run directory without contacting JEV.
Replay never requires a network connection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_agent_eval.schemas.response import JEVResponse
from jev_agent_eval.schemas.result import CaseStatus, TransportRecord
from jev_agent_eval.storage.artifacts import ArtifactStore


def _parse_raw_response(raw: Any) -> dict[str, Any] | None:
    """responses.jsonl stores the (redacted) raw body as JSON text; parse
    it back to the object form CaseResult.raw_response expects. Returns
    None for absent or unparseable bodies (never a bare string, which the
    schema forbids)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def load_run_results(run_dir: str | Path) -> tuple[dict[str, Any], list[Any]]:
    """Reconstruct case results from a run directory's responses.jsonl + cases.jsonl."""
    store = ArtifactStore(run_dir)
    manifest = store.read_manifest()
    cases_by_id: dict[str, dict[str, Any]] = {}
    for case_rec in store.iter_cases():
        cases_by_id[case_rec["case_id"]] = case_rec

    from jev_agent_eval.schemas.result import CaseResult

    results: list[CaseResult] = []
    # Deduplicate recovered rows: a resumed run appends a fresh row for a
    # re-attempted case without removing the earlier failed row, so
    # responses.jsonl can hold (case_id, repeat) twice — error row first,
    # success row second. Scoring both would double-count the case and
    # permanently inflate transport_error_rate with stale errors. The LAST
    # row per (case_id, repeat) is the case's final outcome.
    latest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in store.iter_responses():
        key = (
            str(rec.get("case_id", "")),
            int(rec.get("repeat_index", str(rec.get("case_id", "")).partition("#")[2] or 0) or 0),
        )
        latest_by_key[key] = rec
    for rec in latest_by_key.values():
        # Response records key case_id as "case_id#repeat" (the resume
        # done-check format); the cases.jsonl join uses the bare id.
        case_id, _, repeat_suffix = str(rec.get("case_id", "")).partition("#")
        case_rec = cases_by_id.get(case_id, {})
        response = rec.get("response")
        predictions: dict[str, Any] = {}
        if response:
            parsed = JEVResponse.model_validate(response)
            predictions = {
                qid: {
                    "type": a.type,
                    "noul": a.noul,
                    "choice": a.choice,
                    "score": a.score,
                    "probabilities": a.probabilities,
                    "confidence": a.confidence,
                }
                for qid, a in parsed.answers.items()
            }
        questions = case_rec.get("questions") or {}
        if response:
            # A response missing expected answers is a schema error, not a
            # success: reclassifying it on replay would flip gate decisions.
            missing = [q for q in questions if q not in predictions]
            status = CaseStatus.SUCCESS if not missing else CaseStatus.SCHEMA_ERROR
        else:
            status = (
                CaseStatus.SCHEMA_ERROR if rec.get("parse_error") else CaseStatus.TRANSPORT_ERROR
            )
        cr = CaseResult(
            case_id=case_id,
            dataset=case_rec.get("dataset", ""),
            dataset_version=case_rec.get("dataset_version", ""),
            capability=case_rec.get("capability", ""),
            repeat_index=int(rec.get("repeat_index", repeat_suffix or 0) or 0),
            status=status,
            request_hash=rec.get("request_hash"),
            question_ids=list(questions.keys()),
            predictions=predictions,
            error_detail=(
                f"missing_answer: {missing}"
                if response and status == CaseStatus.SCHEMA_ERROR
                else (
                    rec.get("parse_error")
                    or rec.get("network_error_class")
                    # A null response with no recorded cause is still an
                    # error case; give triage something to act on rather
                    # than a silent null. (A response-bearing SUCCESS case
                    # that simply scored nothing must NOT hit this
                    # fallback — its answers exist, the scorer categorized
                    # them already.)
                    or (
                        f"no response recorded (http={rec.get('http_status')})"
                        if not response
                        else None
                    )
                )
            ),
            transport=TransportRecord(
                request_start="",
                latency_ms=rec.get("latency_ms"),
                http_status=rec.get("http_status"),
                network_error_class=rec.get("network_error_class"),
                retry_count=rec.get("retry_count", 0),
                from_cache=rec.get("from_cache", False),
                # Token usage must survive replay: metrics.json after a
                # replay must match the original run.
                input_tokens=((response or {}).get("usage") or {}).get("input_tokens"),
                output_tokens=((response or {}).get("usage") or {}).get("output_tokens"),
            ),
            # The persisted raw body (already redacted at write time; stored
            # as JSON text): restore it so the security report's
            # raw_responses_saved reflects what responses.jsonl actually
            # holds, not 0.
            raw_response=_parse_raw_response(rec.get("raw_response")),
            # Canonical top-level fields first, metadata as fallback —
            # matches the live path (see the note in offline.py).
            difficulty=case_rec.get("difficulty")
            or (case_rec.get("metadata") or {}).get("difficulty"),
            adversarial="adversarial" in (case_rec.get("tags") or []),
            source=case_rec.get("source")
            or (case_rec.get("metadata") or {}).get("source"),
            tags=case_rec.get("tags") or [],
            state_chars=len(case_rec.get("state") or ""),
        )
        # Re-attach gold snapshot from the stored case (still outside the JEV input).
        from jev_agent_eval.schemas.case import GoldAnswer

        # case_state drives the `case` command's STATE display; without it
        # every reloaded case printed an empty state (set only on the live
        # path in offline.py).
        object.__setattr__(cr, "case_state", case_rec.get("state") or "")
        gold_snapshot = {
            qid: GoldAnswer.model_validate(g) for qid, g in (case_rec.get("gold") or {}).items()
        }
        object.__setattr__(cr, "gold_snapshot", gold_snapshot)
        # Error taxonomy must match the live runner's (spec 33): a
        # reloaded error case carries the same categories as the original
        # run, or replay's triage contradicts the run's own errors.jsonl.
        if status != CaseStatus.SUCCESS:
            from jev_agent_eval.scoring import classify_failure

            category = classify_failure(cr, None, None)
            if category is not None and category not in cr.failure_categories:
                cr.failure_categories.append(category)
        results.append(cr)
    return manifest, results
