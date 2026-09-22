"""Representative failure extraction for reports (spec sections 35.1, 35.2).

A failure browser needs the actual failing cases — case id, capability, the
question answered wrong, what was predicted, and what gold said — not just
the taxonomy counts. Gold values are included here (reports are trusted
evaluator-side artifacts; the JEV input leakage rules apply to the request,
not the report) but the state text is truncated to keep reports readable.
"""

from __future__ import annotations

from typing import Any

from jev_agent_eval.schemas.result import CaseResult

MAX_FAILURES = 25
STATE_SNIPPET_CHARS = 300


def representative_failures(results: list[CaseResult]) -> list[dict[str, Any]]:
    """Failing cases (correct=False or error status), capped, for the report.

    Ordered by capability then case id for determinism. Each entry carries
    the question-level detail for every wrong answer on the case.
    """
    entries: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda r: (r.capability, r.case_id, r.repeat_index)):
        wrong = {qid: ok for qid, ok in r.correct.items() if not ok}
        errored = r.status.value != "success"
        if not wrong and not errored:
            continue
        state = getattr(r, "case_state", "") or ""
        gold_snap: dict[str, Any] = getattr(r, "gold_snapshot", {}) or {}
        for qid in sorted(wrong):
            gold = gold_snap.get(qid)
            gold_val = getattr(gold, "value", None)
            pred = (r.predictions or {}).get(qid) or {}
            pred_val = (
                pred.get("choice")
                if pred.get("choice") is not None
                else pred.get("noul")
                if pred.get("noul") is not None
                else pred.get("score")
            )
            entries.append({
                "case_id": r.case_id,
                "repeat_index": r.repeat_index,
                "capability": r.capability,
                "question_id": qid,
                "predicted": pred_val,
                "gold": gold_val,
                "confidence": pred.get("confidence"),
                "status": r.status.value,
                "categories": [c.value for c in r.failure_categories],
                "error_detail": r.error_detail,
                "state_snippet": state[:STATE_SNIPPET_CHARS],
            })
        if errored and not r.correct:
            # Whole-case failure (transport/schema error): no question detail.
            entries.append({
                "case_id": r.case_id,
                "repeat_index": r.repeat_index,
                "capability": r.capability,
                "question_id": None,
                "predicted": None,
                "gold": None,
                "confidence": None,
                "status": r.status.value,
                "categories": [c.value for c in r.failure_categories],
                "error_detail": r.error_detail,
                "state_snippet": state[:STATE_SNIPPET_CHARS],
            })
        if len(entries) >= MAX_FAILURES:
            break
    return entries[:MAX_FAILURES]
