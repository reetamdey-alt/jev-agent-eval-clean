"""Consistency / repeatability analysis (spec section 28).

Runs the same immutable case multiple times (recommended release policy:
repeats = 5) and reports per-question stability:

    label_stability     = identical_labels / repeats
    mean_probability    = mean of p_positive across repeats
    std_probability     = standard deviation of p_positive
    max_probability_delta
    score / confidence std where applicable
"""

from __future__ import annotations

import math
from typing import Any

from jev_agent_eval.client.base import InferenceProvider, InferenceResult
from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.request import JEVRequest


def _p_positive(answer: Any) -> float | None:
    probs = getattr(answer, "probabilities", None)
    if isinstance(probs, dict) and "true" in probs:
        return float(probs["true"])
    noul = getattr(answer, "noul", None)
    if noul is not None:
        return float(noul)
    return None


def consistency_report(
    provider: InferenceProvider,
    case: CanonicalCase,
    questions: dict[str, Any],
    model: str,
    repeats: int = 5,
) -> dict[str, Any]:
    """Run one case `repeats` times and measure stability."""
    request = JEVRequest(state=case.state, model=model, questions=questions)
    results: list[InferenceResult] = [provider.infer(request) for _ in range(repeats)]

    report: dict[str, Any] = {
        "case_id": case.case_id,
        "capability": case.capability,
        "repeats": repeats,
        "questions": {},
    }
    qids = list(questions)
    labels: dict[str, list[Any]] = {q: [] for q in qids}
    probs: dict[str, list[float]] = {q: [] for q in qids}
    confs: dict[str, list[float]] = {q: [] for q in qids}
    scores: dict[str, list[float]] = {q: [] for q in qids}
    for r in results:
        if r.response is None:
            continue
        for qid, ans in r.response.answers.items():
            if qid not in labels:
                continue
            labels[qid].append(ans.choice if ans.type == "choice" else ans.noul)
            p = _p_positive(ans)
            if p is not None:
                probs[qid].append(p)
            if ans.confidence is not None:
                confs[qid].append(ans.confidence)
            if ans.score is not None:
                scores[qid].append(ans.score)

    for qid in qids:
        entry: dict[str, Any] = {}
        lab = labels[qid]
        if lab:
            first = lab[0]
            entry["label_stability"] = sum(1 for x in lab if x == first) / len(lab)
            entry["distinct_labels"] = sorted({str(x) for x in lab})
        pv = probs[qid]
        if pv:
            mean = sum(pv) / len(pv)
            entry["mean_probability"] = mean
            entry["std_probability"] = (
                math.sqrt(sum((p - mean) ** 2 for p in pv) / len(pv)) if len(pv) > 1 else 0.0
            )
            entry["max_probability_delta"] = max(pv) - min(pv)
        cv = confs[qid]
        if cv:
            cmean = sum(cv) / len(cv)
            entry["std_confidence"] = (
                math.sqrt(sum((c - cmean) ** 2 for c in cv) / len(cv)) if len(cv) > 1 else 0.0
            )
        sv = scores[qid]
        if sv:
            smean = sum(sv) / len(sv)
            entry["std_score"] = (
                math.sqrt(sum((s - smean) ** 2 for s in sv) / len(sv)) if len(sv) > 1 else 0.0
            )
        report["questions"][qid] = entry
    return report
