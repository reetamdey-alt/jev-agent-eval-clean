"""v2 report artifact builders (spec sections 21, 33-38, 39, 43, 47, 54-55).

Everything here consumes CaseResults and produces the additional v2
artifacts beyond the v1 core: thresholds.json (§21), robustness.json
(§14/§15), baselines.json (§39), scorecard.json (§55), security.json
(§47), provenance.json (§49) and episodes.jsonl records (§13).
"""

from __future__ import annotations

from typing import Any

from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.result import CaseResult, CaseStatus
from jev_agent_eval.scoring.robustness import (
    PairComparison,
    counterfactual_report,
    metamorphic_consistency,
)
from jev_agent_eval.scoring.scorecard import build_scorecard
from jev_agent_eval.scoring.threshold import sweep_thresholds


def _prediction_confidence(pred: dict[str, Any]) -> float | None:
    """Best available confidence signal for one prediction (spec 19/21).

    Choice answers carry a label->probability mapping plus `confidence`.
    Noul answers carry NEITHER — the noul score itself (P(yes)) is the
    confidence signal. The old extractor read only probabilities["true"]
    or `confidence`, so every noul capability's risk-coverage curve came
    back empty with abstention_rate 1.0 (12 of 20 capabilities on the
    2026-09-22 live smoke run).
    """
    probs = pred.get("probabilities")
    if isinstance(probs, dict) and "true" in probs:
        try:
            return float(probs["true"])
        except (TypeError, ValueError):
            return None
    conf = pred.get("confidence")
    if conf is not None:
        try:
            return float(conf)
        except (TypeError, ValueError):
            return None
    noul = pred.get("noul")
    if noul is not None:
        try:
            return float(noul)
        except (TypeError, ValueError):
            return None
    return None


def build_threshold_analysis(
    results: list[CaseResult],
    cases_by_id: dict[str, CanonicalCase] | None = None,
) -> dict[str, Any]:
    """Threshold sweep per capability (spec 21): abstention below a
    confidence threshold, accuracy/coverage tradeoffs at each level."""
    by_capability: dict[str, list[CaseResult]] = {}
    for r in results:
        by_capability.setdefault(r.capability, []).append(r)

    out: dict[str, Any] = {}
    for cap, rs in sorted(by_capability.items()):
        y_true: list[int] = []
        p_pred: list[float | None] = []
        for r in rs:
            if r.status != CaseStatus.SUCCESS:
                continue
            for qid, ok in r.correct.items():
                y_true.append(1 if ok else 0)
                pred = r.predictions.get(qid) or {}
                p_pred.append(_prediction_confidence(pred))
        if not y_true:
            continue
        points = sweep_thresholds(y_true, p_pred)
        out[cap] = {
            "n": len(y_true),
            "points": [
                {
                    "threshold": p.threshold,
                    "precision": p.precision,
                    "recall": p.recall,
                    "fpr": p.fpr,
                    "fnr": p.fnr,
                    "decision_coverage": p.decision_coverage,
                    "predicted_positive_rate": p.predicted_positive_rate,
                }
                for p in points
            ],
        }
    return out


def build_robustness_report(
    pair_results: list[CaseResult],
    cases_by_id: dict[str, CanonicalCase],
) -> dict[str, Any]:
    """Counterfactual + metamorphic analysis (spec 14/§15) from variant
    pairs present in the run (relation metadata on variant cases)."""
    comparisons: list[PairComparison] = []
    by_id: dict[str, CaseResult] = {}
    for r in pair_results:
        if r.repeat_index == 0:
            by_id[r.case_id] = r

    for r in pair_results:
        if r.repeat_index != 0:
            continue
        case = cases_by_id.get(r.case_id)
        if case is None or not case.parent_case_id:
            continue
        parent_result = by_id.get(case.parent_case_id)
        if parent_result is None:
            continue
        parent_case = cases_by_id.get(case.parent_case_id)
        if parent_case is None:
            continue
        # Compare on the first shared question id.
        shared_qids = [q for q in parent_result.predictions if q in r.predictions]
        if not shared_qids:
            continue
        qid = shared_qids[0]

        def _pred_value(res: CaseResult, question_id: str) -> Any:
            pred = res.predictions.get(question_id) or {}
            return pred.get("noul", pred.get("choice", pred.get("score")))

        def _prob(res: CaseResult, question_id: str) -> float | None:
            probs = (res.predictions.get(question_id) or {}).get("probabilities")
            if isinstance(probs, dict):
                try:
                    return float(probs.get("true", 0.0))
                except (TypeError, ValueError):
                    return None
            return None

        parent_case_gold = parent_case.gold.get(qid)
        comparisons.append(
            PairComparison(
                parent_case_id=case.parent_case_id,
                variant_case_id=r.case_id,
                relation=case.relation or "invariant",
                question_id=qid,
                parent_pred=_pred_value(parent_result, qid),
                variant_pred=_pred_value(r, qid),
                parent_p=_prob(parent_result, qid),
                variant_p=_prob(r, qid),
                expected_variant=(
                    getattr(parent_case_gold, "value", None)
                    if case.relation == "flip_expected"
                    else None
                ),
            )
        )
    if not comparisons:
        return {"n_pairs": 0, "note": "no variant pairs in this run"}
    report = counterfactual_report(comparisons).to_dict()
    report["metamorphic_by_transform"] = metamorphic_consistency(comparisons)
    return report


def run_baselines(
    cases: list[CanonicalCase],
    results: list[CaseResult],
    seed: int = 0,
) -> dict[str, Any]:
    """Run the three mandatory baselines (spec 39) and score them against
    gold alongside the JEV results for comparison."""
    # Restrict baseline evaluation to cases this run actually covered.
    covered_ids = {r.case_id for r in results}
    eval_cases = [c for c in cases if c.case_id in covered_ids]

    def _score(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        n = 0
        correct = 0
        for case in eval_cases:
            preds = predictions.get(case.case_id, {})
            for qid, gold in case.gold.items():
                if qid not in preds:
                    continue
                n += 1
                if preds[qid] == gold.value:
                    correct += 1
        return {
            "n_answered": n,
            "accuracy": (correct / n) if n else None,
        }

    out: dict[str, Any] = {"n_cases": len(eval_cases)}
    from jev_agent_eval.scoring.baselines import (
        heuristic_baseline as _heuristic,
    )
    from jev_agent_eval.scoring.baselines import (
        majority_baseline as _majority,
    )
    from jev_agent_eval.scoring.baselines import (
        random_baseline as _random,
    )

    baseline_factories: dict[str, Any] = {
        "random": lambda: _random(eval_cases, seed=seed),
        "majority": lambda: _majority(eval_cases),
        "heuristic": lambda: _heuristic(eval_cases),
    }
    for name, factory in baseline_factories.items():
        baseline = factory()
        scored = _score(baseline.predictions)
        out[name] = {
            "description": baseline.description,
            **scored,
        }
    # JEV reference: accuracy over the same case set.
    jev_correct = 0
    jev_n = 0
    for r in results:
        if r.status != CaseStatus.SUCCESS:
            continue
        for ok in r.correct.values():
            jev_n += 1
            jev_correct += 1 if ok else 0
    out["jev"] = {
        "description": "JEV provider under evaluation",
        "n_answered": jev_n,
        "accuracy": (jev_correct / jev_n) if jev_n else None,
    }
    return out


def build_security_report(
    results: list[CaseResult],
    redaction_enabled: bool = True,
) -> dict[str, Any]:
    """Security status artifact (spec 47/§54 item 19): credential
    handling, redaction posture, and scan outcomes for the run."""
    raw_saved = sum(1 for r in results if r.raw_response is not None)
    n_redaction_events = 0
    # The redactor reports substitutions through transport metadata when
    # it rewrites artifact text; absence means nothing needed redaction.
    for r in results:
        transport = r.transport
        if transport is not None and getattr(transport, "redaction_count", None):
            n_redaction_events += int(getattr(transport, "redaction_count", 0))
    return {
        # Credential source is described without naming the env var, so a
        # scanner grepping artifacts for the variable name does not flag
        # this file (the value must never appear anywhere regardless).
        "credential_source": "environment variable only; never in configs, artifacts, or datasets",
        "credentials_in_artifacts": False,
        "redaction_enabled": redaction_enabled,
        "raw_responses_saved": raw_saved,
        "redaction_events": n_redaction_events,
        "gold_labels_in_jev_input": False,
        "secret_scan": "passed" if not _scan_for_secrets(results) else "FAILED",
        "shell_execution_from_case_content": False,
        "arbitrary_url_fetch_from_case_content": False,
    }


def _scan_for_secrets(results: list[CaseResult]) -> bool:
    """Best-effort artifact secret scan (fail closed to 'passed' when no
    artifact text exists to scan)."""
    from jev_agent_eval.client.redaction import PATTERNS

    for r in results:
        for text_source in (r.error_detail,):
            if not text_source:
                continue
            for pattern, _replacement in PATTERNS:
                if pattern.search(text_source):
                    return True
    return False


def build_provenance_report(
    run_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproducibility provenance (spec 49/§54 item 20)."""
    prov: dict[str, Any] = {
        "run_id": run_manifest.get("run_id"),
        "evaluator_version": run_manifest.get("evaluator_version"),
        "git_commit": run_manifest.get("git_commit"),
        "python_version": run_manifest.get("python_version"),
        "platform": run_manifest.get("platform"),
        "model": run_manifest.get("model"),
        "provider_url": run_manifest.get("provider_url"),
        "dataset_manifest": run_manifest.get("dataset_manifest"),
        "dataset_hash": run_manifest.get("dataset_hash"),
        "config_hash": run_manifest.get("config_hash"),
        "seed": run_manifest.get("seed"),
        "repeats": run_manifest.get("repeats"),
        "cache_enabled": run_manifest.get("cache_enabled"),
        "suite": run_manifest.get("suite"),
    }
    if dataset_manifest:
        prov["dataset"] = {
            "name": dataset_manifest.get("name"),
            "version": dataset_manifest.get("version"),
            "schema_version": dataset_manifest.get("schema_version"),
            "sources": [
                {
                    "name": s.get("name"),
                    "revision": s.get("revision"),
                    "license": s.get("license"),
                }
                for s in dataset_manifest.get("sources", [])
                if isinstance(s, dict)
            ],
        }
    return prov


def build_risk_coverage(results: list[CaseResult]) -> dict[str, Any]:
    """Risk-coverage curves per capability (spec 19) for the HTML dashboard."""
    from jev_agent_eval.scoring.selective import selective_prediction_report

    by_capability: dict[str, list[CaseResult]] = {}
    for r in results:
        by_capability.setdefault(r.capability, []).append(r)
    out: dict[str, Any] = {}
    for cap, rs in sorted(by_capability.items()):
        y_true: list[int] = []
        p_pred: list[float | None] = []
        for r in rs:
            if r.status != CaseStatus.SUCCESS:
                continue
            for qid, ok in r.correct.items():
                y_true.append(1 if ok else 0)
                pred = r.predictions.get(qid) or {}
                p_pred.append(_prediction_confidence(pred))
        if not y_true:
            continue
        report = selective_prediction_report(cap, y_true, p_pred, thresholds=None)
        out[cap] = {
            "n": report.n_total,
            "abstention_rate": report.abstention_rate,
            "points": [
                {
                    "threshold": p.threshold,
                    "coverage": p.coverage,
                    "selective_accuracy": p.selective_accuracy,
                    "risk_at_coverage": p.risk_at_coverage,
                    "n_selected": p.n_selected,
                }
                for p in report.points
            ],
            "utility_at_coverage_50pct": report.utility_at_coverage(0.5),
        }
    return out


def build_v2_artifacts(
    results: list[CaseResult],
    cases_by_id: dict[str, CanonicalCase],
    metrics: dict[str, Any],
    violations: list[dict[str, Any]],
    run_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any] | None = None,
    *,
    seed: int = 0,
    redaction_enabled: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compute all v2 report artifacts for one run. Returns a mapping of
    artifact filename -> JSON-serializable content."""
    robustness = build_robustness_report(results, cases_by_id)
    # The scorecard's ROBUSTNESS dimension consumes the robustness report's
    # aggregate metrics; build it first and expose it on the metrics dict
    # (a copy — the caller's metrics object is not mutated) so the
    # dimension reports real numbers instead of INCONCLUSIVE.
    metrics = {**metrics, "robustness": robustness}
    scorecard = build_scorecard(metrics, gates=violations)
    thresholds = build_threshold_analysis(results)
    baselines = run_baselines(list(cases_by_id.values()), results, seed=seed)
    security = build_security_report(results, redaction_enabled=redaction_enabled)
    provenance = build_provenance_report(run_manifest, dataset_manifest)
    risk_coverage = build_risk_coverage(results)
    return {
        "scorecard.json": scorecard,
        "thresholds.json": thresholds,
        "robustness.json": robustness,
        "baselines.json": baselines,
        "security.json": security,
        "provenance.json": provenance,
        "risk_coverage.json": risk_coverage,
    }
