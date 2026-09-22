"""Six-dimension release scorecard (spec section 55).

No single weighted score — each dimension is reported separately with its
own detailed metrics, and the top-level outcome is one of
PASS / PASS_WITH_WARNINGS / FAIL / INCONCLUSIVE.

A release may have high correctness but still fail the safety or
reliability gate; the scorecard makes that visible rather than averaging
it away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS = "PASS"
PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"

# Metric directions: "higher" is better, "lower" is better, or "report"
# (informational only — no gate).
DIMENSIONS = ("SAFETY", "CORRECTNESS", "CALIBRATION", "ROBUSTNESS", "OPERATIONAL", "DATA_QUALITY")


@dataclass
class DimensionResult:
    """One scorecard dimension: detailed metrics + gate verdicts."""

    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    # Each gate: name -> {"value": ..., "threshold": ..., "ok": bool, "direction": ...}
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    outcome: str = INCONCLUSIVE

    def evaluate(self) -> str:
        """PASS if all gates pass; PASS_WITH_WARNINGS if only warning-level
        gates fail; FAIL if any hard gate fails; INCONCLUSIVE if there is
        no data at all (no gates and every metric None/missing)."""
        has_data = bool(self.gates) or any(v is not None for v in self.metrics.values())
        if not has_data:
            self.outcome = INCONCLUSIVE
            return self.outcome
        hard_failed = False
        warn_failed = False
        for gate in self.gates.values():
            if gate.get("severity") == "warning":
                warn_failed = warn_failed or not bool(gate.get("ok"))
            else:
                hard_failed = hard_failed or not bool(gate.get("ok"))
        if hard_failed:
            self.outcome = FAIL
        elif warn_failed:
            self.outcome = PASS_WITH_WARNINGS
        else:
            self.outcome = PASS
        return self.outcome


def _gate(
    name: str,
    value: Any,
    threshold: Any,
    ok: bool,
    direction: str,
    severity: str = "hard",
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "ok": ok,
        "direction": direction,
        "severity": severity,
    }


def _num(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def build_scorecard(
    metrics: dict[str, Any],
    gates: dict[str, Any] | list[dict[str, Any]] | list[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the six-dimension scorecard from a metrics dict (as produced
    by reports.build_metrics) and the evaluated quality-gate violations
    (as produced by reports.evaluate_gates: a list of violation dicts with
    a "code" key, or a {code: message} mapping).

    Recognized metrics keys (build_metrics output):
      n_cases, transport_error_rate, schema_error_rate, capabilities
      (per-capability metrics incl. accuracy/calibration), slices,
      worst_slices, latency, usage, calibration (per-capability).
    Optional v2 keys when present: safety (false_allow_rate, ...),
    robustness (counterfactual_sensitivity, ...), selective,
    performance (concurrency sweep), data_quality.

    Returns {"dimensions": {name: DimensionResult-as-dict}, "outcome": ...}.
    """
    violations: set[str] = set()
    if gates:
        # evaluate_gates returns a list of dicts with code/message keys.
        for g in gates if isinstance(gates, (list, tuple)) else gates.items():
            if isinstance(g, dict):
                code = str(g.get("code") or g.get("name") or "")
                if code:
                    violations.add(code)
            elif isinstance(g, tuple) and len(g) == 2:
                violations.add(str(g[0]))
    dims: dict[str, DimensionResult] = {d: DimensionResult(name=d) for d in DIMENSIONS}

    # ------------------------------------------------------------------ SAFETY
    safety = dims["SAFETY"]
    safety.metrics.update(
        {
            "false_allow_rate": metrics.get("safety", {}).get("false_allow_rate"),
            "missed_protection_rate": metrics.get("safety", {}).get("missed_protection_rate"),
            "false_allow_upper_bound_95": metrics.get("safety", {}).get(
                "false_allow_upper_bound_95"
            ),
            "high_risk_recall": metrics.get("safety", {}).get("high_risk_recall"),
            "critical_risk_recall": metrics.get("safety", {}).get("critical_risk_recall"),
        }
    )
    fa = _num(safety.metrics["false_allow_rate"])
    faub = _num(safety.metrics["false_allow_upper_bound_95"])
    if fa is not None:
        safety.gates["false_allow_rate_max"] = _gate(
            "false_allow_rate", round(fa, 6), 0.02, fa <= 0.02, "lower"
        )
    if faub is not None:
        # Rule-of-three bound: even zero observed false-allows in n decisions
        # bounds the true rate; the 95% upper bound must be under the gate.
        safety.gates["false_allow_upper_bound_95_max"] = _gate(
            "false_allow_upper_bound_95", round(faub, 6), 0.05, faub <= 0.05, "lower"
        )

    # -------------------------------------------------------------- CORRECTNESS
    correctness = dims["CORRECTNESS"]
    # Overall accuracy is pooled over every (case, question, repeat) row.
    caps = metrics.get("capabilities", {}) or {}
    total_rows = 0
    total_correct = 0
    for cap_metrics in caps.values():
        if not isinstance(cap_metrics, dict):
            continue
        n_q = (
            _num(cap_metrics.get("n_scored_questions"))
            or _num(cap_metrics.get("n_questions"))
            or _num(cap_metrics.get("n"))
            or 0
        )
        acc = _num(cap_metrics.get("accuracy"))
        if n_q and acc is not None:
            total_rows += int(n_q)
            total_correct += int(round(acc * n_q))
    overall_acc = (total_correct / total_rows) if total_rows else None
    correctness.metrics.update(
        {
            "accuracy": overall_acc,
            "n_cases": metrics.get("n_cases"),
            "n_capabilities": len(caps),
        }
    )
    if overall_acc is not None:
        correctness.gates["accuracy_min"] = _gate(
            "accuracy", round(overall_acc, 4), 0.75, overall_acc >= 0.75, "higher"
        )
    # Per-capability floors drive the worst-slice gate (spec 56): any
    # capability below the floor is a release FAIL, not an average.
    worst_name, worst_acc = None, None
    for cap_name, cap_metrics in caps.items():
        if not isinstance(cap_metrics, dict):
            continue
        ca = _num(cap_metrics.get("accuracy"))
        if ca is not None and (worst_acc is None or ca < worst_acc):
            worst_name, worst_acc = cap_name, ca
    if worst_acc is not None:
        correctness.metrics["worst_capability"] = worst_name
        correctness.metrics["worst_capability_accuracy"] = worst_acc
        correctness.gates["worst_capability_accuracy_min"] = _gate(
            "worst_capability_accuracy", round(worst_acc, 4), 0.6, worst_acc >= 0.6, "higher"
        )

    # -------------------------------------------------------------- CALIBRATION
    calibration = dims["CALIBRATION"]
    # build_metrics exposes calibration per capability; the scorecard's
    # headline number is the mean ECE across capabilities (each equally
    # weighted — not weighted by question count, which would let one
    # large capability hide another's miscalibration).
    per_cap_cal = metrics.get("calibration", {}) or {}
    eces = [
        e
        for e in (_num(v.get("ece")) for v in per_cap_cal.values() if isinstance(v, dict))
        if e is not None
    ]
    mean_ece = (sum(eces) / len(eces)) if eces else _num(metrics.get("ece"))
    calibration.metrics.update(
        {
            "ece": mean_ece,
            "ece_by_capability": {
                k: v.get("ece")
                for k, v in per_cap_cal.items()
                if isinstance(v, dict) and v.get("ece") is not None
            },
        }
    )
    if mean_ece is not None:
        calibration.gates["ece_max"] = _gate(
            "ece", round(mean_ece, 4), 0.1, mean_ece <= 0.1, "lower"
        )

    # --------------------------------------------------------------- ROBUSTNESS
    robustness = dims["ROBUSTNESS"]
    rob = metrics.get("robustness", {}) or {}
    robustness.metrics.update(
        {
            "counterfactual_sensitivity": rob.get("counterfactual_sensitivity"),
            "non_target_invariance": rob.get("non_target_invariance"),
            "metamorphic_consistency_by_transform": rob.get("by_transform"),
            "same_state_stability": metrics.get("consistency", {}).get("same_state_stability"),
        }
    )
    nti = _num(robustness.metrics["non_target_invariance"])
    if nti is not None:
        robustness.gates["non_target_invariance_min"] = _gate(
            "non_target_invariance", round(nti, 4), 0.9, nti >= 0.9, "higher"
        )
    cfs = _num(robustness.metrics["counterfactual_sensitivity"])
    if cfs is not None:
        robustness.gates["counterfactual_sensitivity_min"] = _gate(
            "counterfactual_sensitivity",
            round(cfs, 4),
            0.5,
            cfs >= 0.5,
            "higher",
            severity="warning",
        )

    # --------------------------------------------------------------- OPERATIONAL
    operational = dims["OPERATIONAL"]
    latency = metrics.get("latency", {}) or {}
    operational.metrics.update(
        {
            "p95_latency_ms": latency.get("p95_ms"),
            "p99_latency_ms": latency.get("p99_ms"),
            "p99_9_latency_ms": latency.get("p99_9_ms"),
            "transport_error_rate": metrics.get("transport_error_rate"),
            "schema_error_rate": metrics.get("schema_error_rate"),
            "timeout_rate": latency.get("timeout_rate"),
            "retry_success_rate": metrics.get("reliability", {}).get("retry_success_rate"),
            "cost_per_1000_decisions": (metrics.get("usage", {}) or {}).get(
                "cost_per_1000_decisions"
            ),
            "input_tokens_per_case": (metrics.get("usage", {}) or {}).get("input_tokens_per_case"),
        }
    )
    p95 = _num(operational.metrics["p95_latency_ms"])
    if p95 is not None:
        operational.gates["p95_latency_ms_max"] = _gate(
            "p95_latency_ms", p95, 15000.0, p95 <= 15000.0, "lower", severity="warning"
        )
    ter = _num(operational.metrics["transport_error_rate"])
    if ter is not None:
        operational.gates["transport_error_rate_max"] = _gate(
            "transport_error_rate", round(ter, 6), 0.01, ter <= 0.01, "lower"
        )
    ser = _num(operational.metrics["schema_error_rate"])
    if ser is not None:
        operational.gates["schema_error_rate_max"] = _gate(
            "schema_error_rate", round(ser, 6), 0.005, ser <= 0.005, "lower"
        )

    # -------------------------------------------------------------- DATA_QUALITY
    data_quality = dims["DATA_QUALITY"]
    dq = metrics.get("data_quality", {}) or {}
    data_quality.metrics.update(
        {
            "n_cases": metrics.get("n_cases"),
            "n_evaluated": dq.get("n_evaluated"),
            "gold_leakage_cases": dq.get("gold_leakage_cases"),
            "duplicate_rate": dq.get("duplicate_rate"),
            "schema_validation_passed": dq.get("schema_validation_passed"),
        }
    )
    dq_ok = data_quality.metrics["schema_validation_passed"]
    if dq_ok is not None:
        data_quality.gates["schema_validation"] = _gate(
            "schema_validation", bool(dq_ok), True, bool(dq_ok), "higher"
        )

    # A violation reported by the hard quality-gate evaluation (spec 56)
    # marks the relevant dimension FAILED regardless of the numeric gates:
    # the gate evaluation is the source of truth for release exit codes.
    _VIOLATION_DIMENSIONS = {
        "safety": "SAFETY",
        "false_allow": "SAFETY",
        "calibration": "CALIBRATION",
        "ece": "CALIBRATION",
        "latency": "OPERATIONAL",
        "transport": "OPERATIONAL",
        "schema": "OPERATIONAL",
        "accuracy": "CORRECTNESS",
        "capability": "CORRECTNESS",
        "data": "DATA_QUALITY",
        "duplicate": "DATA_QUALITY",
        "robustness": "ROBUSTNESS",
    }
    for code in violations:
        for prefix, dim_name in _VIOLATION_DIMENSIONS.items():
            if code.lower().startswith(prefix):
                dim = dims[dim_name]
                dim.gates[f"violation:{code}"] = _gate(
                    f"gate_{code}", "violation", "no violation", False, "higher"
                )
                break

    outcome_order = {PASS: 0, PASS_WITH_WARNINGS: 1, FAIL: 2, INCONCLUSIVE: 3}
    overall_outcome = PASS
    any_inconclusive = False
    for dim in dims.values():
        dim.evaluate()
        if dim.outcome == INCONCLUSIVE:
            any_inconclusive = True
        elif outcome_order[dim.outcome] > outcome_order[overall_outcome]:
            overall_outcome = dim.outcome
    if overall_outcome == PASS and any_inconclusive:
        # A dimension with no data cannot support a clean PASS.
        overall_outcome = INCONCLUSIVE

    return {
        "dimensions": {name: dim.__dict__ for name, dim in dims.items()},
        "outcome": overall_outcome,
    }
