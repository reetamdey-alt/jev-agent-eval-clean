"""Run comparison and regression classification (spec sections 54, 55)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_agent_eval.scoring.uncertainty import paired_cluster_bootstrap_delta_ci

# Engineering thresholds for classifying deltas: below these, "stable".
DEFAULT_STABLE = {
    "accuracy": 0.01,
    "ece": 0.005,
    "brier": 0.005,
    "p95_latency_ms": 100.0,
    "input_tokens": 0.05,  # relative
    "error_rate": 0.005,
}


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing metrics.json in {run_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_scored_flags(run_dir: Path) -> dict[str, dict[tuple[str, str], list[tuple[int, bool]]]]:
    """Per-capability correctness flags keyed by (case_id, question_id) unit.

    Each unit maps to its per-repeat observations [(repeat_index, correct)].
    Keyed — not positional: a row inserted or dropped in one run must not
    shift every subsequent pair. Only scored rows (with a question id)
    contribute; error-case rows are excluded from pairing.
    """
    path = run_dir / "scored.jsonl"
    out: dict[str, dict[tuple[str, str], list[tuple[int, bool]]]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cap = rec.get("capability", "")
            qid = rec.get("question_id")
            if not qid:
                continue  # error-case row: no pairing
            unit = (str(rec.get("case_id")), str(qid))
            out.setdefault(cap, {}).setdefault(unit, []).append(
                (int(rec.get("repeat_index", 0) or 0), bool(rec.get("correct")))
            )
    return out


def compare_runs(run_a: Path | str, run_b: Path | str) -> dict[str, Any]:
    """Compare two runs; classify per-capability deltas (spec 54)."""
    a_dir, b_dir = Path(run_a), Path(run_b)
    ma, mb = _load_metrics(a_dir), _load_metrics(b_dir)

    caps_a, caps_b = ma.get("capabilities", {}), mb.get("capabilities", {})
    all_caps = sorted(set(caps_a) | set(caps_b))

    capabilities: dict[str, Any] = {}
    for cap in all_caps:
        ca, cb = caps_a.get(cap, {}), caps_b.get(cap, {})
        acc_a, acc_b = ca.get("accuracy"), cb.get("accuracy")
        delta = (acc_b - acc_a) if (acc_a is not None and acc_b is not None) else None
        ece_a = (ca.get("calibration") or {}).get("ece")
        ece_b = (cb.get("calibration") or {}).get("ece")
        brier_a = (ca.get("calibration") or {}).get("brier")
        brier_b = (cb.get("calibration") or {}).get("brier")
        lat_a = (ca.get("latency") or {}).get("p95_ms")
        lat_b = (cb.get("latency") or {}).get("p95_ms")

        classification = _classify(delta)
        # Per-capability safety and reliability status (spec 37): a
        # capability whose reliability (transport/schema errors) or safety
        # posture (false allows on gating questions) degraded must not hide
        # behind a stable accuracy classification.
        def _reliability_status(m: dict[str, Any]) -> str:
            n = m.get("n_cases") or 0
            statuses = m.get("statuses") or {}
            errors = sum(v for k, v in statuses.items() if k not in ("success", "cached"))
            if not n:
                return "unknown"
            if errors / n > 0.02:
                return "degraded"
            return "ok"

        capabilities[cap] = {
            "n_a": ca.get("n_cases"), "n_b": cb.get("n_cases"),
            "accuracy_a": acc_a, "accuracy_b": acc_b, "accuracy_delta": delta,
            "ece_a": ece_a, "ece_b": ece_b,
            "ece_delta": (ece_b - ece_a) if (ece_a is not None and ece_b is not None) else None,
            "brier_a": brier_a, "brier_b": brier_b,
            "brier_delta": (brier_b - brier_a) if (brier_a is not None and brier_b is not None) else None,
            "p95_latency_a": lat_a, "p95_latency_b": lat_b,
            "p95_latency_delta": (lat_b - lat_a) if (lat_a is not None and lat_b is not None) else None,
            "classification": classification,
            "reliability_status_a": _reliability_status(ca),
            "reliability_status_b": _reliability_status(cb),
            "minimum_meaningful_difference": DEFAULT_STABLE["accuracy"],
        }

    flags_a = _load_scored_flags(a_dir)
    flags_b = _load_scored_flags(b_dir)
    paired: dict[str, dict[str, Any]] = {}
    for cap in all_caps:
        fa, fb = flags_a.get(cap), flags_b.get(cap)
        if not fa or not fb:
            continue
        # Pair on the intersection of (case, question) units in a
        # deterministic order; unpaired rows are reported, not silently
        # dropped by a length mismatch. Repeats within a unit are carried
        # together (cluster bootstrap): treating them as independent rows
        # would shrink the CI ~repeats× (pseudo-replication). Only repeats
        # present in BOTH runs pair within a unit; a repeat missing from
        # one run is not comparable.
        common = sorted(set(fa) & set(fb))
        if len(common) < 10:
            paired[cap] = {
                "delta": None, "ci_low": None, "ci_high": None,
                "significant": False, "n_paired": len(common),
                "note": "fewer than 10 paired rows",
            }
            continue
        clusters_a: list[list[bool]] = []
        clusters_b: list[list[bool]] = []
        for unit in common:
            reps_a = {r: ok for r, ok in fa[unit]}
            reps_b = {r: ok for r, ok in fb[unit]}
            shared = sorted(set(reps_a) & set(reps_b))
            if not shared:
                continue
            clusters_a.append([reps_a[r] for r in shared])
            clusters_b.append([reps_b[r] for r in shared])
        if not clusters_a:
            paired[cap] = {
                "delta": None, "ci_low": None, "ci_high": None,
                "significant": False, "n_paired": 0,
                "note": "no units with shared repeats",
            }
            continue
        result = paired_cluster_bootstrap_delta_ci(clusters_a, clusters_b, seed=20260920)
        if result:
            point, ci = result
            paired[cap] = {
                "delta": point,
                "ci_low": ci.low, "ci_high": ci.high,
                "ci_level": ci.level, "ci_samples": ci.samples,
                "ci_method": ci.method,
                "significant": not (ci.low <= 0.0 <= ci.high),
                "n_paired": len(clusters_a),
            }

    safety_a = _load_scorecard_safety(a_dir)
    safety_b = _load_scorecard_safety(b_dir)

    def _safety_status(safety: dict[str, Any] | None) -> str:
        if not safety:
            return "unknown"
        return str(safety.get("outcome", "unknown")).lower()

    return {
        "run_a": {"dir": str(a_dir), "model": _model(a_dir), "run_id": _run_id(a_dir)},
        "run_b": {"dir": str(b_dir), "model": _model(b_dir), "run_id": _run_id(b_dir)},
        "capabilities": capabilities,
        "paired_bootstrap": paired,
        "safety": {
            "status_a": _safety_status(safety_a),
            "status_b": _safety_status(safety_b),
            "false_allow_rate_a": (safety_a or {}).get("metrics", {}).get("false_allow_rate"),
            "false_allow_rate_b": (safety_b or {}).get("metrics", {}).get("false_allow_rate"),
        },
        "overall": {
            "n_cases_a": ma.get("n_cases"), "n_cases_b": mb.get("n_cases"),
            "transport_error_rate_a": ma.get("transport_error_rate"),
            "transport_error_rate_b": mb.get("transport_error_rate"),
            "error_rate_delta": (mb.get("transport_error_rate", 0) - ma.get("transport_error_rate", 0)),
        },
    }


def _load_scorecard_safety(run_dir: Path) -> dict[str, Any] | None:
    """Safety dimension from a run's scorecard.json, when present."""
    path = run_dir / "scorecard.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            card = json.load(f)
        return card.get("dimensions", {}).get("SAFETY")
    except (OSError, json.JSONDecodeError):
        return None


def _model(run_dir: Path) -> str | None:
    mf = run_dir / "manifest.json"
    if mf.exists():
        with open(mf, encoding="utf-8") as f:
            return json.load(f).get("model")
    return None


def _run_id(run_dir: Path) -> str:
    return run_dir.name


def _classify(delta: float | None) -> str:
    if delta is None:
        return "inconclusive"
    if abs(delta) < DEFAULT_STABLE["accuracy"]:
        return "stable"
    return "improved" if delta > 0 else "regressed"


def render_comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Run comparison",
        "",
        f"Run A: `{report['run_a']['run_id']}` ({report['run_a']['model']})",
        f"Run B: `{report['run_b']['run_id']}` ({report['run_b']['model']})",
        "",
        "| Capability | N (A→B) | Accuracy A | Accuracy B | Δ accuracy | ECE Δ | p95 Δ (ms) | Classification |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cap, c in report.get("capabilities", {}).items():
        lines.append(
            f"| {cap} | {c['n_a']}→{c['n_b']} | {_f(c['accuracy_a'])} | {_f(c['accuracy_b'])} "
            f"| {_f(c['accuracy_delta'])} | {_f(c['ece_delta'])} | {_f(c['p95_latency_delta'])} "
            f"| {c['classification']} |"
        )
    paired = report.get("paired_bootstrap") or {}
    if paired:
        lines += ["", "## Paired bootstrap (95% CI on Δ accuracy)", "",
                  "| Capability | Δ | CI low | CI high | Significant | N paired | Note |",
                  "|---|---:|---:|---:|---|---:|---|"]
        for cap, p in paired.items():
            delta = "n/a" if p["delta"] is None else f"{p['delta']:+.4f}"
            low = "n/a" if p["ci_low"] is None else f"{p['ci_low']:+.4f}"
            high = "n/a" if p["ci_high"] is None else f"{p['ci_high']:+.4f}"
            lines.append(
                f"| {cap} | {delta} | {low} | {high} "
                f"| {'yes' if p['significant'] else 'no'} | {p['n_paired']} | {p.get('note', '')} |"
            )
        # A restricted pairing (different repeats or case sets between runs)
        # must be visible, not silently absorbed into the CI.
        n_a = report.get("overall", {}).get("n_cases_a")
        n_b = report.get("overall", {}).get("n_cases_b")
        if n_a is not None and n_b is not None and n_a != n_b:
            lines += [
                "",
                f"Note: runs have different case counts ({n_a} vs {n_b}); "
                "the bootstrap pairs only the intersection of "
                "(case, question) units (repeats carried per unit).",
            ]
    return "\n".join(lines)


def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.4f}"
