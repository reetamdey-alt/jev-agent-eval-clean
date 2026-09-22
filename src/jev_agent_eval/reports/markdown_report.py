"""Markdown summary report (spec section 35.1).

Must include: run identity, model/API target, dataset versions, N, transport
reliability, correctness by capability, calibration, latency, token usage,
regression deltas, failure taxonomy, worst slices, representative failures,
and reproducibility information.
"""

from __future__ import annotations

from typing import Any


def render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    violations: list[dict[str, Any]],
    failures: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    a = lines.append

    a("# JEV-Agent-Eval Run Report")
    a("")
    a("## 1. Run identity")
    a("")
    a(f"- Run ID: `{manifest.get('run_id', 'unknown')}`")
    a(f"- Created at: {manifest.get('created_at', 'unknown')}")
    a(f"- Evaluator version: {manifest.get('evaluator_version', 'unknown')}")
    a(f"- Git commit: {manifest.get('git_commit') or 'n/a'}")
    a(f"- Suite: {manifest.get('suite') or 'release'}")
    a("")
    a("## 2. Model and API target")
    a("")
    a(f"- Model: `{manifest.get('model', 'unknown')}`")
    a(f"- Provider URL: `{manifest.get('provider_url', 'unknown')}`")
    a(f"- Seed: {manifest.get('seed')}  |  Repeats: {manifest.get('repeats')}  |  Cache: {manifest.get('cache_enabled')}")
    a("")
    a("## 3. Dataset")
    a("")
    a(f"- Manifest: `{manifest.get('dataset_manifest', 'unknown')}`")
    a(f"- Dataset hash: `{manifest.get('dataset_hash') or 'n/a'}`")
    a(f"- Config hash: `{manifest.get('config_hash') or 'n/a'}`")
    a("")
    a("## 4. Sample sizes")
    a("")
    a(f"- Cases: **{metrics.get('n_cases', 0)}**")
    a(f"- Successful inferences: {metrics.get('n_successful', 0)}")
    a("")
    a("## 5. Transport reliability")
    a("")
    a(f"- Transport error rate: {metrics.get('transport_error_rate', 0):.4f}")
    a(f"- Schema error rate: {metrics.get('schema_error_rate', 0):.4f}")
    a("")
    a("## 6. Correctness by capability")
    a("")
    # The CI level is configurable; label the column with the actual level
    # instead of a hard-coded "95%" that would mislabel a 90% interval.
    _cap_ci = next(
        (m.get("accuracy_ci") for m in metrics.get("capabilities", {}).values()
         if m.get("accuracy_ci")),
        None,
    )
    _ci_label = (
        f"{int(round((_cap_ci.get('level', 0.95)) * 100))}% CI"
        if _cap_ci else "CI"
    )
    a(f"| Capability | N | Accuracy | {_ci_label} | ECE | Brier | p95 latency (ms) |")
    a("|---|---:|---:|---|---:|---:|---:|")
    for cap, m in sorted(metrics.get("capabilities", {}).items()):
        ci = m.get("accuracy_ci")
        ci_str = f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci else "n/a"
        ece = m.get("calibration", {}).get("ece")
        brier = m.get("calibration", {}).get("brier")
        p95 = m.get("latency", {}).get("p95_ms")
        a(
            f"| {cap} | {m.get('n_scored_questions', m.get('n_cases', 0))} | "
            f"{_fmt(m.get('accuracy'))} | {ci_str} | {_fmt(ece)} | {_fmt(brier)} | {_fmt(p95)} |"
        )
    a("")
    a("## 7. Calibration")
    a("")
    for cap, cal in (metrics.get("calibration") or {}).items():
        a(f"### {cap}")
        a("")
        a(f"- ECE: {_fmt(cal.get('ece'))}  |  Adaptive ECE: {_fmt(cal.get('adaptive_ece'))}")
        a(f"- Brier: {_fmt(cal.get('brier'))}  |  NLL: {_fmt(cal.get('nll'))}  |  N: {cal.get('n', 0)}")
        rel = cal.get("reliability") or []
        if rel:
            a("")
            a("| Confidence bin | N | Mean confidence | Empirical accuracy | Gap |")
            a("|---|---:|---:|---:|---:|")
            for b in rel:
                a(
                    f"| {b['lower']:.1f}-{b['upper']:.1f} | {b['n']} | "
                    f"{b['mean_confidence']:.3f} | {b['empirical_accuracy']:.3f} | {b['calibration_gap']:.3f} |"
                )
        a("")
    a("## 8. Latency")
    a("")
    lat = metrics.get("latency", {})
    a(f"- Live measurements: {lat.get('n_live', 0)} (cached latencies excluded)")
    a(f"- p50: {_fmt(lat.get('p50_ms'))} ms  |  p90: {_fmt(lat.get('p90_ms'))} ms  |  p95: {_fmt(lat.get('p95_ms'))} ms  |  p99: {_fmt(lat.get('p99_ms'))} ms")
    a("")
    a("## 9. Token usage")
    a("")
    usage = metrics.get("usage", {})
    a(f"- Input tokens total: {usage.get('input_tokens_total', 0)}")
    a(f"- Output tokens total: {usage.get('output_tokens_total', 0)}")
    cost = usage.get("cost_per_1000_decisions")
    if cost is not None:
        a(f"- Cost per 1000 decisions: ${cost:.4f}")
    else:
        a("- Cost per 1000 decisions: unavailable (pricing not configured)")
    a("")
    a("## 10. Quality gates")
    a("")
    if violations:
        a("**GATE FAILURES:**")
        a("")
        for v in violations:
            # gate_not_evaluated has no measured value — format guards
            # against None (previously crashed the whole report build).
            value = v.get("value")
            value_str = f"{value:.4f}" if value is not None else "n/a"
            a(f"- `{v['gate']}`: value {value_str} exceeds limit {v.get('limit')}")
    else:
        a("All gates passed.")
    a("")
    a("## 11. Failure taxonomy")
    a("")
    taxonomy = metrics.get("failures") or {}
    if not taxonomy or not any(taxonomy.values()):
        a("No recorded failure categories.")
    else:
        for cap, counts in sorted(taxonomy.items()):
            if not counts:
                continue
            a(f"### {cap}")
            a("")
            for cat, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                a(f"- {cat}: {count}")
    a("")
    a("## 12. Worst slices")
    a("")
    worst = metrics.get("worst_slices") or []
    if worst:
        a("| Dimension | Slice | Accuracy | N |")
        a("|---|---|---:|---:|")
        for w in worst:
            a(f"| {w['dimension']} | {w['slice']} | {_fmt(w['accuracy'])} | {w['n']} |")
    else:
        a("No slices with N >= 5.")
    a("")
    a("## 13. Representative failures")
    a("")
    if not failures:
        a("No failing cases in this run.")
    else:
        a("| Capability | Case | Question | Predicted | Gold | Confidence | Status |")
        a("|---|---|---|---|---|---:|---|")
        for f in failures:
            pred = f.get("predicted")
            gold = f.get("gold")
            conf = f.get("confidence")
            a(
                f"| {f['capability']} | `{f['case_id']}#{f.get('repeat_index', 0)}` | "
                f"{f.get('question_id') or '—'} | "
                f"{pred if pred is not None else '—'} | "
                f"{gold if gold is not None else '—'} | "
                f"{conf if conf is not None else '—'} | {f.get('status', '')} |"
            )
    a("")
    a("## 14. Reproducibility")
    a("")
    a(f"- Python: {manifest.get('python_version', 'unknown')}")
    a(f"- Platform: {manifest.get('platform', 'unknown')}")
    a("- Replay: `jev-eval replay <run_dir>` (no network required)")
    a("")
    a("## 15. Traceability")
    a("")
    a("Every case is traceable via `cases.jsonl` / `responses.jsonl` / `scored.jsonl`:")
    a("`run -> dataset version -> case -> question -> raw response -> probability/confidence -> gold -> scorer`.")
    a("")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
