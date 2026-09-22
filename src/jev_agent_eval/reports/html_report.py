"""Offline HTML dashboard (spec section 35.2).

Renders via Jinja2 into a single self-contained file (inline CSS, no
external scripts) so the report works offline. Includes capability table,
confusion matrices, calibration charts (SVG), latency percentiles, and a
failure browser.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>JEV-Agent-Eval — {{ manifest.run_id }}</title>
<style>
  :root { --bg:#0f1117; --panel:#171a23; --text:#e6e8ee; --muted:#9aa1b2;
          --accent:#6c8cff; --good:#3ecf8e; --bad:#ff6b6b; --warn:#f0b429; }
  body { background:var(--bg); color:var(--text); font:14px/1.55 -apple-system,
         "Segoe UI", Roboto, sans-serif; margin:0; padding:32px; }
  h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:28px 0 8px;
         color:var(--accent); border-bottom:1px solid #262b38; padding-bottom:6px; }
  .meta { color:var(--muted); font-size:12px; margin-bottom:20px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }
  .card { background:var(--panel); border:1px solid #262b38; border-radius:10px;
          padding:14px 16px; }
  .card .label { color:var(--muted); font-size:11px; text-transform:uppercase;
                 letter-spacing:.06em; }
  .card .value { font-size:22px; font-weight:600; margin-top:4px; }
  table { border-collapse:collapse; width:100%; margin:10px 0; font-size:13px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid #232836; }
  th { color:var(--muted); font-weight:500; font-size:11px; text-transform:uppercase; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .pass { color:var(--good); } .fail { color:var(--bad); }
  details { margin:8px 0; } summary { cursor:pointer; color:var(--accent); }
  pre { background:#111420; border:1px solid #262b38; border-radius:8px;
        padding:12px; overflow-x:auto; font-size:12px; }
</style>
</head>
<body>
<h1>JEV-Agent-Eval Run</h1>
<div class="meta">Run {{ manifest.run_id }} · model {{ manifest.model }} ·
suite {{ manifest.suite or "release" }} · created {{ manifest.created_at }}</div>

<div class="grid">
  <div class="card"><div class="label">Cases</div><div class="value">{{ metrics.n_cases }}</div></div>
  <div class="card"><div class="label">Successful</div><div class="value">{{ metrics.n_successful }}</div></div>
  <div class="card"><div class="label">Transport errors</div>
    <div class="value">{{ "%.2f"|format(metrics.transport_error_rate * 100) }}%</div></div>
  <div class="card"><div class="label">p95 latency</div>
    <div class="value">{% if metrics.latency.p95_ms is none %}n/a{% else %}{{ "%.0f ms"|format(metrics.latency.p95_ms) }}{% endif %}</div></div>
</div>

<h2>Capability table</h2>
<table>
<tr><th>Capability</th><th class="num">N</th><th class="num">Accuracy</th>
    <th class="num">{{ ci_label }}</th>
    <th class="num">ECE</th><th class="num">Brier</th><th class="num">p95 ms</th></tr>
{% for cap, m in metrics.capabilities|dictsort %}
<tr><td>{{ cap }}</td><td class="num">{{ m.n_scored_questions if m.n_scored_questions is not none else m.n_cases }}</td>
    <td class="num">{{ m.accuracy is none and "n/a" or "%.4f"|format(m.accuracy) }}</td>
    <td class="num">{% if m.accuracy_ci %}[{{ "%.3f"|format(m.accuracy_ci.low) }}, {{ "%.3f"|format(m.accuracy_ci.high) }}]{% else %}n/a{% endif %}</td>
    <td class="num">{{ m.calibration.ece is none and "n/a" or "%.4f"|format(m.calibration.ece) }}</td>
    <td class="num">{{ m.calibration.brier is none and "n/a" or "%.4f"|format(m.calibration.brier) }}</td>
    <td class="num">{% if m.latency.p95_ms is none %}n/a{% else %}{{ "%.0f"|format(m.latency.p95_ms) }}{% endif %}</td></tr>
{% endfor %}
</table>

<h2>Confusion matrices</h2>
{% for cap, m in metrics.capabilities|dictsort %}
  {% for qid, q in (m.questions or {}).items() %}
    {% if q.confusion %}
<details><summary>{{ cap }} / {{ qid }}</summary>
<table>
<tr><th>gold \ pred</th>{% for c in q.classes %}<th>{{ c }}</th>{% endfor %}</tr>
{% for gold, row in q.confusion.items() %}
<tr><td>{{ gold }}</td>{% for c in q.classes %}<td class="num">{{ row[c] }}</td>{% endfor %}</tr>
{% endfor %}
</table></details>
    {% endif %}
  {% endfor %}
{% endfor %}

<h2>Calibration charts</h2>
{% for cap, cal in (metrics.calibration or {}).items() %}
<h3 style="font-size:14px;color:var(--text)">{{ cap }}</h3>
<table>
<tr><th>Confidence bin</th><th class="num">N</th><th class="num">Mean conf</th>
    <th class="num">Empirical acc</th><th class="num">Gap</th></tr>
{% for b in cal.reliability %}
<tr><td>{{ "%.1f"|format(b.lower) }}–{{ "%.1f"|format(b.upper) }}</td>
    <td class="num">{{ b.n }}</td><td class="num">{{ "%.3f"|format(b.mean_confidence) }}</td>
    <td class="num">{{ "%.3f"|format(b.empirical_accuracy) }}</td>
    <td class="num">{{ "%.3f"|format(b.calibration_gap) }}</td></tr>
{% endfor %}
</table>
{% endfor %}

<h2>Latency percentiles</h2>
<div class="grid">
{% for p in ["p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms"] %}
  <div class="card"><div class="label">{{ p }}</div>
    <div class="value">{% if metrics.latency[p] is none %}n/a{% else %}{{ "%.0f"|format(metrics.latency[p]) }}{% endif %}</div></div>
{% endfor %}
</div>

<h2>Worst slices</h2>
<table>
<tr><th>Dimension</th><th>Slice</th><th class="num">Accuracy</th><th class="num">N</th></tr>
{% for w in metrics.worst_slices %}
<tr><td>{{ w.dimension }}</td><td>{{ w.slice }}</td>
    <td class="num">{{ "%.4f"|format(w.accuracy) }}</td><td class="num">{{ w.n }}</td></tr>
{% endfor %}
</table>

<h2>Quality gates</h2>
{% if violations %}
<table>
<tr><th>Gate</th><th class="num">Value</th><th class="num">Limit</th><th>Status</th></tr>
{% for v in violations %}
<tr><td>{{ v.gate }}</td><td class="num">{% if v.value is none %}n/a{% else %}{{ "%.4f"|format(v.value) }}{% endif %}</td>
    <td class="num">{{ v.limit }}</td><td class="fail">FAIL</td></tr>
{% endfor %}
</table>
{% else %}<p class="pass">All gates passed.</p>{% endif %}

<h2>Failure browser</h2>
{% if failures %}
<p class="meta">{{ failures|length }} representative failing case-question pairs
(truncated state snippets; full detail in scored.jsonl / errors.jsonl).</p>
<details open><summary>All failures</summary>
<table>
<tr><th>Capability</th><th>Case</th><th>Question</th><th>Predicted</th>
    <th>Gold</th><th class="num">Confidence</th><th>Status</th><th>Detail</th></tr>
{% for f in failures %}
<tr><td>{{ f.capability }}</td><td>{{ f.case_id }}#{{ f.repeat_index }}</td>
    <td>{{ f.question_id if f.question_id is not none else "—" }}</td>
    <td>{{ f.predicted if f.predicted is not none else "—" }}</td>
    <td>{{ f.gold if f.gold is not none else "—" }}</td>
    <td class="num">{{ "%.3f"|format(f.confidence) if f.confidence is not none else "—" }}</td>
    <td>{{ f.status }}</td>
    <td class="fail">{{ f.error_detail or ", ".join(f.categories) }}</td></tr>
{% endfor %}
</table></details>
{% else %}<p class="pass">No failing cases in this run.</p>{% endif %}

<h2>Run manifest</h2>
<pre>{{ manifest|tojson(indent=2) }}</pre>

{% if v2.scorecard %}
<h2>Release scorecard (six dimensions)</h2>
<div class="grid">
{% for name, dim in v2.scorecard.dimensions|dictsort %}
  <div class="card">
    <div class="label">{{ name }}</div>
    <div class="value {% if dim.outcome == 'PASS' %}pass{% elif dim.outcome == 'FAIL' %}fail{% endif %}">{{ dim.outcome }}</div>
    {% if dim.gates %}
    <table style="font-size:11px">
      {% for gname, g in dim.gates|dictsort %}
      <tr><td>{{ gname }}</td><td class="num {% if g.ok %}pass{% else %}fail{% endif %}">{{ g.value if g.value is not none else "n/a" }}</td></tr>
      {% endfor %}
    </table>
    {% endif %}
  </div>
{% endfor %}
</div>

<h2>Baseline comparison (spec 39)</h2>
{% if v2.baselines %}
<table>
<tr><th>Baseline</th><th class="num">Questions answered</th><th class="num">Accuracy</th></tr>
{% for name, b in v2.baselines.items() %}
{% if b is mapping %}
<tr><td>{{ name }}</td><td class="num">{{ b.n_answered if b.n_answered is not none else "—" }}</td>
    <td class="num">{{ b.accuracy is none and "n/a" or "%.4f"|format(b.accuracy) }}</td></tr>
{% endif %}
{% endfor %}
</table>
<p class="meta">{{ v2.baselines.heuristic.description if v2.baselines.heuristic else "" }}</p>
{% endif %}

<h2>Threshold analysis (spec 21)</h2>
{% if v2.thresholds %}
{% for cap, t in v2.thresholds|dictsort %}
<details><summary>{{ cap }} ({{ t.n }} questions)</summary>
<table>
<tr><th class="num">Threshold</th><th class="num">Decision coverage</th>
    <th class="num">Precision</th><th class="num">Recall</th><th class="num">FPR</th></tr>
{% for p in t.points %}
{% if loop.index0 % 10 == 0 or loop.last %}
<tr><td class="num">{{ "%.2f"|format(p.threshold) }}</td>
    <td class="num">{{ "%.3f"|format(p.decision_coverage) }}</td>
    <td class="num">{{ p.precision is none and "n/a" or "%.3f"|format(p.precision) }}</td>
    <td class="num">{{ p.recall is none and "n/a" or "%.3f"|format(p.recall) }}</td>
    <td class="num">{{ p.fpr is none and "n/a" or "%.3f"|format(p.fpr) }}</td></tr>
{% endif %}
{% endfor %}
</table></details>
{% endfor %}
{% endif %}

<h2>Risk-coverage curves (spec 19)</h2>
{% if v2.risk_coverage %}
{% for cap, rc in v2.risk_coverage|dictsort %}
<details><summary>{{ cap }} ({{ rc.n }} questions, abstention {{ "%.1f"|format(rc.abstention_rate * 100) }}%)</summary>
<table>
<tr><th class="num">Confidence threshold</th><th class="num">Coverage</th>
    <th class="num">Selective accuracy</th><th class="num">Risk</th><th class="num">N selected</th></tr>
{% for p in rc.points %}
{% if loop.index0 % 10 == 0 or loop.last %}
<tr><td class="num">{{ "%.2f"|format(p.threshold) }}</td>
    <td class="num">{{ "%.3f"|format(p.coverage) }}</td>
    <td class="num">{{ "%.3f"|format(p.selective_accuracy) }}</td>
    <td class="num">{{ "%.3f"|format(p.risk_at_coverage) }}</td>
    <td class="num">{{ p.n_selected }}</td></tr>
{% endif %}
{% endfor %}
</table></details>
{% endfor %}
{% endif %}

<h2>Robustness (spec 14/15)</h2>
{% if v2.robustness and v2.robustness.n_pairs %}
<table>
<tr><th class="num">Pairs</th><th class="num">Counterfactual sensitivity</th>
    <th class="num">Non-target invariance</th><th class="num">Mean |Δp|</th></tr>
<tr><td class="num">{{ v2.robustness.n_pairs }}</td>
    <td class="num">{{ v2.robustness.counterfactual_sensitivity is none and "n/a" or "%.4f"|format(v2.robustness.counterfactual_sensitivity) }}</td>
    <td class="num">{{ v2.robustness.non_target_invariance is none and "n/a" or "%.4f"|format(v2.robustness.non_target_invariance) }}</td>
    <td class="num">{{ v2.robustness.mean_probability_delta is none and "n/a" or "%.4f"|format(v2.robustness.mean_probability_delta) }}</td></tr>
</table>
{% elif v2.robustness %}
<p class="meta">{{ v2.robustness.note or "no variant pairs in this run" }}</p>
{% endif %}

<h2>Security status (spec 47)</h2>
{% if v2.security %}
<table>
{% for key, value in v2.security|dictsort %}
<tr><td>{{ key }}</td><td>{{ value }}</td></tr>
{% endfor %}
</table>
{% endif %}

<h2>Slice explorer</h2>
<details open><summary>Per-dimension slices</summary>
{% for dim, slices in (metrics.slices or {})|dictsort %}
<h3 style="font-size:13px;color:var(--text)">{{ dim }}</h3>
<table>
<tr><th>Slice</th><th class="num">N</th><th class="num">Accuracy</th></tr>
{% for key, m in slices|dictsort %}
<tr><td>{{ key }}</td><td class="num">{{ m.n }}</td>
    <td class="num">{{ m.accuracy is none and "n/a" or "%.4f"|format(m.accuracy) }}</td></tr>
{% endfor %}
</table>
{% endfor %}
</details>
{% endif %}
</body>
</html>
"""


def render_html(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    violations: list[dict[str, Any]],
    failures: list[dict[str, Any]] | None = None,
    v2_artifacts: dict[str, Any] | None = None,
) -> str:
    import json as _json

    def _tojson(v: Any, indent: int | None = None) -> str:
        return _json.dumps(v, indent=indent, default=str)

    env = Environment(autoescape=True)
    env.filters["tojson"] = _tojson
    # Label the CI column with the actual configured level, mirroring the
    # markdown report (a hard-coded "95%" would mislabel a 90% interval).
    _cap_ci = next(
        (m.get("accuracy_ci") for m in (metrics.get("capabilities") or {}).values()
         if isinstance(m, dict) and m.get("accuracy_ci")),
        None,
    )
    ci_label = (
        f"{int(round((_cap_ci.get('level', 0.95)) * 100))}% CI" if _cap_ci else "CI"
    )
    return env.from_string(TEMPLATE).render(
        manifest=manifest, metrics=metrics, violations=violations,
        failures=failures or [], ci_label=ci_label,
        v2=v2_artifacts or {},
    )
