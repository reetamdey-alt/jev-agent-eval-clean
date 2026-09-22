"""Terminal report with Rich (spec sections 53, 35)."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table


def print_summary(manifest: dict[str, Any], metrics: dict[str, Any], violations: list[dict[str, Any]]) -> None:
    console = Console()
    console.print(f"[bold]JEV-Agent-Eval run {manifest.get('run_id')}[/bold]")
    console.print(
        f"model={manifest.get('model')}  suite={manifest.get('suite') or 'release'}  "
        f"cases={metrics.get('n_cases')}  successful={metrics.get('n_successful')}"
    )
    console.print(
        f"transport_error_rate={metrics.get('transport_error_rate', 0):.4f}  "
        f"schema_error_rate={metrics.get('schema_error_rate', 0):.4f}"
    )
    # Spec 12: reports must identify live vs replayed measurements. Error
    # rates are computed over live calls only; say so when cache served
    # part of the run, or a reader assumes the rates cover all cases.
    n_cache = metrics.get("n_cache_served") or 0
    if n_cache:
        console.print(
            f"[dim]cache served {n_cache}/{metrics.get('n_cases')} cases; "
            f"error rates and latency are over the "
            f"{metrics.get('n_live_calls')} live calls[/dim]"
        )
    table = Table(title="Capabilities")
    table.add_column("capability")
    table.add_column("N", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("p95 ms", justify="right")
    for cap, m in sorted((metrics.get("capabilities") or {}).items()):
        ece = (m.get("calibration") or {}).get("ece")
        p95 = (m.get("latency") or {}).get("p95_ms")
        # N is the accuracy's actual denominator (scored questions), not
        # attempted cases: a table reading "N=25, accuracy=1.0" while 5 of
        # the 25 cases errored hides the errors behind the accuracy
        # (spec 35: never hide a problem behind a high accuracy number).
        n = m.get("n_scored_questions")
        if n is None:
            n = m.get("n_cases", 0)
        table.add_row(
            cap,
            str(n),
            _fmt(m.get("accuracy")),
            _fmt(ece),
            _fmt(p95),
        )
    console.print(table)
    if violations:
        console.print("[bold red]GATE FAILURES:[/bold red]")
        for v in violations:
            # recall_min with no positive gold carries value=None — the
            # n/a must print, not crash the summary.
            value = v.get("value")
            value_str = f"{value:.4f}" if value is not None else "n/a"
            console.print(f"  {v['gate']}: {value_str} > {v['limit']}")
    else:
        console.print("[green]All quality gates passed.[/green]")


def print_failure_triage(record: dict[str, Any]) -> None:
    """Failure triage UX (spec section 53).

    Renders EVERY question's answer for the case: a multi-question case
    (tool_risk asks is_high_risk AND risk_class) must show all answers,
    not just the first.
    """
    console = Console()
    console.print(f"[bold]CASE:[/bold] {record.get('case_id')}")
    console.print(f"[bold]CAPABILITY:[/bold] {record.get('capability')}")
    console.print(f"[bold]SOURCE:[/bold] {record.get('source') or 'unknown'}")
    console.print(f"[bold]STATE:[/bold]\n{record.get('state', '')[:600]}")
    console.print(f"[bold]GOLD:[/bold] {record.get('gold')}")
    for qid, pred in (record.get("predictions") or {}).items():
        probs = pred.get("probabilities") if isinstance(pred, dict) else None
        # Rich would parse [question_id] as a style tag and silently strip
        # it from the output; escape the brackets so the label survives.
        label = qid.replace("[", "\\[")
        console.print(f"[bold]JEV \\[{label}]:[/bold] {pred}")
        console.print(f"[bold]PROBABILITY \\[{label}]:[/bold] {probs}")
        console.print(
            f"[bold]CONFIDENCE \\[{label}]:[/bold] "
            f"{pred.get('confidence') if isinstance(pred, dict) else None}"
        )
    console.print(f"[bold]LATENCY:[/bold] {record.get('latency_ms')} ms")
    console.print(f"[bold]ERROR CLASS:[/bold] {record.get('error_class')}")
    console.print(f"[bold]TRACE:[/bold] request_hash = {record.get('request_hash')}")


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.4f}"
