"""Typer CLI (spec sections 37, 38).

Stable exit codes:
    0 success, 1 quality-gate failure, 2 configuration error, 3 dataset error,
    4 authentication/provider error, 5 transport failure, 6 schema/scoring
    failure, 7 internal evaluator error.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console

from jev_agent_eval import __version__
from jev_agent_eval.client.base import InferenceProvider
from jev_agent_eval.config import ConfigError, config_hash, load_config
from jev_agent_eval.datasets.registry import load_manifest

app = typer.Typer(
    name="jev-eval",
    help="Evaluation framework for JEV as a coding-agent inference/control model.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
datasets_app = typer.Typer(help="Dataset management", no_args_is_help=True)
app.add_typer(datasets_app, name="datasets")

console = Console()
err = Console(stderr=True)

EXIT_SUCCESS = 0
EXIT_QUALITY_GATE = 1
EXIT_CONFIG = 2
EXIT_DATASET = 3
EXIT_AUTH = 4
EXIT_TRANSPORT = 5
EXIT_SCHEMA = 6
EXIT_INTERNAL = 7


def _exit(code: int, message: str | None = None) -> NoReturn:
    if message:
        err.print(message)
    raise typer.Exit(code)


# ---------------------------------------------------------------- datasets


@datasets_app.command("list")
def datasets_list(
    manifests_dir: Path = typer.Option(Path("datasets/manifests"), help="Manifests directory"),
) -> None:
    """List available dataset manifests."""
    if not manifests_dir.exists():
        _exit(EXIT_DATASET, f"manifests directory not found: {manifests_dir}")
    found = 0
    for p in sorted(manifests_dir.glob("*.yaml")):
        try:
            m = load_manifest(p)
        except Exception as e:
            console.print(f"[red]{p.name}: invalid ({e})[/red]")
            continue
        console.print(f"{p.name}: [bold]{m.name}[/bold] v{m.version} ({len(m.cases)} files)")
        found += 1
    _exit(EXIT_SUCCESS if found else EXIT_DATASET)


@datasets_app.command("validate")
def datasets_validate(manifest_path: Path = typer.Argument(...)) -> None:
    """Validate a dataset manifest: schema, integrity, case invariants, leakage."""
    from jev_agent_eval.datasets.local_jsonl import LeakageError
    from jev_agent_eval.datasets.registry import load_manifest_cases

    try:
        manifest = load_manifest(manifest_path)
        cases = load_manifest_cases(manifest, manifest_path.parent)
    except FileNotFoundError as e:
        _exit(EXIT_DATASET, str(e))
    except LeakageError as e:
        _exit(EXIT_DATASET, f"GOLD LEAKAGE: {e}")
    except Exception as e:
        _exit(EXIT_DATASET, f"invalid manifest: {e}")
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.capability] = counts.get(c.capability, 0) + 1
    console.print(f"[green]OK[/green] {manifest.name} v{manifest.version}: {len(cases)} cases")
    for cap in sorted(counts):
        console.print(f"  {cap}: {counts[cap]}")
    _exit(EXIT_SUCCESS)


def _manifest_case_paths(manifest_path: Path) -> list[Path]:
    """Case-file paths referenced by a dataset manifest (datasets root is
    the manifest's grandparent directory). Includes integrity-listed
    variant files (variants.jsonl): they are not under manifest.cases but
    the runner evaluates them, so replay's id map must contain them for
    robustness pair analysis (spec 14/15)."""
    from jev_agent_eval.datasets.registry import load_manifest

    loaded = load_manifest(manifest_path)
    root = manifest_path.resolve().parent.parent
    paths = [root / entry.path for entry in loaded.cases]
    for integrity_path in (getattr(loaded, "integrity", None) or {}):
        if "variant" in Path(integrity_path).name:
            candidate = root / integrity_path
            if candidate.exists() and candidate not in paths:
                paths.append(candidate)
    return paths


def _scored_records(results) -> list[dict[str, Any]]:
    """Per-question scored records (drives paired bootstrap in `compare`)."""
    out: list[dict[str, Any]] = []
    for r in results:
        if r.correct:
            for qid, ok in r.correct.items():
                out.append(
                    {
                        "case_id": r.case_id,
                        "capability": r.capability,
                        "question_id": qid,
                        "repeat_index": r.repeat_index,
                        "correct": bool(ok),
                        "request_hash": r.request_hash,
                    }
                )
        else:
            out.append(
                {
                    "case_id": r.case_id,
                    "capability": r.capability,
                    "question_id": None,
                    "repeat_index": r.repeat_index,
                    "correct": False,
                    "request_hash": r.request_hash,
                }
            )
    return out


def _error_records(results) -> list[dict[str, Any]]:
    """Failed-case records with their taxonomy category."""
    out: list[dict[str, Any]] = []
    for r in results:
        if r.correct:
            continue
        out.append(
            {
                "case_id": r.case_id,
                "capability": r.capability,
                "status": r.status.value,
                "error_detail": r.error_detail,
                "categories": [c.value for c in r.failure_categories],
                "request_hash": r.request_hash,
            }
        )
    return out


# ---------------------------------------------------------------- run


@app.command()
def run(
    dataset: Path | None = typer.Option(None, help="Dataset manifest path"),
    suite: str = typer.Option("smoke", help="smoke | pr | release"),
    model: str | None = typer.Option(None, help="Model override"),
    provider: str = typer.Option("systemone", help="Provider: systemone | mock"),
    config: Path | None = typer.Option(None, help="Config YAML"),
    concurrency: int | None = typer.Option(None),
    rate_limit: float | None = typer.Option(None, help="Requests per second"),
    repeats: int | None = typer.Option(None),
    no_cache: bool = typer.Option(False),
    live_only: bool = typer.Option(False, help="Bypass the response cache for this run"),
    capability: str | None = typer.Option(None),
    tag: str | None = typer.Option(None),
    max_cases: int | None = typer.Option(None),
    shuffle: bool = typer.Option(False, help="Randomize execution order (seeded, spec 72)"),
    seed: int | None = typer.Option(None),
    fail_fast: bool = typer.Option(False),
    output_dir: Path | None = typer.Option(None),
    resume_id: str | None = typer.Option(None, help="Internal: reuse an existing run id (resume)"),
) -> None:
    """Run an offline benchmark suite against JEV."""
    # Suite-aware default manifest (the error message's promised "default
    # manifest"): smoke/pr/nightly/release/redteam evaluate the public
    # release dataset; holdout evaluates the private one.
    if dataset is None:
        if suite == "holdout":
            _default = Path("datasets/manifests/holdout-v2.yaml")
        elif suite == "smoke":
            _default = Path("datasets/manifests/release-v2.yaml")
            if not _default.exists():
                _default = Path("datasets/manifests/core-v1.yaml")
        else:
            _default = Path("datasets/manifests/release-v2.yaml")
        if not _default.exists():
            _exit(
                EXIT_CONFIG,
                f"--dataset is required (default manifest not found at {_default})",
            )
        dataset = _default
    try:
        cfg = load_config(config)
    except ConfigError as e:
        _exit(EXIT_CONFIG, str(e))
    # CLI overrides land via assignment, which re-validates field
    # constraints (validate_assignment): an invalid override (repeats=0,
    # a negative threshold) must exit 2 with a clean message, not raise
    # a raw ValidationError traceback.
    try:
        if model:
            cfg.provider.model = model
        if concurrency:
            cfg.provider.concurrency = concurrency
        if rate_limit is not None:
            cfg.provider.requests_per_second = rate_limit
        if repeats is not None:
            # `is not None` (not truthiness): repeats=0 would be silently
            # dropped by `if repeats:` and run with the config's default
            # instead of failing RunConfig's ge=1 validation. The override
            # must land so the validation can reject it.
            cfg.run.repeats = repeats
        # `is not None` (not truthiness): seed=0 is a legitimate value
        # and must not be silently dropped by the override.
        if seed is not None:
            cfg.run.seed = seed
        if fail_fast:
            cfg.run.fail_fast = True
        if no_cache or live_only:
            cfg.run.cache = False
    except ValidationError as e:
        _exit(EXIT_CONFIG, f"invalid CLI override: {e}")
    # A typo'd suite is a config error (exit 2), not an internal error:
    # validate before the runner raises it mid-run.
    from jev_agent_eval.runners.offline import SUITE_LIMITS

    if suite not in SUITE_LIMITS:
        _exit(EXIT_CONFIG, f"unknown suite {suite!r}; expected one of {sorted(SUITE_LIMITS)}")

    from jev_agent_eval.datasets.registry import load_manifest as _lm
    from jev_agent_eval.runners.offline import OfflineRunner
    from jev_agent_eval.storage.artifacts import ArtifactStore
    from jev_agent_eval.utils.hashing import sha256_files

    assert dataset is not None  # narrowed by the default-manifest logic above
    try:
        loaded = _lm(dataset)  # validate the manifest before any work starts
        # The dataset hash covers the manifest AND the case files it
        # references: a case file changing under an unchanged manifest must
        # change the hash (and invalidate run comparisons).
        datasets_root = dataset.resolve().parent.parent
        case_paths = [datasets_root / entry.path for entry in loaded.cases]
        dataset_hash = sha256_files([dataset, *case_paths])
    except FileNotFoundError as e:
        _exit(EXIT_DATASET, str(e))
    except Exception as e:
        _exit(EXIT_DATASET, f"invalid manifest: {e}")

    if provider == "mock":
        from jev_agent_eval.client.systemone import MockJEVProvider

        prov: InferenceProvider = MockJEVProvider(model=cfg.provider.model)
        provider_url = "mock://local"
    elif provider != "systemone":
        # An unknown provider must be a config error, not a silent fall
        # through to SystemOne (which then fails as an auth error).
        _exit(EXIT_CONFIG, f"unknown provider {provider!r}: expected 'mock' or 'systemone'")
    else:
        from jev_agent_eval.client.systemone import SystemOneProvider

        try:
            prov = SystemOneProvider(
                base_url=cfg.provider.base_url,
                model=cfg.provider.model,
                timeout_s=cfg.provider.timeout_s,
                connect_timeout_s=cfg.provider.connect_timeout_s,
                read_timeout_s=cfg.provider.read_timeout_s,
                max_retries=cfg.provider.max_retries,
                backoff_base_s=cfg.provider.backoff_base_s,
                backoff_max_s=cfg.provider.backoff_max_s,
                requests_per_second=cfg.provider.requests_per_second,
            )
        except RuntimeError as e:
            _exit(EXIT_AUTH, str(e))
        provider_url = cfg.provider.base_url

    runner = OfflineRunner(
        prov,
        manifest_path=dataset,
        suite=suite,
        concurrency=cfg.provider.concurrency,
        repeats=cfg.run.repeats,
        seed=cfg.run.seed,
        use_cache=cfg.run.cache,
        save_raw=cfg.run.save_raw,
        redact_secrets=cfg.run.redact_secrets,
        output_dir=output_dir or cfg.reports.output_dir,
        run_id=resume_id,
        capability=capability,
        tag=tag,
        max_cases=max_cases,
        fail_fast=cfg.run.fail_fast,
        shuffle=shuffle,
        dataset_hash=dataset_hash,
        provider_url=provider_url,
        model=cfg.provider.model,
    )
    try:
        run_id, results = runner.run()
    except RuntimeError as e:
        _exit(EXIT_INTERNAL, str(e))
    except (ValueError, FileNotFoundError) as e:
        # Case-file problems (validation failures, missing files, duplicate
        # ids) surface here because the runner loads cases lazily; they are
        # dataset errors (exit 3), not evaluator-internal ones.
        _exit(EXIT_DATASET, f"invalid dataset: {e}")
    except Exception as e:
        _exit(EXIT_INTERNAL, f"run failed: {e}")

    run_dir = Path(output_dir or cfg.reports.output_dir) / "runs" / run_id
    store = ArtifactStore(run_dir)
    run_manifest = store.read_manifest()
    run_manifest["config_hash"] = config_hash(cfg)
    # Snapshot the resolved config so `replay` can restore the EXACT gates
    # and thresholds the run was evaluated under — otherwise a run that
    # exited 1 under a strict config would replay to exit 0 under defaults.
    run_manifest["config"] = cfg.model_dump(exclude_none=True)

    from jev_agent_eval.reports import build_metrics, evaluate_gates, print_summary

    metrics = build_metrics(
        results,
        bootstrap_samples=cfg.scoring.bootstrap_samples,
        confidence_level=cfg.scoring.confidence_level,
        ece_bins=cfg.scoring.ece_bins,
        seed=cfg.run.seed,
        input_usd_per_million=cfg.pricing.input_usd_per_million,
        output_usd_per_million=cfg.pricing.output_usd_per_million,
    )
    passed, violations = evaluate_gates(
        metrics,
        cfg.quality_gates,
        transport_error_rate_max=cfg.thresholds.transport_error_rate_max,
        schema_error_rate_max=cfg.thresholds.schema_error_rate_max,
        p95_latency_ms_max=cfg.thresholds.p95_latency_ms_max,
        # An explicit --capability/--tag filter declares a subset run: the
        # excluded capabilities are intentionally absent, so their gates
        # must not fire gate_not_evaluated (only the filtered-in capability
        # is gated). Without this, any filtered run would always exit 1.
        # Any narrowing flag (--capability/--tag/--max-cases) declares a
        # subset run: absent capabilities are intentionally absent and
        # must not fire gate_not_evaluated. --max-cases previously did not
        # opt out, so a 40-case subset run failed on capabilities the
        # sample never reached — inconsistent with the other two flags.
        skip_absent_capabilities=bool(capability or tag or max_cases),
    )

    # Per-case scored records (drives paired bootstrap in `compare`).
    # Written wholesale, not appended: on a resume, `results` already
    # contains the reloaded completed cases, and appending them again on
    # top of the rows the first pass wrote duplicates every resumed row
    # (250 rows for a 125-question run), corrupting compare's pairing.
    # Rewriting from the in-memory results is idempotent and always
    # matches metrics.json.
    store.write_scored(_scored_records(results))
    store.write_errors(_error_records(results))

    from jev_agent_eval.reports import render_html, render_markdown, representative_failures

    # v2 report artifacts (spec 54): scorecard, thresholds, robustness,
    # baselines, security, provenance, risk-coverage. Computed BEFORE the
    # reports render so the HTML dashboard can embed them. A failure here
    # is an evaluator bug, not a data problem, so it surfaces as an
    # internal error.
    v2_artifacts: dict[str, Any] = {}
    try:
        from jev_agent_eval.datasets.local_jsonl import LocalJSONLDataset
        from jev_agent_eval.reports.v2 import build_v2_artifacts

        _cases_by_id: dict = {}
        try:
            # Prefer the run's OWN cases.jsonl: it holds exactly what this
            # run executed — including variants the runner loaded via glob
            # (spec 14/15), which the dataset manifest does not list under
            # manifest.cases. Robustness pair analysis needs those variant
            # rows (parent_case_id / relation); a manifest-only map misses
            # them and reports n_pairs=0 (holdout run audit).
            _run_cases = store.run_dir / "cases.jsonl"
            if _run_cases.exists():
                for case in LocalJSONLDataset([_run_cases]).iter_cases():
                    _cases_by_id[case.case_id] = case
            else:
                for entry in loaded.cases:
                    for case in LocalJSONLDataset([(Path(datasets_root) / entry.path)]).iter_cases():
                        _cases_by_id[case.case_id] = case
                # Variant files (variants.jsonl) are not listed under
                # manifest.cases — the runner loads them via glob — but
                # robustness pair analysis needs the variant rows. Load any
                # integrity-listed variant files as a fallback.
                for integrity_path in (getattr(loaded, "integrity", None) or {}):
                    if "variant" not in Path(integrity_path).name:
                        continue
                    vpath = Path(datasets_root) / integrity_path
                    if vpath.exists():
                        for case in LocalJSONLDataset([vpath]).iter_cases():
                            if getattr(case, "is_variant", False):
                                _cases_by_id[case.case_id] = case
        except Exception:
            pass  # case-file access already validated above; skip on absence
        dataset_manifest_dict = None
        try:
            dataset_manifest_dict = {
                "name": getattr(loaded, "name", None),
                "version": getattr(loaded, "version", None),
                "schema_version": getattr(loaded, "schema_version", None),
                "sources": [
                    s.model_dump() if hasattr(s, "model_dump") else s
                    for s in getattr(loaded, "sources", [])
                ],
            }
        except Exception:
            dataset_manifest_dict = None
        v2_artifacts = build_v2_artifacts(
            results,
            _cases_by_id,
            metrics,
            violations,
            run_manifest,
            dataset_manifest_dict,
            seed=cfg.run.seed,
            redaction_enabled=cfg.run.redact_secrets,
        )
        for filename, content in v2_artifacts.items():
            getattr(store, "write_" + filename.replace(".json", "").replace(".jsonl", ""))(content)
        # The HTML template keys v2 sections without file extensions.
        v2_artifacts = {
            k.removesuffix(".json").removesuffix(".jsonl"): v for k, v in v2_artifacts.items()
        }
    except Exception as e:  # noqa: BLE001
        _exit(EXIT_INTERNAL, f"v2 report artifacts failed: {e}")

    store.write_manifest(run_manifest)
    store.write_summary({"manifest": run_manifest, "metrics": metrics, "violations": violations})
    store.write_metrics(metrics)
    store.write_slices(metrics.get("slices", {}))
    store.write_calibration(metrics.get("calibration", {}))
    failures = representative_failures(results)
    if "markdown" in cfg.reports.formats:
        store.write_markdown(render_markdown(run_manifest, metrics, violations, failures))
    if "html" in cfg.reports.formats:
        store.write_html(
            render_html(run_manifest, metrics, violations, failures, v2_artifacts=v2_artifacts)
        )

    print_summary(run_manifest, metrics, violations)
    console.print(f"run directory: {run_dir}")
    _exit(EXIT_SUCCESS if passed else EXIT_QUALITY_GATE)


# ---------------------------------------------------------------- replay


@app.command()
def replay(
    run_dir: Path = typer.Argument(..., help="Run directory to replay"),
    config: Path | None = typer.Option(
        None, help="Config override (defaults to the run's snapshot)"
    ),
) -> None:
    """Recompute scoring and reports without contacting JEV."""
    from jev_agent_eval.reports import (
        build_metrics,
        evaluate_gates,
        print_summary,
        representative_failures,
    )
    from jev_agent_eval.reports.markdown_report import render_markdown
    from jev_agent_eval.runners.replay import load_run_results
    from jev_agent_eval.storage.artifacts import ArtifactStore

    if not (run_dir / "manifest.json").exists():
        _exit(EXIT_DATASET, f"not a run directory: {run_dir}")
    manifest, results = load_run_results(run_dir)
    # Restore the run's OWN configuration when snapshotted: replay must
    # evaluate the same gates the run was judged by. An explicit --config
    # overrides; the defaults apply only for runs with no snapshot.
    try:
        if config is not None:
            cfg = load_config(config)
        elif isinstance(manifest.get("config"), dict):
            from jev_agent_eval.config import EvalConfig

            cfg = EvalConfig.model_validate(manifest["config"])
        else:
            cfg = load_config(None)
    except ConfigError as e:
        _exit(EXIT_CONFIG, str(e))
    except Exception as e:
        _exit(EXIT_CONFIG, f"cannot restore run configuration: {e}")
    metrics = build_metrics(
        results,
        ece_bins=cfg.scoring.ece_bins,
        seed=manifest.get("seed", cfg.run.seed),
    )
    passed, violations = evaluate_gates(
        metrics,
        cfg.quality_gates,
        transport_error_rate_max=cfg.thresholds.transport_error_rate_max,
        schema_error_rate_max=cfg.thresholds.schema_error_rate_max,
        p95_latency_ms_max=cfg.thresholds.p95_latency_ms_max,
    )
    store = ArtifactStore(run_dir)
    # scored.jsonl / errors.jsonl must be REGENERATED from the replayed
    # results: they drive compare's paired bootstrap, and a stale copy
    # would contradict the recomputed metrics.json (bootstrap said
    # Δ=+0.0000 while the accuracy table showed −0.80).
    scored_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    for r in results:
        if r.correct:
            for qid, ok in r.correct.items():
                scored_records.append(
                    {
                        "case_id": r.case_id,
                        "capability": r.capability,
                        "question_id": qid,
                        "repeat_index": r.repeat_index,
                        "correct": bool(ok),
                        "request_hash": r.request_hash,
                    }
                )
        else:
            scored_records.append(
                {
                    "case_id": r.case_id,
                    "capability": r.capability,
                    "question_id": None,
                    "repeat_index": r.repeat_index,
                    "correct": False,
                    "request_hash": r.request_hash,
                }
            )
            error_records.append(
                {
                    "case_id": r.case_id,
                    "capability": r.capability,
                    "status": r.status.value,
                    "error_detail": r.error_detail,
                    "categories": [c.value for c in r.failure_categories],
                    "request_hash": r.request_hash,
                }
            )
    store.write_scored(scored_records)
    store.write_errors(error_records)
    store.write_summary({"manifest": manifest, "metrics": metrics, "violations": violations})
    store.write_metrics(metrics)
    store.write_slices(metrics.get("slices", {}))
    store.write_calibration(metrics.get("calibration", {}))
    store.write_markdown(
        render_markdown(manifest, metrics, violations, representative_failures(results))
    )
    # Replay must regenerate the SAME artifact set as `run` (spec 85 item
    # 11): scorecard, thresholds, robustness, baselines, security,
    # provenance, risk-coverage, and the HTML dashboard — recomputed from
    # cached responses, never contacting the provider.
    try:
        from jev_agent_eval.datasets.local_jsonl import LocalJSONLDataset
        from jev_agent_eval.reports.v2 import build_v2_artifacts

        cases_by_id: dict = {}
        # Prefer the run's OWN cases.jsonl (exactly what was executed,
        # including glob-loaded variants — see the run-path note); the
        # dataset manifest alone misses variant rows for runs whose
        # manifest doesn't list variants.jsonl (e.g. holdout).
        _run_cases_path = Path(run_dir) / "cases.jsonl"
        if _run_cases_path.exists():
            for case in LocalJSONLDataset([_run_cases_path]).iter_cases():
                cases_by_id[case.case_id] = case
        dataset_ref = manifest.get("dataset_manifest") or manifest.get("dataset")
        if dataset_ref and not cases_by_id:
            ds_path = Path(str(dataset_ref))
            if not ds_path.exists() and (Path.cwd() / ds_path).exists():
                ds_path = Path.cwd() / ds_path
            if ds_path.exists():
                for entry_path in _manifest_case_paths(ds_path):
                    for case in LocalJSONLDataset([entry_path]).iter_cases():
                        cases_by_id[case.case_id] = case
        dataset_manifest_dict = None
        if dataset_ref and Path(str(dataset_ref)).exists():
            from jev_agent_eval.datasets.registry import load_manifest as _lm2

            loaded_ds = _lm2(Path(str(dataset_ref)))
            dataset_manifest_dict = {
                "name": getattr(loaded_ds, "name", None),
                "version": getattr(loaded_ds, "version", None),
                "schema_version": getattr(loaded_ds, "schema_version", None),
                "sources": [
                    s.model_dump() if hasattr(s, "model_dump") else s
                    for s in getattr(loaded_ds, "sources", [])
                ],
            }
        v2 = build_v2_artifacts(
            results,
            cases_by_id,
            metrics,
            violations,
            manifest,
            dataset_manifest_dict,
            seed=cfg.run.seed,
            redaction_enabled=cfg.run.redact_secrets,
        )
        for filename, content in v2.items():
            getattr(store, "write_" + filename.replace(".json", "").replace(".jsonl", ""))(content)
        from jev_agent_eval.reports import render_html

        store.write_html(
            render_html(
                manifest,
                metrics,
                violations,
                representative_failures(results),
                v2_artifacts={
                    k.removesuffix(".json").removesuffix(".jsonl"): v for k, v in v2.items()
                },
            )
        )
    except Exception as e:  # noqa: BLE001
        # v2 artifacts are additive; a failure to regenerate them must not
        # invalidate the replayed core metrics.
        console.print(f"[yellow]v2 artifact regeneration failed: {e}[/yellow]")
    print_summary(manifest, metrics, violations)
    console.print(f"[dim]replayed (no network access): {run_dir}[/dim]")
    _exit(EXIT_SUCCESS if passed else EXIT_QUALITY_GATE)


# ---------------------------------------------------------------- compare


@app.command()
def compare(
    run_a: Path = typer.Argument(...),
    run_b: Path = typer.Argument(...),
    format: str = typer.Option("markdown", help="markdown | json"),
) -> None:
    """Compare two runs: paired deltas with confidence classification (spec 54-55)."""
    from jev_agent_eval.reports.compare import compare_runs

    try:
        report = compare_runs(run_a, run_b)
    except FileNotFoundError as e:
        _exit(EXIT_DATASET, str(e))
    if format == "json":
        console.print_json(json.dumps(report, default=str))
    else:
        from jev_agent_eval.reports.compare import render_comparison_markdown

        console.print(render_comparison_markdown(report))
    regressed = any(
        c["classification"] == "regressed" for c in report.get("capabilities", {}).values()
    )
    _exit(EXIT_QUALITY_GATE if regressed else EXIT_SUCCESS)


# ---------------------------------------------------------------- misc commands


@app.command()
def case(case_id: str, run_dir: Path | None = typer.Option(None)) -> None:
    """Inspect a single case, including its raw response and failure detail."""
    from jev_agent_eval.reports.terminal_report import print_failure_triage
    from jev_agent_eval.runners.replay import load_run_results

    dirs: list[Path] = []
    if run_dir:
        dirs = [run_dir]
    else:
        root = Path("reports/runs")
        dirs = sorted(root.iterdir()) if root.exists() else []
    for d in reversed(dirs):
        if not (d / "responses.jsonl").exists():
            continue
        manifest, results = load_run_results(d)
        for r in results:
            if r.case_id == case_id:
                gold = getattr(r, "gold_snapshot", {}) or {}
                # Show EVERY answer for the case, not just the first: a
                # multi-question case (tool_risk asks is_high_risk AND
                # risk_class) silently hid all but one answer, so the
                # triage view could not show what the model actually
                # answered for the second question.
                print_failure_triage(
                    {
                        "case_id": r.case_id,
                        "capability": r.capability,
                        "source": r.source,
                        "state": (getattr(r, "case_state", "") or "")[:600],
                        "gold": {k: getattr(v, "value", v) for k, v in gold.items()},
                        "predictions": r.predictions or {},
                        "latency_ms": r.transport.latency_ms if r.transport else None,
                        "error_class": r.status.value,
                        "request_hash": r.request_hash,
                    }
                )
                _exit(EXIT_SUCCESS)
    _exit(EXIT_DATASET, f"case {case_id!r} not found in any run")


@app.command()
def report(run_dir: Path = typer.Argument(...)) -> None:
    """Re-print the terminal summary for a run directory."""
    from jev_agent_eval.reports import print_summary
    from jev_agent_eval.storage.artifacts import ArtifactStore

    try:
        store = ArtifactStore(run_dir)
        summary = store.read_manifest()
        # summary.json holds metrics + violations
        with open(run_dir / "summary.json") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        _exit(EXIT_DATASET, str(e))
    print_summary(summary, data.get("metrics", {}), data.get("violations", []))
    # The report command must reflect the run's gate verdict: run and
    # replay exit 1 on gate failure, and a CI script checking a stored
    # run via `report` must get the same answer, not a 0 that greens a
    # failed run.
    _exit(EXIT_QUALITY_GATE if data.get("violations") else EXIT_SUCCESS)


@app.command()
def validate_response(json_path: Path = typer.Argument(...)) -> None:
    """Validate a JEV response JSON file against the response schema."""
    from jev_agent_eval.schemas.response import JEVResponse

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        JEVResponse.model_validate(data)
    except FileNotFoundError as e:
        _exit(EXIT_DATASET, str(e))
    except json.JSONDecodeError as e:
        _exit(EXIT_SCHEMA, f"malformed JSON: {e}")
    except Exception as e:
        _exit(EXIT_SCHEMA, f"schema validation failed: {e}")
    # Structural validation passed; apply the same SEMANTIC checks the
    # scorer applies (spec 34). The schema deliberately tolerates invalid
    # probability vectors so bad responses can be recorded and categorized
    # during runs — but this command exists to check a provider response
    # against the full contract, so a probabilities sum of 2.4 must fail.
    from jev_agent_eval.scoring import validate_probability_vector

    problems: list[str] = []
    if data.get("usage") is None:
        problems.append("usage field absent")
    for qid, answer in (data.get("answers") or {}).items():
        probs = answer.get("probabilities")
        if isinstance(probs, dict) and not validate_probability_vector(probs):
            problems.append(f"invalid probability vector for {qid!r}: {probs}")
    if problems:
        _exit(EXIT_SCHEMA, "; ".join(problems))
    console.print("[green]valid JEV response[/green]")
    _exit(EXIT_SUCCESS)


@app.command()
def export(run_dir: Path = typer.Argument(...), format: str = typer.Option("json")) -> None:
    """Export a run's results in a machine-readable format."""
    from jev_agent_eval.storage.artifacts import ArtifactStore

    if format != "json":
        _exit(EXIT_CONFIG, f"unsupported export format: {format}")
    try:
        store = ArtifactStore(run_dir)
        payload = {
            "manifest": store.read_manifest(),
            "cases": list(store.iter_cases()),
            "responses": list(store.iter_responses()),
        }
    except FileNotFoundError as e:
        _exit(EXIT_DATASET, str(e))
    console.print_json(json.dumps(payload, default=str))
    _exit(EXIT_SUCCESS)


@app.command()
def contract(live: bool = typer.Option(False, help="Run against the live provider")) -> None:
    """Contract suite against JEV: 5-20 cases, disabled by default (spec 42)."""
    from jev_agent_eval.client.systemone import MockJEVProvider, SystemOneProvider
    from jev_agent_eval.schemas.request import JEVQuestion, JEVRequest

    prov: InferenceProvider
    if live:
        try:
            prov = SystemOneProvider()
        except RuntimeError as e:
            _exit(EXIT_AUTH, str(e))
    else:
        prov = MockJEVProvider()

    request = JEVRequest(
        state="Contract check: the user asked the agent to fix a failing test.",
        model="jev-latest",
        questions={
            "needs_tool": JEVQuestion(type="noul", instructions="Does this require a tool call?"),
            "task_type": JEVQuestion(
                type="choice",
                instructions="Classify the task.",
                criteria={"bug_fix": "Fixing broken behavior.", "research": "Investigation only."},
            ),
            "effort": JEVQuestion(
                type="score",
                instructions="How much effort is needed?",
                criteria=["minimal", "moderate", "substantial"],
            ),
        },
    )
    result = asyncio.run(prov.infer_async(request)) if live else prov.infer(request)
    response = result.response
    if response is None:
        _exit(
            EXIT_TRANSPORT,
            f"contract failed: {result.network_error_class or result.parse_error} "
            f"http={result.http_status}",
        )
    answers = response.answers
    problems: list[str] = []
    for qid in request.questions:
        if qid not in answers:
            problems.append(f"missing answer: {qid}")
    # model_dump() always emits the usage key (Optional, default None), so
    # presence must be checked against the raw parsed answer, not the dump.
    if response.usage is None:
        problems.append("usage field absent")
    if problems:
        _exit(EXIT_SCHEMA, "; ".join(problems))
    console.print("[green]contract suite passed[/green]")
    _exit(EXIT_SUCCESS)


@app.command()
def resume(run_dir: Path = typer.Argument(...)) -> None:
    """Resume an interrupted run: execute only missing cases (spec 71)."""
    from jev_agent_eval.storage.artifacts import ArtifactStore

    try:
        manifest = ArtifactStore(run_dir).read_manifest()
    except FileNotFoundError as e:
        _exit(EXIT_DATASET, str(e))
    if manifest.get("completed"):
        console.print("[green]run already complete[/green]")
        _exit(EXIT_SUCCESS)
    dataset = manifest.get("dataset_manifest")
    if not dataset:
        _exit(EXIT_CONFIG, "run manifest lacks dataset_manifest")
    # The run directory name MUST equal the manifest's run_id: resume
    # writes into <output-dir>/runs/<run_id>, so a renamed directory would
    # otherwise silently produce a fresh full run next to the stale one
    # instead of completing the interrupted run.
    run_id = manifest.get("run_id")
    if run_id and run_dir.name != run_id:
        _exit(
            EXIT_DATASET,
            f"run directory {run_dir.name!r} does not match manifest run_id {run_id!r}; "
            "rename the directory to the run_id to resume it",
        )
    if run_dir.parent.name != "runs":
        _exit(
            EXIT_DATASET,
            f"run directory must live under a 'runs/' parent (got {run_dir.parent}); "
            "resume re-creates <output-dir>/runs/<run_id>",
        )
    # Re-invoke the run command in-process via the Typer app (calling the
    # command function directly would pass OptionInfo sentinels instead
    # of real values, and `python -m jev_agent_eval` would crash: there is
    # no __main__ module). --resume-id reuses the original run id so
    # artifacts land in this directory and completed cases are skipped.
    argv = [
        "run",
        "--dataset",
        dataset,
        "--suite",
        manifest.get("suite") or "smoke",
        "--resume-id",
        run_id or "",
        # Keep artifacts in the original run's directory tree.
        "--output-dir",
        str(run_dir.parent.parent),
    ]
    if manifest.get("model"):
        argv += ["--model", manifest["model"]]
    # Reuse the original provider: a mock run must not try to reach the
    # live SystemOne endpoint on resume.
    provider_name = manifest.get("provider") or (
        "mock" if str(manifest.get("provider_url", "")).startswith("mock://") else "systemone"
    )
    argv += ["--provider", provider_name]
    # Restore the original run's resolved configuration (concurrency, rate
    # limit, repeats, gates) when the manifest snapshots it: resume must
    # continue under the SAME operating parameters the run started with,
    # not silently fall back to defaults. The snapshot is dumped to a temp
    # YAML file and passed through --config so the run command's normal
    # config-loading and validation path applies unchanged.
    cfg_snapshot = manifest.get("config")
    if isinstance(cfg_snapshot, dict):
        import atexit
        import json as _json
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as cfg_file:
            _json.dump(cfg_snapshot, cfg_file)
            cfg_path = Path(cfg_file.name)
        argv += ["--config", str(cfg_path)]
        # Best-effort cleanup once the (in-process) child invocation has
        # loaded the config; a leftover temp file is harmless.
        atexit.register(lambda: cfg_path.unlink(missing_ok=True))
    try:
        app(argv, standalone_mode=False)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
        raise typer.Exit(code) from e


@app.command()
def trace(
    input_path: Path = typer.Argument(...),
    suite: str = typer.Option("control-plane"),
    model: str = typer.Option("jev-latest"),
    provider: str = typer.Option("mock"),
    output_dir: Path = typer.Option(Path("reports")),
) -> None:
    """Evaluate JEV decisions over a normalized agent trace file (spec 13.3)."""
    from jev_agent_eval.client.systemone import MockJEVProvider, SystemOneProvider
    from jev_agent_eval.runners.offline import OfflineRunner
    from jev_agent_eval.runners.trace import trace_to_cases

    if not input_path.exists():
        _exit(EXIT_DATASET, f"trace file not found: {input_path}")
    try:
        cases = trace_to_cases(input_path)
    except ValueError as e:
        # Malformed trace gold is a user-input (dataset) error: exit 3 with
        # a clean message, not a raw traceback.
        _exit(EXIT_DATASET, f"invalid trace file: {e}")
    if not cases:
        _exit(EXIT_DATASET, "no evaluable steps in trace (no gold labels found)")
    prov = MockJEVProvider(model=model) if provider == "mock" else SystemOneProvider(model=model)
    runner = OfflineRunner(
        prov,
        manifest_path=input_path,  # label only; direct cases bypass the loader
        suite="release",
        output_dir=output_dir,
        model=model,
        cases=cases,
        dataset_hash="trace",
    )
    run_id, results = runner.run()
    # A trace run must produce the same artifact set as `run`: previously it
    # stopped after the runner, leaving responses without metrics, scored
    # records, or a report — an evaluated trace was impossible to inspect
    # or compare without hand-rolling the scoring.
    from jev_agent_eval.config import load_config as _load_cfg
    from jev_agent_eval.reports import (
        build_metrics,
        evaluate_gates,
        print_summary,
        render_html,
        render_markdown,
        representative_failures,
    )
    from jev_agent_eval.storage.artifacts import ArtifactStore

    cfg = _load_cfg(None)
    run_dir = output_dir / "runs" / run_id
    store = ArtifactStore(run_dir)
    run_manifest = store.read_manifest()
    run_manifest["config"] = cfg.model_dump(exclude_none=True)
    metrics = build_metrics(
        results,
        bootstrap_samples=cfg.scoring.bootstrap_samples,
        confidence_level=cfg.scoring.confidence_level,
        ece_bins=cfg.scoring.ece_bins,
        seed=cfg.run.seed,
    )
    passed, violations = evaluate_gates(
        metrics,
        cfg.quality_gates,
        transport_error_rate_max=cfg.thresholds.transport_error_rate_max,
        schema_error_rate_max=cfg.thresholds.schema_error_rate_max,
        p95_latency_ms_max=cfg.thresholds.p95_latency_ms_max,
        # A trace contains whatever capabilities the traced agent exercised;
        # unlike a dataset run (where the manifest promises the full set), an
        # absent capability is not a gate violation here.
        skip_absent_capabilities=True,
    )
    # Wholesale write, not append — see the run command's note on resume
    # duplication; the same idempotence guarantee applies to trace runs.
    store.write_scored(_scored_records(results))
    store.write_errors(_error_records(results))
    store.write_manifest(run_manifest)
    store.write_summary({"manifest": run_manifest, "metrics": metrics, "violations": violations})
    store.write_metrics(metrics)
    store.write_slices(metrics.get("slices", {}))
    store.write_calibration(metrics.get("calibration", {}))
    failures = representative_failures(results)
    if "markdown" in cfg.reports.formats:
        store.write_markdown(render_markdown(run_manifest, metrics, violations, failures))
    if "html" in cfg.reports.formats:
        store.write_html(render_html(run_manifest, metrics, violations, failures))
    print_summary(run_manifest, metrics, violations)
    console.print(f"trace run complete: {len(results)} steps, run_id={run_id}")
    console.print(f"run directory: {run_dir}")
    _exit(EXIT_SUCCESS if passed else EXIT_QUALITY_GATE)


@app.command()
def metamorphic(
    dataset: Path = typer.Option(..., help="Dataset manifest path"),
    provider: str = typer.Option("mock", help="Provider: systemone | mock"),
    model: str = typer.Option("jev-latest"),
    max_originals: int = typer.Option(20, help="Number of original cases to vary"),
    seed: int = typer.Option(20260920),
    output: Path = typer.Option(Path("reports/metamorphic.json")),
) -> None:
    """Metamorphic invariance suite: transformations that must not change decisions (spec 27)."""
    import json as _json

    from jev_agent_eval.client.systemone import MockJEVProvider, SystemOneProvider
    from jev_agent_eval.metamorphic import MetamorphicReport, build_variants
    from jev_agent_eval.questions import get_pack
    from jev_agent_eval.schemas.request import JEVRequest

    try:
        manifest = load_manifest(dataset)
        from jev_agent_eval.datasets.registry import load_manifest_cases

        cases = load_manifest_cases(manifest, dataset.parent)
    except Exception as e:
        _exit(EXIT_DATASET, f"dataset error: {e}")

    prov: InferenceProvider
    if provider == "mock":
        prov = MockJEVProvider(model=model)
    else:
        try:
            prov = SystemOneProvider(model=model)
        except RuntimeError as e:
            _exit(EXIT_AUTH, str(e))

    originals = cases[:max_originals]
    variants = build_variants(originals, seed=seed, max_originals=max_originals)
    report = MetamorphicReport(n_originals=len(originals), n_variants=len(variants))

    def decide(case) -> dict[str, str]:
        pack = get_pack(case.capability).build(case)
        req = JEVRequest(state=case.state, model=model, questions=pack.questions)
        res = prov.infer(req)
        out: dict[str, str] = {}
        if res.response is not None:
            for qid, ans in res.response.answers.items():
                out[qid] = str(ans.choice if ans.type == "choice" else ans.noul)
        return out

    base_decisions = {c.case_id: decide(c) for c in originals}
    for v in variants:
        variant_decision = decide(v.case)
        base = base_decisions.get(v.case.case_id.split("::mm-")[0], {})
        for qid, val in variant_decision.items():
            if v.transform == "semantic_negation":
                # Negation is only defined for boolean propositions: a class
                # label ("safe" -> ???) or an ordinal score has no opposite.
                # Including them made the relation unsatisfiable (a flipped
                # risk_class can never be checked against gold-less variants
                # and permanently reported pass_rate=0.0).
                q = v.case.questions.get(qid)
                if q is None or q.get("type") != "noul":
                    continue
            report.comparisons.setdefault(v.transform, []).append((base.get(qid), val, None))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    summary = report.to_dict()
    for transform, stats in sorted(summary["transforms"].items()):
        console.print(
            f"  {transform}: pass_rate={stats['pass_rate']}"
            if stats["pass_rate"] is not None
            else f"  {transform}: n=0"
        )
    # A failing invariance relation is a quality regression, not a success:
    # CI distinguishes quality (exit 1) from evaluator breakage (exit 7)
    # via the exit-code contract, so a suite with pass_rate < 1 must not
    # exit 0 and silently green the pipeline.
    failed = [
        t
        for t, stats in summary["transforms"].items()
        if stats["pass_rate"] is not None and stats["pass_rate"] < 1.0
    ]
    if failed:
        console.print(f"[red]metamorphic relations failed: {', '.join(sorted(failed))}[/red]")
        _exit(EXIT_QUALITY_GATE)
    _exit(EXIT_SUCCESS)


@app.command()
def consistency(
    dataset: Path = typer.Option(..., help="Dataset manifest path"),
    provider: str = typer.Option("mock", help="Provider: systemone | mock"),
    model: str = typer.Option("jev-latest"),
    repeats: int = typer.Option(5, help="Repeats per case (spec 28: 5)"),
    max_cases: int = typer.Option(10),
    output: Path = typer.Option(Path("reports/consistency.json")),
) -> None:
    """Repeatability suite: same case run N times, label/probability stability (spec 28)."""
    import json as _json

    from jev_agent_eval.client.systemone import MockJEVProvider, SystemOneProvider
    from jev_agent_eval.consistency import consistency_report
    from jev_agent_eval.datasets.registry import load_manifest_cases

    try:
        manifest = load_manifest(dataset)
        cases = load_manifest_cases(manifest, dataset.parent)
    except Exception as e:
        _exit(EXIT_DATASET, f"dataset error: {e}")

    prov: InferenceProvider
    if provider == "mock":
        prov = MockJEVProvider(model=model)
    else:
        try:
            prov = SystemOneProvider(model=model)
        except RuntimeError as e:
            _exit(EXIT_AUTH, str(e))

    reports = [
        consistency_report(prov, c, c.questions, model, repeats=repeats) for c in cases[:max_cases]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _json.dumps({"repeats": repeats, "cases": reports}, indent=2), encoding="utf-8"
    )
    n_stable = sum(
        1 for r in reports if all(q.get("label_stability") == 1.0 for q in r["questions"].values())
    )
    console.print(
        f"consistency: {n_stable}/{len(reports)} cases fully stable over {repeats} repeats"
    )
    # A model with unstable labels on identical inputs is a quality
    # finding: exit 1 so CI sees it (exit-code contract: 1 = quality
    # regression, 7 = evaluator broken).
    if reports and n_stable < len(reports):
        console.print("[red]label instability detected[/red]")
        _exit(EXIT_QUALITY_GATE)
    _exit(EXIT_SUCCESS)


# ---------------------------------------------------------------- performance


@app.command()
def performance(
    model: str | None = typer.Option(None),
    config: Path | None = typer.Option(None),
    profile: str = typer.Option("release", help="Performance profile (spec 22.2 levels)"),
    levels: str = typer.Option("1,2,4,8", help="Concurrency levels (comma-separated, spec 22.2)"),
    requests_per_level: int = typer.Option(20, help="Requests per concurrency level"),
    rates: str | None = typer.Option(
        None, help="Rate-limit sweep RPS values (comma-separated, spec 22.3)"
    ),
    output: Path | None = typer.Option(None, help="Output JSON path"),
) -> None:
    """Run the latency/concurrency/rate-limit performance suite (spec 22)."""
    import json as _json

    try:
        cfg = load_config(config)
    except ConfigError as e:
        _exit(EXIT_CONFIG, str(e))
    if model:
        cfg.provider.model = model

    # Profile presets (spec 22.2): full sweep vs smoke-level verification.
    profile_levels = {
        "release": "1,2,4,8,16,32,64",
        "smoke": "1,2",
        "pr": "1,4,8",
        "nightly": "1,2,4,8,16",
    }
    if profile and profile not in ("default",):
        if profile in profile_levels:
            levels = profile_levels[profile]
        else:
            _exit(
                EXIT_CONFIG,
                f"unknown profile {profile!r}; expected one of {sorted(profile_levels)}",
            )

    try:
        level_values = tuple(int(x) for x in levels.split(",") if x.strip())
    except ValueError:
        _exit(EXIT_CONFIG, f"invalid --levels {levels!r}: expected comma-separated integers")
    rate_values: list[float] | None = None
    if rates:
        try:
            rate_values = [float(x) for x in rates.split(",") if x.strip()]
        except ValueError:
            _exit(EXIT_CONFIG, f"invalid --rates {rates!r}: expected comma-separated numbers")

    from jev_agent_eval.client.systemone import SystemOneProvider
    from jev_agent_eval.runners.performance import run_performance_suite

    try:
        prov = SystemOneProvider(
            base_url=cfg.provider.base_url,
            model=cfg.provider.model,
            timeout_s=cfg.provider.timeout_s,
            connect_timeout_s=cfg.provider.connect_timeout_s,
            read_timeout_s=cfg.provider.read_timeout_s,
            max_retries=cfg.provider.max_retries,
            backoff_base_s=cfg.provider.backoff_base_s,
            backoff_max_s=cfg.provider.backoff_max_s,
            requests_per_second=cfg.provider.requests_per_second,
        )
    except RuntimeError as e:
        _exit(EXIT_AUTH, str(e))

    try:
        result = run_performance_suite(
            prov,
            levels=level_values,
            requests_per_level=requests_per_level,
            rates=rate_values,
        )
    except Exception as e:  # noqa: BLE001
        _exit(EXIT_INTERNAL, f"performance suite failed: {e}")

    out_path = output or Path(cfg.reports.output_dir) / "performance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(result, indent=2), encoding="utf-8")
    sweep = result.get("concurrency_sweep", {})
    console.print(f"performance suite: {len(sweep)} concurrency levels -> {out_path}")
    for name, stats in sorted(sweep.items()):
        p95 = stats.get("attempt_latency_p95_ms")
        console.print(f"  {name}: n={stats.get('n')} attempt_p95={p95 and round(p95, 1)}ms")
    _exit(EXIT_SUCCESS)


# ---------------------------------------------------------------- soak


def _parse_duration(value: str) -> float:
    """Parse '30m', '2h', '90s', or a bare seconds number."""
    value = value.strip().lower()
    multipliers = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    if value and value[-1] in multipliers:
        try:
            return float(value[:-1]) * multipliers[value[-1]]
        except ValueError:
            pass
    try:
        return float(value)
    except ValueError as e:
        raise ConfigError(f"invalid duration {value!r}: expected e.g. '30m', '2h', '1800'") from e


@app.command()
def soak(
    model: str | None = typer.Option(None),
    config: Path | None = typer.Option(None),
    duration: str = typer.Option(
        "30m", help="Soak duration, e.g. '30m' or '1800' (spec 22.4: 30 min)"
    ),
    workers: int = typer.Option(4, help="Concurrent workers (1-8 depending on provider limits)"),
    sample_interval_s: float = typer.Option(30.0),
    output: Path | None = typer.Option(None),
) -> None:
    """Run the sustained-load soak test (spec 22.4)."""
    import json as _json

    try:
        cfg = load_config(config)
        duration_s = _parse_duration(duration)
    except ConfigError as e:
        _exit(EXIT_CONFIG, str(e))
    if model:
        cfg.provider.model = model

    from jev_agent_eval.client.systemone import SystemOneProvider
    from jev_agent_eval.runners.performance import run_soak_suite

    try:
        prov = SystemOneProvider(
            base_url=cfg.provider.base_url,
            model=cfg.provider.model,
            timeout_s=cfg.provider.timeout_s,
            connect_timeout_s=cfg.provider.connect_timeout_s,
            read_timeout_s=cfg.provider.read_timeout_s,
            max_retries=cfg.provider.max_retries,
            backoff_base_s=cfg.provider.backoff_base_s,
            backoff_max_s=cfg.provider.backoff_max_s,
            requests_per_second=cfg.provider.requests_per_second,
        )
    except RuntimeError as e:
        _exit(EXIT_AUTH, str(e))

    console.print(f"soak: {duration_s}s at {workers} workers (this runs to completion)")
    try:
        result = run_soak_suite(
            prov,
            duration_s=duration_s,
            workers=workers,
            sample_interval_s=sample_interval_s,
        )
    except Exception as e:  # noqa: BLE001
        _exit(EXIT_INTERNAL, f"soak suite failed: {e}")

    out_path = output or Path(cfg.reports.output_dir) / "soak.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(result, indent=2), encoding="utf-8")
    drift = result.get("drift", {})
    console.print(
        f"soak complete: {result.get('completed_requests')} requests, "
        f"throughput drift {drift.get('throughput_drift_pct')}%, "
        f"memory growth {result.get('memory_growth_mb')} MB -> {out_path}"
    )
    _exit(EXIT_SUCCESS)


@app.command()
def version() -> None:
    console.print(f"jev-agent-eval {__version__}")
    _exit(EXIT_SUCCESS)


if __name__ == "__main__":
    app()
