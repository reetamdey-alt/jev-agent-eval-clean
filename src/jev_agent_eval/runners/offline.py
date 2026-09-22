"""Offline benchmark runner (spec sections 13.1, 70, 72).

Executes canonical cases against a provider with bounded async concurrency,
deterministic ordering (capability, case_id), caching, incremental artifact
writes, and resume support.
"""

from __future__ import annotations

import asyncio
import json
import platform
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jev_agent_eval import __version__
from jev_agent_eval.client.base import InferenceProvider, InferenceResult
from jev_agent_eval.client.redaction import redact_text
from jev_agent_eval.config import ConfigError
from jev_agent_eval.datasets.registry import load_manifest, load_manifest_cases
from jev_agent_eval.questions.base import QuestionPack
from jev_agent_eval.questions.registry import get_pack
from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.request import JEVRequest
from jev_agent_eval.schemas.result import (
    CaseResult,
    CaseStatus,
    FailureCategory,
    TransportRecord,
    utc_now_iso,
)
from jev_agent_eval.storage.artifacts import ArtifactStore, new_run_id
from jev_agent_eval.storage.cache import ResponseCache, cache_key
from jev_agent_eval.utils import iter_jsonl

# Suite case caps (spec 39): smoke ~100, pr ~500, release = all.
# v2 spec 52/76 suite sizes (base cases): smoke 250-400, pr 1500-2000,
# nightly 5000-7500, release unlimited (12,000+), holdout/redteam private.
SUITE_LIMITS = {
    "smoke": 300,
    "pr": 1500,
    "nightly": 5000,
    "release": None,
    "holdout": None,
    "redteam": None,
}


def _proportional_sample(cases: list[CanonicalCase], limit: int) -> list[CanonicalCase]:
    """Take a suite-size-limited sample that covers every capability.

    A naive head-cut after sorting by (capability, case_id) would take only
    the alphabetically-first capabilities; instead each capability receives
    a share of `limit` proportional to its share of the full dataset
    (deterministic: sorted case ids, largest-remainder rounding).
    """
    by_cap: dict[str, list[CanonicalCase]] = {}
    for c in cases:
        by_cap.setdefault(c.capability, []).append(c)
    quotas: dict[str, int] = {}
    remainder: list[tuple[float, str]] = []
    for cap, group in by_cap.items():
        exact = limit * len(group) / len(cases)
        quotas[cap] = int(exact)
        remainder.append((exact - quotas[cap], cap))
    # Distribute leftover slots to the largest fractional remainders.
    leftover = limit - sum(quotas.values())
    for _, cap in sorted(remainder, key=lambda t: (-t[0], t[1]))[:leftover]:
        quotas[cap] += 1
    sampled: list[CanonicalCase] = []
    for cap in sorted(by_cap):
        group = by_cap[cap]
        sampled.extend(group[: quotas[cap]])
    # Re-apply deterministic global ordering (spec 72).
    sampled.sort(key=lambda c: (c.capability, c.case_id))
    return sampled


def _parse_raw_response_for_reload(raw: Any) -> dict[str, Any] | None:
    """responses.jsonl stores the (redacted) raw body as JSON text; parse
    it back for CaseResult.raw_response (mirror of replay.py's helper)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class OfflineRunner:
    def __init__(
        self,
        provider: InferenceProvider,
        *,
        manifest_path: str | Path,
        suite: str = "release",
        concurrency: int = 8,
        repeats: int = 1,
        seed: int = 20260920,
        use_cache: bool = True,
        save_raw: bool = True,
        redact_secrets: bool = True,
        output_dir: str | Path = "reports",
        run_id: str | None = None,
        capability: str | None = None,
        tag: str | None = None,
        max_cases: int | None = None,
        fail_fast: bool = False,
        shuffle: bool = False,
        pack_name: str = "prebaked",
        dataset_hash: str | None = None,
        provider_url: str = "unknown",
        model: str = "jev-latest",
        cases: list[CanonicalCase] | None = None,
    ):
        self.provider = provider
        self.manifest_path = Path(manifest_path)
        # Direct case list (trace mode / programmatic use) bypasses the
        # manifest loader and its allowed-roots path check.
        self.direct_cases = cases
        self.suite = suite
        self.concurrency = max(1, concurrency)
        self.repeats = repeats
        self.seed = seed
        self.use_cache = use_cache
        self.save_raw = save_raw
        self.redact_secrets = redact_secrets
        self.output_dir = Path(output_dir)
        self.run_id = run_id or new_run_id()
        self.capability_filter = capability
        self.tag_filter = tag
        self.max_cases = max_cases
        self.fail_fast = fail_fast
        self.shuffle = shuffle
        self.pack: QuestionPack = get_pack(pack_name)
        self.dataset_hash = dataset_hash
        self.provider_url = provider_url
        self.model = model

    # -- Case loading ---------------------------------------------------------

    def load_cases(self) -> list[CanonicalCase]:
        if self.direct_cases is not None:
            cases = list(self.direct_cases)
        else:
            manifest = load_manifest(self.manifest_path)
            cases = load_manifest_cases(manifest, self.manifest_path.parent)
        if self.capability_filter:
            cases = [c for c in cases if c.capability == self.capability_filter]
        if self.tag_filter:
            cases = [c for c in cases if self.tag_filter in c.tags]
        # dev_split isolation (spec 7): private-holdout rows living inside a
        # combined case store must never be evaluated by a public suite, and
        # the holdout suite must never see anything else.
        if self.suite in ("holdout", "redteam"):
            cases = [c for c in cases if str(c.dev_split or "") == "private_holdout"] or cases
        else:
            cases = [c for c in cases if str(c.dev_split or "") != "private_holdout"]
        # Variant loading (spec 14/15): release-grade suites also evaluate
        # generated variants for counterfactual/metamorphic pair analysis.
        # Variants carry is_variant=True so they never enter primary sample
        # counts (spec 64 invariant 1); smoke/pr keep runs minimal.
        if self.suite in ("release", "nightly", "holdout") and self.direct_cases is None:
            cases = cases + self._load_variants(cases)
        # Deterministic ordering (spec 72).
        cases.sort(key=lambda c: (c.capability, c.case_id))
        limit = SUITE_LIMITS.get(self.suite)
        if limit is None and self.suite not in SUITE_LIMITS:
            raise ConfigError(
                f"unknown suite {self.suite!r}; expected one of {sorted(SUITE_LIMITS)}"
            )
        if self.max_cases is not None:
            limit = self.max_cases if limit is None else min(limit, self.max_cases)
        if limit is not None and len(cases) > limit:
            cases = _proportional_sample(cases, limit)
        # --shuffle (spec 72): the execution ORDER is randomized, seeded by
        # the run seed so the shuffle itself is reproducible. The sampled
        # case SET is unaffected — sampling happens above, pre-shuffle.
        if self.shuffle:
            random.Random(self.seed).shuffle(cases)
        return cases

    def _load_variants(self, base_cases: list[CanonicalCase]) -> list[CanonicalCase]:
        """Load generated variants (variants.jsonl) referenced by the
        manifest's integrity section. Variants whose parent is not in the
        loaded base set are skipped: a pair without its parent cannot be
        compared and would only inflate the request count."""
        try:
            load_manifest(self.manifest_path)  # validates the manifest exists
        except Exception:  # noqa: BLE001
            return []
        # Variants live next to the primary case file(s) in the datasets root.
        datasets_root = self.manifest_path.resolve().parent.parent
        variant_paths = sorted(datasets_root.glob("**/variants.jsonl"))
        if not variant_paths:
            return []
        base_ids = {c.case_id for c in base_cases}
        variants: list[CanonicalCase] = []
        seen: set[str] = set()
        for vp in variant_paths:
            try:
                for obj in iter_jsonl(vp):
                    c = CanonicalCase.model_validate(obj)
                    if c.case_id in seen or c.case_id in base_ids:
                        continue
                    if not c.is_variant or not c.parent_case_id:
                        continue
                    if c.parent_case_id not in base_ids:
                        continue
                    seen.add(c.case_id)
                    variants.append(c)
            except Exception:  # noqa: BLE001
                continue
        return variants

    # -- Execution ------------------------------------------------------------

    async def _execute_one(
        self,
        case: CanonicalCase,
        repeat_index: int,
        store: ArtifactStore,
        cache: ResponseCache | None,
        sem: asyncio.Semaphore,
    ) -> CaseResult:
        async with sem:
            pack_result = self.pack.build(case)
            request = JEVRequest(
                state=case.state,
                model=self.model,
                questions=pack_result.questions,
            )
            # Cache key includes the provider endpoint (deployments sharing
            # provider.name+model must not silently reuse each other's
            # responses) and the repeat index (repeats 2+ must actually
            # contact the provider, not replay the same cached response).
            key = cache_key(
                f"{self.provider.name}/{self.provider_url}/{self.model}#{repeat_index}",
                request,
            )
            from_cache = False
            result: InferenceResult | None = None
            request_start = utc_now_iso()
            cached = cache.get(key) if cache is not None else None
            if cached and cached.get("status") == "success":
                from jev_agent_eval.schemas.response import JEVResponse

                try:
                    cached_response = JEVResponse.model_validate(cached["response"])
                except Exception:
                    # A corrupt cache entry (e.g. status "success" with a
                    # null/malformed response) is a cache MISS, not a run
                    # crash: re-infer and let the fresh response overwrite it.
                    cached_response = None
                if cached_response is not None:
                    from_cache = True
                    result = InferenceResult(
                        response=cached_response,
                        http_status=200,
                        latency_ms=cached.get("latency_ms"),
                        from_cache=True,
                    )

            if result is None:
                result = await self.provider.infer_async(request)
                if cache is not None and result.response is not None:
                    cache.put(
                        key,
                        model=self.model,
                        response=result.response.model_dump(),
                        latency_ms=result.latency_ms,
                        status="success" if result.response else "error",
                    )

            transport = TransportRecord(
                # Captured BEFORE dispatch, not after the response arrives.
                request_start=request_start,
                latency_ms=result.latency_ms,
                http_status=result.http_status,
                network_error_class=result.network_error_class,
                retry_count=result.retry_count,
                retry_history=result.retry_history,
                server_model=result.response.model if result.response else None,
                from_cache=from_cache,
                input_tokens=result.response.usage.input_tokens
                if result.response and result.response.usage
                else None,
                output_tokens=result.response.usage.output_tokens
                if result.response and result.response.usage
                else None,
            )

            # The raw body persisted for this case (redacted per config),
            # computed ONCE: it goes into the responses.jsonl row (as text)
            # AND onto the CaseResult (as the parsed object), so the
            # security report's raw_responses_saved counts what was
            # actually written — it used to read the never-populated
            # CaseResult.raw_response and always report 0 while
            # responses.jsonl held 300 raw bodies.
            raw_response_persisted = None
            raw_response_object = None
            if self.save_raw and result.raw_body:
                raw_response_persisted = (
                    redact_text(result.raw_body[:50000])
                    if self.redact_secrets
                    else result.raw_body[:50000]
                )
                try:
                    raw_response_object = json.loads(raw_response_persisted)
                except (json.JSONDecodeError, TypeError):
                    raw_response_object = None

            metadata = case.metadata or {}
            # Slice dimensions prefer the CANONICAL top-level fields:
            # 279/300 live cases carried source/difficulty only at the top
            # level (adapters and ingested traces), so reading them from
            # metadata dropped those cases from source/difficulty slices
            # and rendered "SOURCE: unknown" in the case triage.
            cr = CaseResult(
                case_id=case.case_id,
                dataset=case.dataset,
                dataset_version=case.dataset_version,
                capability=case.capability,
                repeat_index=repeat_index,
                status=CaseStatus.SUCCESS
                if result.response
                else (
                    CaseStatus.SCHEMA_ERROR if result.parse_error else CaseStatus.TRANSPORT_ERROR
                ),
                request_hash=key,
                question_ids=list(pack_result.questions.keys()),
                transport=transport,
                difficulty=case.difficulty or metadata.get("difficulty"),
                adversarial="adversarial" in case.tags,
                source=case.source or metadata.get("source"),
                tags=case.tags,
                state_chars=len(case.state),
                is_variant=case.is_variant,
                raw_response=raw_response_object,
            )
            # Gold snapshot rides on the result object (never in the JEV input).
            object.__setattr__(cr, "gold_snapshot", pack_result.gold)
            object.__setattr__(cr, "case_state", case.state)

            if result.response:
                predictions: dict[str, Any] = {}
                for qid, ans in result.response.answers.items():
                    predictions[qid] = {
                        "type": ans.type,
                        "noul": ans.noul,
                        "choice": ans.choice,
                        "score": ans.score,
                        "probabilities": ans.probabilities,
                        "confidence": ans.confidence,
                    }
                missing = [q for q in pack_result.questions if q not in predictions]
                if missing:
                    cr.status = CaseStatus.SCHEMA_ERROR
                    cr.error_detail = f"missing_answer: {missing}"
                    cr.failure_categories.append(FailureCategory.MISSING_ANSWER)
                else:
                    cr.predictions = predictions
            else:
                cr.error_detail = result.parse_error or (
                    f"{result.network_error_class or ''} http={result.http_status}"
                    f"{': ' + result.error_body if result.error_body else ''}"
                )
                # Apply the spec-33 taxonomy (rate_limited / provider_5xx /
                # transport_error / schema_error); previously only the
                # missing-answer branch categorized anything, leaving
                # network and parse failures unclassified in errors.jsonl.
                from jev_agent_eval.scoring import classify_failure

                category = classify_failure(cr, None, None)
                if category is not None and category not in cr.failure_categories:
                    cr.failure_categories.append(category)

            store.append_response(
                {
                    # Key format matches the resume done-check
                    # (f"{case_id}#{repeat}"): a resumed run must skip
                    # already-completed (case, repeat) pairs, not re-run
                    # and duplicate them.
                    "case_id": f"{case.case_id}#{repeat_index}",
                    "case_id_bare": case.case_id,
                    "repeat_index": repeat_index,
                    "request_hash": key,
                    "from_cache": from_cache,
                    "http_status": result.http_status,
                    "latency_ms": result.latency_ms,
                    "retry_count": result.retry_count,
                    # Per-attempt history (status, error, backoff actually
                    # slept): without it, responses.jsonl cannot answer
                    # "did the server-aware 429 backoff fire?" — the log
                    # audit of the 2026-09-22 rate-limit storm had to
                    # reconstruct it from latencies alone.
                    "retry_history": result.retry_history,
                    "network_error_class": result.network_error_class,
                    "parse_error": result.parse_error,
                    "error_body": (
                        redact_text(result.error_body)
                        if (self.redact_secrets and result.error_body)
                        else result.error_body
                    ),
                    "raw_response": raw_response_persisted,
                    "response": result.response.model_dump() if result.response else None,
                }
            )
            store.append_trace(
                {
                    "run_id": self.run_id,
                    # Same key format as responses.jsonl: a run with
                    # repeats > 1 has several trace records per bare case,
                    # and without the repeat suffix / index they cannot be
                    # told apart without a request_hash join.
                    "case_id": f"{case.case_id}#{repeat_index}",
                    "case_id_bare": case.case_id,
                    "repeat_index": repeat_index,
                    "request_hash": key,
                    "model": self.model,
                    "attempt": result.retry_count + 1,
                    "latency_ms": result.latency_ms,
                    "status": "success" if result.response else "error",
                    "input_tokens": transport.input_tokens,
                    "output_tokens": transport.output_tokens,
                }
            )
            return cr

    async def run_async(self) -> tuple[str, list[CaseResult]]:
        cases = self.load_cases()
        if not cases:
            raise RuntimeError("no cases selected for this run")

        run_dir = self.output_dir / "runs" / self.run_id
        store = ArtifactStore(run_dir)
        cache_path = self.output_dir / "cache" / "responses.jsonl"
        cache = ResponseCache(cache_path) if self.use_cache else None
        if cache is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume: skip cases already completed in this run directory (spec 71).
        done = store.completed_case_ids()
        results: list[CaseResult] = []

        manifest = {
            "run_id": self.run_id,
            "created_at": utc_now_iso(),
            "evaluator_version": __version__,
            "git_commit": _git_commit(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "model": self.model,
            "provider": getattr(self.provider, "name", "systemone"),
            "provider_url": self.provider_url,
            "dataset_manifest": str(self.manifest_path),
            "dataset_hash": self.dataset_hash or "",
            "config_hash": "",
            "seed": self.seed,
            # Spec 72: when shuffled, the seed (recorded above) reproduces
            # the order. Record the flag so replay/compare can see it.
            "shuffled": self.shuffle,
            "repeats": self.repeats,
            "cache_enabled": self.use_cache,
            "suite": self.suite,
            "case_count": len(cases) * self.repeats,
        }
        store.write_manifest(manifest)
        # On resume, cases.jsonl already holds records for previously
        # started cases; append only the ones not yet recorded.
        existing_case_ids = {str(crec.get("case_id")) for crec in store.iter_cases()}
        for case in cases:
            if case.case_id not in existing_case_ids:
                store.append_case(case.model_dump())

        sem = asyncio.Semaphore(self.concurrency)
        # fail-fast: after the first error result, no new case is dispatched
        # (in-flight requests still complete so artifacts stay consistent).
        abort = asyncio.Event()

        async def _run_one(case: CanonicalCase, repeat: int) -> CaseResult | None:
            if abort.is_set():
                return None
            cr = await self._execute_one(case, repeat, store, cache, sem)
            if self.fail_fast and cr.status != CaseStatus.SUCCESS:
                abort.set()
            return cr

        tasks = []
        for case in cases:
            for repeat in range(self.repeats):
                if f"{case.case_id}#{repeat}" in done:
                    continue
                tasks.append(_run_one(case, repeat))
        gathered = await asyncio.gather(*tasks)
        results = [r for r in gathered if r is not None]
        # A fail-fast abort means some planned (case, repeat) units were
        # never dispatched; the run is partial and the manifest must not
        # claim otherwise — a resumed run would otherwise skip them.
        aborted = abort.is_set() and len(results) < len(tasks)

        # Merge any previously-completed results for resume. Only records
        # that were already present when this run started (`done`) are
        # reloaded: responses.jsonl now ALSO contains the records this
        # run just appended, and reloading those would double-count every
        # newly executed case (n_cases 240 -> 473 on a 7/240 resume).
        # The reloaded set is intersected with THIS run's case selection:
        # a resume that narrows the run (--capability/--tag/--max-cases)
        # must not drag in cases the current selection excludes
        # (previously a full-run resume with --capability claim_evidence
        # produced n_cases 135 = 25 fresh + all 100 prior units).
        if done:
            selected_ids = {c.case_id for c in cases}
            if selected_ids:
                done = {k for k in done if k.partition("#")[0] in selected_ids}
            if done:
                results = list(self._reload_completed(run_dir, done)) + list(results)

        manifest["completed"] = not aborted
        store.write_manifest(manifest)
        return self.run_id, results

    def _reload_completed(self, run_dir: Path, only_keys: set[str]) -> Iterable[CaseResult]:
        """Rebuild full-fidelity CaseResults for previously-completed cases.

        Each responses.jsonl record is joined with cases.jsonl by case_id
        so dataset/capability/gold survive the resume; error records are
        emitted as error CaseResults so transport/schema error rates (and
        the gates they feed) are preserved across resume. This must equal
        what a fresh non-resumed run of the same cases would produce.
        """
        store = ArtifactStore(run_dir)
        cases_by_id: dict[str, Any] = {}
        for crec in store.iter_cases():
            cid = crec.get("case_id")
            if cid:
                cases_by_id[cid] = crec

        from jev_agent_eval.schemas.response import JEVResponse

        for rec in store.iter_responses():
            raw_key = rec.get("case_id", "")
            if str(raw_key) not in only_keys:
                continue
            case_id, _, repeat = str(raw_key).partition("#")
            case_rec = cases_by_id.get(case_id, {})
            questions = case_rec.get("questions") or {}
            gold_rec = case_rec.get("gold") or {}

            predictions: dict[str, Any] = {}
            response = rec.get("response")
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

            parse_error = rec.get("parse_error")
            network_error = rec.get("network_error_class")
            if response:
                status = CaseStatus.SUCCESS
                missing = [q for q in questions if q not in predictions]
                if missing:
                    status = CaseStatus.SCHEMA_ERROR
            elif parse_error:
                status = CaseStatus.SCHEMA_ERROR
            else:
                status = CaseStatus.TRANSPORT_ERROR

            # Slice attributes (difficulty/source/adversarial) must be
            # restored from the stored case record exactly as the live
            # path derives them, or reloaded cases silently vanish from
            # slice metrics on resume.
            metadata = case_rec.get("metadata") or {}
            tags = case_rec.get("tags") or []
            cr = CaseResult(
                case_id=case_id,
                dataset=case_rec.get("dataset", ""),
                dataset_version=case_rec.get("dataset_version", ""),
                capability=case_rec.get("capability", ""),
                repeat_index=int(rec.get("repeat_index", repeat or 0) or 0),
                status=status,
                request_hash=rec.get("request_hash"),
                question_ids=list(questions.keys()),
                predictions=predictions if response else {},
                is_variant=bool(case_rec.get("is_variant", False)),
                error_detail=(
                    f"missing_answer: {missing}"
                    if response and missing
                    else (
                        parse_error
                        or (
                            f"{network_error or ''} http={rec.get('http_status')}"
                            f"{': ' + str(rec['error_body']) if rec.get('error_body') else ''}"
                        )
                    )
                ),
                transport=TransportRecord(
                    request_start="",
                    latency_ms=rec.get("latency_ms"),
                    http_status=rec.get("http_status"),
                    from_cache=rec.get("from_cache", False),
                    input_tokens=(response or {}).get("usage", {}).get("input_tokens")
                    if isinstance(response, dict)
                    else None,
                    output_tokens=(response or {}).get("usage", {}).get("output_tokens")
                    if isinstance(response, dict)
                    else None,
                ),
                # Same canonical-first preference as the live path (see
                # the note at CaseResult construction there).
                difficulty=case_rec.get("difficulty") or metadata.get("difficulty"),
                adversarial="adversarial" in tags,
                source=case_rec.get("source") or metadata.get("source"),
                tags=tags,
                state_chars=len(case_rec.get("state") or ""),
                # Persisted raw body survives the resume reload so the
                # security report counts it (see replay.py's note).
                raw_response=_parse_raw_response_for_reload(rec.get("raw_response")),
            )
            if response and status == CaseStatus.SUCCESS and questions:
                # Re-attach gold for scoring; matches the live path's
                # object.__setattr__ attachment.
                gold_snapshot = {}
                for qid, g in gold_rec.items():
                    try:
                        from jev_agent_eval.schemas.case import GoldAnswer

                        gold_snapshot[qid] = GoldAnswer.model_validate(g)
                    except Exception:  # noqa: BLE001 - malformed stored gold
                        gold_snapshot[qid] = g
                object.__setattr__(cr, "gold_snapshot", gold_snapshot)
            yield cr

    def run(self) -> tuple[str, list[CaseResult]]:
        return asyncio.run(self.run_async())


def _git_commit() -> str | None:
    try:
        import subprocess

        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None
