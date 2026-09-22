"""Latency, throughput, concurrency, rate-limit, and soak evaluation
(spec section 22).

Separate from the correctness benchmark: these runs measure the provider
as a system (transport, retry, queueing behavior), not answer quality.
The primary API latency metric EXCLUDES retry/backoff sleep (22.1).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from jev_agent_eval.client.systemone import InferenceResult, SystemOneProvider
from jev_agent_eval.utils.timing import percentile


@dataclass
class AttemptStats:
    n: int = 0
    connection_ms: list[float] = field(default_factory=list)
    ttfb_ms: list[float] = field(default_factory=list)  # time to first byte when available
    attempt_latency_ms: list[float] = field(default_factory=list)  # successful HTTP attempt only
    e2e_latency_ms: list[float] = field(default_factory=list)  # full request incl. retries
    retry_sleep_ms: list[float] = field(default_factory=list)
    statuses: dict[int, int] = field(default_factory=dict)  # http status -> count
    timeouts: int = 0
    network_errors: int = 0
    retries: int = 0

    def percentile_summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "connection_p50_ms": percentile(self.connection_ms, 50) if self.connection_ms else None,
            "ttfb_p50_ms": percentile(self.ttfb_ms, 50) if self.ttfb_ms else None,
            # Primary API latency: successful attempt only, retry-free.
            "attempt_latency_p50_ms": percentile(self.attempt_latency_ms, 50)
            if self.attempt_latency_ms
            else None,
            "attempt_latency_p90_ms": percentile(self.attempt_latency_ms, 90)
            if self.attempt_latency_ms
            else None,
            "attempt_latency_p95_ms": percentile(self.attempt_latency_ms, 95)
            if self.attempt_latency_ms
            else None,
            "attempt_latency_p99_ms": percentile(self.attempt_latency_ms, 99)
            if self.attempt_latency_ms
            else None,
            "e2e_latency_p50_ms": percentile(self.e2e_latency_ms, 50)
            if self.e2e_latency_ms
            else None,
            "e2e_latency_p90_ms": percentile(self.e2e_latency_ms, 90)
            if self.e2e_latency_ms
            else None,
            "e2e_latency_p95_ms": percentile(self.e2e_latency_ms, 95)
            if self.e2e_latency_ms
            else None,
            "e2e_latency_p99_ms": percentile(self.e2e_latency_ms, 99)
            if self.e2e_latency_ms
            else None,
            "e2e_latency_p99_9_ms": percentile(self.e2e_latency_ms, 99.9)
            if self.e2e_latency_ms
            else None,
            "retry_sleep_mean_ms": (sum(self.retry_sleep_ms) / len(self.retry_sleep_ms))
            if self.retry_sleep_ms
            else None,
            "attempt_latency_mean_ms": (sum(self.attempt_latency_ms) / len(self.attempt_latency_ms))
            if self.attempt_latency_ms
            else None,
            "e2e_latency_mean_ms": (sum(self.e2e_latency_ms) / len(self.e2e_latency_ms))
            if self.e2e_latency_ms
            else None,
            "statuses": dict(self.statuses),
            "timeout_rate": self.timeouts / self.n if self.n else None,
            "network_error_rate": self.network_errors / self.n if self.n else None,
            "retry_rate": self.retries / self.n if self.n else None,
            "requests_with_retry": self.retries,
        }


def _probe_request() -> dict[str, Any]:
    """Minimal lightweight request for performance probing (22.4: '10k+
    lightweight requests where practical')."""
    return {
        "state": "performance probe: user asks to list files. no action needed.",
        "questions": {
            "perf_probe": {
                "type": "noul",
                "instructions": "Is the proposed action high risk?",
            }
        },
    }


async def _timed_probe(provider: SystemOneProvider) -> InferenceResult:
    from jev_agent_eval.schemas.request import JEVQuestion, JEVRequest

    request = JEVRequest(
        state="performance probe: user asks to list files. no action needed.",
        model=getattr(provider, "model", "jev-latest"),
        questions={
            "perf_probe": JEVQuestion(
                type="noul",
                instructions="Is the proposed action high risk?",
            )
        },
    )
    return await provider.infer_async(request)


class PerformanceRunner:
    """Concurrency sweep (22.2), rate-limit sweep (22.3), soak (22.4)."""

    def __init__(
        self,
        provider: SystemOneProvider,
        *,
        levels: tuple[int, ...] = (1, 2, 4, 8),
        requests_per_level: int = 20,
    ):
        self.provider = provider
        self.levels = levels
        self.requests_per_level = requests_per_level

    async def concurrency_sweep(self) -> dict[str, Any]:
        """Run separate profiles at each concurrency level (22.2).

        Only levels permitted by provider limits should be passed in.
        """
        results: dict[str, Any] = {}
        for level in self.levels:
            stats = await self._run_batch(concurrency=level, n=self.requests_per_level)
            results[f"concurrency_{level}"] = stats.percentile_summary()
            results[f"concurrency_{level}"]["configured_concurrency"] = level
            # Tail amplification vs the single-worker baseline (22.3).
            if "concurrency_1" in results:
                base_p95 = results["concurrency_1"].get("attempt_latency_p95_ms")
                cur_p95 = stats.percentile_summary().get("attempt_latency_p95_ms")
                if base_p95 and cur_p95:
                    results[f"concurrency_{level}"]["p95_amplification"] = round(
                        cur_p95 / base_p95, 3
                    )
        return results

    async def rate_limit_sweep(self, rates: list[float]) -> dict[str, Any]:
        """Measure behavior near configured request-rate limits (22.3)."""
        results: dict[str, Any] = {}
        for rate in rates:
            # Apply the rate to the provider's limiter for this batch.
            original_rate = getattr(self.provider, "requests_per_second", None)
            self.provider.requests_per_second = rate  # type: ignore[attr-defined]
            try:
                stats = await self._run_batch(concurrency=4, n=max(10, int(rate) * 5))
            finally:
                self.provider.requests_per_second = original_rate  # type: ignore[attr-defined]
            summary = stats.percentile_summary()
            summary["configured_rps"] = rate
            status_429 = stats.statuses.get(429, 0)
            status_5xx = sum(c for s, c in stats.statuses.items() if 500 <= s < 600)
            summary["rate_429"] = status_429
            summary["rate_5xx"] = status_5xx
            summary["retry_success_rate"] = (
                (stats.retries - stats.network_errors - stats.timeouts) / stats.retries
                if stats.retries
                else None
            )
            results[f"rate_{rate}"] = summary
        return results

    async def _run_batch(self, *, concurrency: int, n: int) -> AttemptStats:
        stats = AttemptStats()
        semaphore = asyncio.Semaphore(concurrency)

        async def one() -> None:
            async with semaphore:
                start = time.perf_counter()
                result = await _timed_probe(self.provider)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                stats.n += 1
                stats.e2e_latency_ms.append(elapsed_ms)
                # Attempt-level latency excludes retry sleep; InferenceResult
                # carries the timing split when the provider records it.
                attempt_ms = getattr(result, "latency_ms", None)
                stats.attempt_latency_ms.append(
                    attempt_ms if attempt_ms is not None else elapsed_ms
                )
                retry_count = getattr(result, "retry_count", 0)
                if retry_count:
                    stats.retries += 1
                    # Retry sleep time: total e2e minus the successful
                    # attempt latency approximates the backoff spent.
                    stats.retry_sleep_ms.append(max(0.0, elapsed_ms - (attempt_ms or elapsed_ms)))
                status = getattr(result, "http_status", None)
                if status is not None:
                    stats.statuses[status] = stats.statuses.get(status, 0) + 1
                if result.network_error_class:
                    stats.network_errors += 1
                if getattr(result, "network_error_class", None) in (
                    "ReadTimeout",
                    "ConnectTimeout",
                    "WriteTimeout",
                    "PoolTimeout",
                ):
                    stats.timeouts += 1

        await asyncio.gather(*(one() for _ in range(n)))
        return stats


@dataclass
class SoakSample:
    t_seconds: float
    completed: int
    p50_ms: float | None
    p95_ms: float | None
    rss_mb: float | None
    fd_count: int | None
    rate_429: int
    rate_5xx: int
    resets: int


def _process_rss_mb() -> float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def _fd_count() -> int | None:
    try:
        import os

        return len(os.listdir("/proc/self/fd"))
    except Exception:
        try:
            import subprocess

            out = subprocess.run(
                ["lsof", "-p", str(os.getpid())], capture_output=True, text=True, timeout=5
            )
            return len(out.stdout.splitlines()) - 1 if out.returncode == 0 else None
        except Exception:
            return None


class SoakRunner:
    """Sustained-load run tracking drift (22.4): throughput, latency,
    memory, file descriptors, 429/5xx frequency, connection resets."""

    def __init__(
        self,
        provider: SystemOneProvider,
        *,
        duration_s: float = 1800.0,
        workers: int = 4,
        sample_interval_s: float = 30.0,
        report_every: int = 100,
    ):
        self.provider = provider
        self.duration_s = duration_s
        self.workers = workers
        self.sample_interval_s = sample_interval_s
        self.report_every = report_every

    async def run(self) -> dict[str, Any]:
        start = time.perf_counter()
        latencies: list[float] = []
        statuses: dict[int, int] = {}
        resets = 0
        errors_429 = 0
        errors_5xx = 0
        timeouts = 0
        completed = 0
        samples: list[dict[str, Any]] = []
        stop = asyncio.Event()

        async def worker() -> None:
            nonlocal completed, resets, errors_429, errors_5xx, timeouts
            while not stop.is_set():
                t0 = time.perf_counter()
                result = await _timed_probe(self.provider)
                elapsed = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed)
                completed += 1
                status = getattr(result, "http_status", None)
                if status:
                    statuses[status] = statuses.get(status, 0) + 1
                    if status == 429:
                        errors_429 += 1
                    elif 500 <= status < 600:
                        errors_5xx += 1
                if result.network_error_class == "connection_reset":
                    resets += 1
                if result.network_error_class in (
                    "ReadTimeout",
                    "ConnectTimeout",
                    "WriteTimeout",
                    "PoolTimeout",
                    "read_timeout",
                ):
                    timeouts += 1

        async def sampler() -> None:
            last_completed = 0
            last_t = time.perf_counter()
            while not stop.is_set():
                await asyncio.sleep(self.sample_interval_s)
                now = time.perf_counter()
                # Latencies of the trailing window for drift tracking.
                recent = latencies[-min(200, len(latencies)) :]
                samples.append(
                    {
                        "t_seconds": round(now - start, 1),
                        "completed": completed,
                        "throughput_per_s": round(
                            (completed - last_completed) / max(1e-9, now - last_t), 3
                        ),
                        "p50_ms": percentile(recent, 50) if recent else None,
                        "p95_ms": percentile(recent, 95) if recent else None,
                        "p99_9_ms": percentile(recent, 99.9) if recent else None,
                        "rss_mb": _process_rss_mb(),
                        "fd_count": _fd_count(),
                        "rate_429": errors_429,
                        "rate_5xx": errors_5xx,
                        "resets": resets,
                        "timeouts": timeouts,
                    }
                )
                last_completed = completed
                last_t = now

        workers = [asyncio.create_task(worker()) for _ in range(self.workers)]
        sampler_task = asyncio.create_task(sampler())
        try:
            await asyncio.sleep(self.duration_s)
        finally:
            stop.set()
            await asyncio.gather(*workers, return_exceptions=True)
            sampler_task.cancel()
            try:
                await sampler_task
            except asyncio.CancelledError:
                pass

        # Drift: compare first-third vs last-third throughput and p95.
        first = [s for s in samples if s["t_seconds"] <= self.duration_s / 3] if samples else []
        last = [s for s in samples if s["t_seconds"] > 2 * self.duration_s / 3] if samples else []
        drift: dict[str, Any] = {}
        if first and last:
            t_first = sum(s["throughput_per_s"] for s in first) / len(first)
            t_last = sum(s["throughput_per_s"] for s in last) / len(last)
            p_first = [s["p95_ms"] for s in first if s["p95_ms"]]
            p_last = [s["p95_ms"] for s in last if s["p95_ms"]]
            drift = {
                "throughput_first_third": round(t_first, 3),
                "throughput_last_third": round(t_last, 3),
                "throughput_drift_pct": (
                    round((t_last - t_first) / t_first * 100.0, 2) if t_first else None
                ),
                "p95_first_third_ms": (sum(p_first) / len(p_first)) if p_first else None,
                "p95_last_third_ms": (sum(p_last) / len(p_last)) if p_last else None,
                "latency_drift_pct": (
                    round(
                        ((sum(p_last) / len(p_last)) - (sum(p_first) / len(p_first)))
                        / (sum(p_first) / len(p_first))
                        * 100.0,
                        2,
                    )
                    if p_first and p_last
                    else None
                ),
            }
        rss_values = [s["rss_mb"] for s in samples if s["rss_mb"]]
        fd_values = [s["fd_count"] for s in samples if s["fd_count"] is not None]
        return {
            "duration_s": self.duration_s,
            "workers": self.workers,
            "completed_requests": completed,
            "p99_9_ms": percentile(latencies, 99.9) if latencies else None,
            "p50_ms": percentile(latencies, 50) if latencies else None,
            "p95_ms": percentile(latencies, 95) if latencies else None,
            "statuses": statuses,
            "rate_429_total": errors_429,
            "rate_5xx_total": errors_5xx,
            "connection_resets": resets,
            "timeouts": timeouts,
            "memory_growth_mb": (
                round(max(rss_values) - min(rss_values), 2) if len(rss_values) > 1 else None
            ),
            "fd_growth": (max(fd_values) - min(fd_values)) if len(fd_values) > 1 else None,
            "drift": drift,
            "samples": samples,
        }


def run_performance_suite(
    provider: SystemOneProvider,
    *,
    levels: tuple[int, ...] = (1, 2, 4, 8),
    requests_per_level: int = 20,
    rates: list[float] | None = None,
) -> dict[str, Any]:
    """Synchronous entry point for the performance suite (22.2 + 22.3)."""

    async def _run() -> dict[str, Any]:
        runner = PerformanceRunner(provider, levels=levels, requests_per_level=requests_per_level)
        out: dict[str, Any] = {"concurrency_sweep": await runner.concurrency_sweep()}
        if rates:
            out["rate_limit_sweep"] = await runner.rate_limit_sweep(rates)
        try:
            await provider.aclose()
        except Exception:
            pass
        return out

    return asyncio.run(_run())


def run_soak_suite(
    provider: SystemOneProvider,
    *,
    duration_s: float = 1800.0,
    workers: int = 4,
    sample_interval_s: float = 30.0,
) -> dict[str, Any]:
    """Synchronous entry point for the soak suite (22.4)."""

    async def _run() -> dict[str, Any]:
        runner = SoakRunner(
            provider,
            duration_s=duration_s,
            workers=workers,
            sample_interval_s=sample_interval_s,
        )
        out = await runner.run()
        try:
            await provider.aclose()
        except Exception:
            pass
        return out

    return asyncio.run(_run())
