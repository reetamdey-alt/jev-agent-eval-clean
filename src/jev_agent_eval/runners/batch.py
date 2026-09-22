"""Batch and soak runners (spec sections 29, 29.1).

Batch: repeat one small case set at several concurrency levels.
Soak: N requests at fixed concurrency over minutes, tracking throughput,
error rate, latency drift, 429/5xx frequency, and cache growth.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from jev_agent_eval.client.base import InferenceProvider
from jev_agent_eval.schemas.request import JEVQuestion, JEVRequest


def _probe_request(model: str) -> JEVRequest:
    return JEVRequest(
        state="Soak probe: user asked to list files in the current directory.",
        model=model,
        questions={
            "needs_tool": JEVQuestion(
                type="noul",
                instructions="Does this request require calling a tool?",
            )
        },
    )


@dataclass
class BatchProfileResult:
    concurrency: int
    n: int
    total_s: float
    throughput_rps: float
    error_rate: float
    latencies_ms: list[float] = field(default_factory=list)


async def run_batch_profile(
    provider: InferenceProvider,
    model: str,
    *,
    n_requests: int = 50,
    concurrency: int = 8,
) -> BatchProfileResult:
    request = _probe_request(model)
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    counts = {"errors": 0}

    async def one() -> None:
        async with sem:
            result = await provider.infer_async(request)
            if result.response is None:
                counts["errors"] += 1
            elif result.latency_ms is not None:
                latencies.append(result.latency_ms)

    start = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(n_requests)))
    total_s = time.perf_counter() - start
    return BatchProfileResult(
        concurrency=concurrency,
        n=n_requests,
        total_s=total_s,
        throughput_rps=n_requests / total_s if total_s else 0.0,
        error_rate=counts["errors"] / n_requests if n_requests else 0.0,
        latencies_ms=latencies,
    )


@dataclass
class SoakResult:
    n_requests: int
    concurrency: int
    duration_s: float
    throughput_rps: float
    error_rate: float
    rate_limited: int
    server_errors: int
    latency_drift_ms: float | None  # mean(second half) - mean(first half)
    latencies_ms: list[float] = field(default_factory=list)


async def run_soak(
    provider: InferenceProvider,
    model: str,
    *,
    n_requests: int = 1000,
    concurrency: int = 8,
) -> SoakResult:
    request = _probe_request(model)
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    counts = {"errors": 0, "rate_limited": 0, "server_errors": 0}
    order: list[float] = []  # latencies in completion order, for drift

    async def one() -> None:
        async with sem:
            result = await provider.infer_async(request)
            if result.response is None:
                counts["errors"] += 1
                if result.http_status == 429:
                    counts["rate_limited"] += 1
                elif result.http_status and result.http_status >= 500:
                    counts["server_errors"] += 1
            elif result.latency_ms is not None:
                latencies.append(result.latency_ms)
                order.append(result.latency_ms)

    start = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(n_requests)))
    duration_s = time.perf_counter() - start

    drift = None
    if len(order) >= 4:
        half = len(order) // 2
        first = sum(order[:half]) / half
        second = sum(order[half:]) / (len(order) - half)
        drift = second - first

    return SoakResult(
        n_requests=n_requests,
        concurrency=concurrency,
        duration_s=duration_s,
        throughput_rps=n_requests / duration_s if duration_s else 0.0,
        error_rate=counts["errors"] / n_requests if n_requests else 0.0,
        rate_limited=counts["rate_limited"],
        server_errors=counts["server_errors"],
        latency_drift_ms=drift,
        latencies_ms=latencies,
    )
