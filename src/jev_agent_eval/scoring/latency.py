"""Latency and usage aggregation (spec sections 29, 30)."""

from __future__ import annotations

from dataclasses import dataclass

from jev_agent_eval.utils.timing import percentile


@dataclass
class LatencyStats:
    n: int
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    mean: float | None
    minimum: float | None
    maximum: float | None
    timeout_rate: float = 0.0
    retry_adjusted_mean: float | None = None


def latency_stats(
    latencies_ms: list[float],
    *,
    timeout_ms: float | None = None,
    retry_adjusted_ms: list[float] | None = None,
) -> LatencyStats:
    n = len(latencies_ms)
    if n == 0:
        return LatencyStats(
            n=0, p50=None, p90=None, p95=None, p99=None, mean=None, minimum=None, maximum=None
        )
    timeout_rate = sum(1 for x in latencies_ms if x >= timeout_ms) / n if timeout_ms else 0.0
    return LatencyStats(
        n=n,
        p50=percentile(latencies_ms, 50),
        p90=percentile(latencies_ms, 90),
        p95=percentile(latencies_ms, 95),
        p99=percentile(latencies_ms, 99),
        mean=sum(latencies_ms) / n,
        minimum=min(latencies_ms),
        maximum=max(latencies_ms),
        timeout_rate=timeout_rate,
        retry_adjusted_mean=(sum(retry_adjusted_ms) / len(retry_adjusted_ms))
        if retry_adjusted_ms
        else None,
    )


@dataclass
class UsageStats:
    n_with_usage: int
    input_tokens_total: int
    output_tokens_total: int
    input_tokens_per_case: float | None
    output_tokens_per_case: float | None
    cost_per_1000_decisions: float | None = None  # None unless pricing configured


def usage_stats(
    input_tokens: list[int],
    output_tokens: list[int],
    input_usd_per_million: float | None = None,
    output_usd_per_million: float | None = None,
) -> UsageStats:
    n = len(input_tokens)
    if n == 0:
        return UsageStats(
            n_with_usage=0,
            input_tokens_total=0,
            output_tokens_total=0,
            input_tokens_per_case=None,
            output_tokens_per_case=None,
        )
    in_total = sum(input_tokens)
    out_total = sum(output_tokens)
    cost = None
    if input_usd_per_million is not None and output_usd_per_million is not None:
        # Cost per 1000 decisions based on mean usage per case.
        mean_in = in_total / n
        mean_out = out_total / n
        cost = (
            mean_in / 1e6 * input_usd_per_million + mean_out / 1e6 * output_usd_per_million
        ) * 1000
    return UsageStats(
        n_with_usage=n,
        input_tokens_total=in_total,
        output_tokens_total=out_total,
        input_tokens_per_case=in_total / n,
        output_tokens_per_case=out_total / n,
        cost_per_1000_decisions=cost,
    )
