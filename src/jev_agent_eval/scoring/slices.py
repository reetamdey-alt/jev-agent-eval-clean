"""Slice analysis (spec section 32).

All primary metrics must be sliceable by capability, difficulty, dataset
source, adversarial flag, input-length bucket, tool family, etc. Slices are
never collapsed into a single overall ranking.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from jev_agent_eval.schemas.result import CaseResult

LENGTH_BUCKETS = [
    ("<1K", 0, 1000),
    ("1-4K", 1000, 4000),
    ("4-16K", 4000, 16000),
    ("16-32K", 16000, 32000),
    ("32-64K", 32000, 64000),
    ("64K+", 64000, float("inf")),
]

# Spec 60 defines the length buckets in TOKENS ("<1K tokens"). The evaluator
# records state size in characters; bucketing on raw chars would mislabel
# every case ~4x high (a 3.5K-char state is <1K tokens). chars/4 is the
# standard approximation when the provider does not report token counts.
_CHARS_PER_TOKEN = 4


def length_bucket(state_chars: int) -> str:
    tokens = state_chars // _CHARS_PER_TOKEN
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= tokens < hi:
            return name
    return "64K+"


# Each dimension: name -> function extracting the slice key from a CaseResult.
DEFAULT_DIMENSIONS: dict[str, Callable[[CaseResult], str | None]] = {
    "capability": lambda r: r.capability,
    "dataset": lambda r: r.dataset,
    "difficulty": lambda r: r.difficulty,
    "source": lambda r: r.source,
    "adversarial": lambda r: "adversarial" if r.adversarial else "normal",
    "input_length_bucket": lambda r: length_bucket(r.state_chars),
}


def build_slices(
    results: list[CaseResult],
    metric_fn: Callable[[list[CaseResult]], dict[str, Any]],
    dimensions: dict[str, Callable[[CaseResult], str | None]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute metric_fn per (dimension, slice-key) over case results."""
    dims = dimensions or DEFAULT_DIMENSIONS
    grouped: dict[str, dict[str, list[CaseResult]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for dim_name, key_fn in dims.items():
            key = key_fn(r)
            if key is not None:
                grouped[dim_name][str(key)].append(r)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dim_name, slices in grouped.items():
        out[dim_name] = {}
        for key, rs in slices.items():
            metrics = metric_fn(rs)
            metrics["n"] = len(rs)
            out[dim_name][key] = metrics
    return out


def worst_slices(
    slices: dict[str, dict[str, dict[str, Any]]],
    metric_key: str = "accuracy",
    lower_is_worse: bool = True,
    min_n: int = 5,
    limit: int = 10,
) -> list[tuple[str, str, float, int]]:
    """Automatically surface the worst-performing slices for investigation."""
    rows: list[tuple[str, str, float, int]] = []
    for dim, entries in slices.items():
        for key, metrics in entries.items():
            if metrics.get("n", 0) < min_n:
                continue
            value = metrics.get(metric_key)
            if not isinstance(value, (int, float)) or value is None:
                continue
            rows.append((dim, key, float(value), int(metrics["n"])))
    rows.sort(key=lambda r: r[2], reverse=not lower_is_worse)
    return rows[:limit]
