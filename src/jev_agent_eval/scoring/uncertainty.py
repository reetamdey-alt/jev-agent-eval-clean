"""Bootstrap confidence intervals (spec sections 17, 31)."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class CI:
    low: float
    high: float
    level: float
    samples: int
    method: str = "bootstrap"


def bootstrap_proportion_ci(
    values: list[bool],
    *,
    samples: int = 2000,
    level: float = 0.95,
    seed: int | None = None,
) -> CI | None:
    """Bootstrap CI for the mean of 0/1 values."""
    n = len(values)
    if n == 0:
        return None
    rng = random.Random(seed)
    bools = list(values)
    means: list[float] = []
    for _ in range(samples):
        sample = [bools[rng.randrange(n)] for _ in range(n)]
        means.append(sum(1 for v in sample if v) / n)
    means.sort()
    alpha = (1 - level) / 2
    lo = means[int(alpha * samples)]
    hi = means[min(samples - 1, int((1 - alpha) * samples))]
    return CI(low=lo, high=hi, level=level, samples=samples)


def cluster_bootstrap_proportion_ci(
    clusters: list[list[bool]],
    *,
    samples: int = 2000,
    level: float = 0.95,
    seed: int | None = None,
) -> CI | None:
    """Cluster bootstrap CI for the mean of 0/1 values.

    `clusters[i]` is the list of repeat observations for independent unit i
    (a unique case-question pair). Units are resampled with replacement and
    every repeat within a resampled unit is carried along. The point
    estimate is the per-row mean (sum of all values / count of all values),
    so the CI is centered on the reported accuracy: repeats add information
    about stability but must not shrink the CI (pseudo-replication).
    """
    n = len(clusters)
    rows = sum(len(c) for c in clusters)
    if n == 0 or rows == 0:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        s = 0
        cnt = 0
        for _ in range(n):
            c = clusters[rng.randrange(n)]
            s += sum(c)
            cnt += len(c)
        means.append(s / cnt)
    means.sort()
    alpha = (1 - level) / 2
    lo = means[int(alpha * samples)]
    hi = means[min(samples - 1, int((1 - alpha) * samples))]
    return CI(low=lo, high=hi, level=level, samples=samples, method="cluster_bootstrap")


def paired_cluster_bootstrap_delta_ci(
    clusters_a: list[list[bool]],
    clusters_b: list[list[bool]],
    *,
    samples: int = 2000,
    level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, CI] | None:
    """Paired cluster bootstrap for the delta (b - a) between two aligned runs.

    `clusters_a[i]` and `clusters_b[i]` are the aligned per-unit repeat
    observations of runs A and B (same independent unit — a unique
    (case, question) pair present in both runs). Units are resampled
    together, preserving the pairing; the delta of each resample is the
    difference of per-row means. Positive delta means run B improved — the
    same convention as accuracy_delta in run comparison reports.
    """
    n = len(clusters_a)
    if n == 0 or len(clusters_b) != n:
        return None
    tot_a = sum(sum(c) for c in clusters_a)
    cnt_a = sum(len(c) for c in clusters_a)
    tot_b = sum(sum(c) for c in clusters_b)
    cnt_b = sum(len(c) for c in clusters_b)
    if cnt_a == 0 or cnt_b == 0:
        return None
    point = tot_b / cnt_b - tot_a / cnt_a
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        sa = sb = 0
        na = nb = 0
        for _ in range(n):
            i = rng.randrange(n)
            ca, cb = clusters_a[i], clusters_b[i]
            sa += sum(ca)
            na += len(ca)
            sb += sum(cb)
            nb += len(cb)
        deltas.append(sb / nb - sa / na)
    deltas.sort()
    alpha = (1 - level) / 2
    lo = deltas[int(alpha * samples)]
    hi = deltas[min(samples - 1, int((1 - alpha) * samples))]
    return point, CI(low=lo, high=hi, level=level, samples=samples, method="cluster_bootstrap")


def paired_bootstrap_delta_ci(
    values_a: list[bool],
    values_b: list[bool],
    *,
    samples: int = 2000,
    level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, CI] | None:
    """Paired bootstrap for the delta (b - a) between two aligned runs.

    Positive delta means run B improved — the same convention as
    accuracy_delta in run comparison reports.
    """
    n = len(values_a)
    if n == 0 or len(values_b) != n:
        return None
    point = (sum(1 for v in values_b if v) - sum(1 for v in values_a if v)) / n
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(1 for i in idx if values_a[i]) / n
        db = sum(1 for i in idx if values_b[i]) / n
        deltas.append(db - da)
    deltas.sort()
    alpha = (1 - level) / 2
    lo = deltas[int(alpha * samples)]
    hi = deltas[min(samples - 1, int((1 - alpha) * samples))]
    return point, CI(low=lo, high=hi, level=level, samples=samples)


def wilson_interval(successes: int, n: int, level: float = 0.95) -> CI | None:
    """Wilson score interval for a binary rate."""
    if n == 0:
        return None
    import math
    from statistics import NormalDist

    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    # z-score derived from the requested level, not a hardcoded 95%/99%
    # pair — any other level silently got the 99% interval.
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return CI(
        low=max(0.0, center - margin),
        high=min(1.0, center + margin),
        level=level,
        samples=n,
        method="wilson",
    )
