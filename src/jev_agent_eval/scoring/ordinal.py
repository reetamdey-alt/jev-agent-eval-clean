"""Ordinal / score question scoring (spec section 17.3).

The question pack defines the semantic scale; we do not assume 0-1 or 0-2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ScoreMetrics:
    n: int
    mae: float | None
    rmse: float | None
    spearman: float | None
    mean_signed_error: float | None
    within_one_level: float | None


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def score_scores(y_true: list[float], y_pred: list[float]) -> ScoreMetrics:
    n = len(y_true)
    if n == 0:
        return ScoreMetrics(
            n=0, mae=None, rmse=None, spearman=None, mean_signed_error=None, within_one_level=None
        )
    errors = [p - t for t, p in zip(y_true, y_pred, strict=False)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mse_signed = sum(errors) / n
    # Percentage within one ordinal level. The scale is pack-defined, so
    # the level spacing is derived from the observed distinct gold levels;
    # with fewer than two distinct levels it is undefined (None), not a
    # silent assumption of 1.0.
    distinct = sorted(set(y_true))
    if len(distinct) >= 2:
        spacing = min(b - a for a, b in zip(distinct, distinct[1:], strict=False))
        within_one = sum(1 for e in errors if abs(e) <= spacing + 1e-9) / n
    else:
        within_one = None
    return ScoreMetrics(
        n=n,
        mae=mae,
        rmse=rmse,
        spearman=_spearman(y_true, y_pred),
        mean_signed_error=mse_signed,
        within_one_level=within_one,
    )
