"""Categorical scoring: binary/noul and multiclass/choice (spec section 17).

JEV's binary probability maps to p_positive via the question-pack contract:
for `noul`, `noul` field and `probabilities["true"]` both represent
p(positive). This mapping is validated by golden fixtures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BinaryMetrics:
    n: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    pr_auc: float | None
    brier: float | None
    log_loss: float | None
    base_rate: float | None


@dataclass
class MulticlassMetrics:
    n: int
    accuracy: float | None
    macro_f1: float | None
    top2_accuracy: float | None
    log_loss: float | None
    brier: float | None
    per_class: dict[str, dict[str, float | None]]
    confusion: dict[str, dict[str, int]]
    classes: list[str] = field(default_factory=list)


def binary_scores(
    y_true: list[int], y_pred: list[int], p_pred: list[float | None] | None
) -> BinaryMetrics:
    """Compute binary metrics. p_pred entries (or the whole list) may be None."""
    p_pred = p_pred if p_pred is not None else [None] * len(y_true)
    n = len(y_true)
    if n == 0:
        return BinaryMetrics(
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            roc_auc=None,
            pr_auc=None,
            brier=None,
            log_loss=None,
            base_rate=None,
            n=0,
        )
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 0)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    # F1 is 0.0 when tp=0 (both precision and recall are defined and 0);
    # it is None only when precision or recall is genuinely undefined.
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else (0.0 if precision is not None and recall is not None else None)
    )
    base_rate = (tp + fn) / n

    probs = [(t, p) for t, p in zip(y_true, p_pred, strict=False) if p is not None]
    roc_auc = _roc_auc(y_true, p_pred) if probs else None
    pr_auc = _pr_auc(y_true, p_pred) if probs else None
    brier = sum((p - t) ** 2 for t, p in probs) / len(probs) if probs else None
    log_loss = None
    if probs:
        ll = 0.0
        for t, p in probs:
            p = min(max(p, 1e-15), 1 - 1e-15)
            ll += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        log_loss = ll / len(probs)

    return BinaryMetrics(
        n=n,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier=brier,
        log_loss=log_loss,
        base_rate=base_rate,
    )


def _roc_auc(y_true: list[int], p_pred: list[float | None]) -> float | None:
    pairs = sorted(
        ((p, t) for t, p in zip(y_true, p_pred, strict=False) if p is not None),
        key=lambda x: x[0],
    )
    pos = sum(1 for _, t in pairs if t == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    # Rank-based AUC with tie handling.
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank of the tie group
        rank_sum_pos += sum(avg_rank for _, t in pairs[i : j + 1] if t == 1)
        i = j + 1
    return (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def _pr_auc(y_true: list[int], p_pred: list[float | None]) -> float | None:
    pairs = sorted(
        ((p, t) for t, p in zip(y_true, p_pred, strict=False) if p is not None),
        key=lambda x: -x[0],
    )
    pos = sum(1 for _, t in pairs if t == 1)
    if pos == 0:
        return None
    tp = fp = 0
    prev_recall = 0.0
    auc = 0.0
    i = 0
    # Tie-corrected average precision: evaluate precision once at the end
    # of each group of equally-scored cases, so the value does not depend
    # on the order of tied cases (analogous to _roc_auc tie grouping).
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for _, t in pairs[i : j + 1]:
            if t == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / pos
        auc += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return auc


def multiclass_scores(
    y_true: list[str],
    y_pred: list[str],
    prob_vectors: list[dict[str, float] | None],
) -> MulticlassMetrics:
    n = len(y_true)
    # None predictions (missing answers) must never enter the class set:
    # sorted() over str|None raises TypeError and aborts the report build.
    # A None prediction still counts in n (and the accuracy denominator) as
    # an incorrect prediction; it is excluded from per-class tallies.
    classes = sorted({c for c in y_true if c is not None} | {c for c in y_pred if c is not None})
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p)
    # accuracy is None when nothing was scored, never a fabricated 0.0.
    accuracy = correct / n if n else None

    per_class: dict[str, dict[str, float | None]] = {}
    confusion: dict[str, dict[str, int]] = {c: {d: 0 for d in classes} for c in classes}
    for t, p in zip(y_true, y_pred, strict=False):
        if t in confusion and p in confusion[t]:
            confusion[t][p] += 1
    f1s: list[float] = []
    for c in classes:
        tp = confusion.get(c, {}).get(c, 0)
        fp = sum(confusion.get(other, {}).get(c, 0) for other in classes if other != c)
        fn = sum(confusion.get(c, {}).get(other, 0) for other in classes if other != c)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else (0.0 if precision is not None and recall is not None else None)
        )
        # macro-F1 (zero_division=0 semantics): a class the model never gets
        # right contributes 0.0, it is never silently dropped — dropping it
        # would average over only the surviving classes and inflate macro_f1.
        if f1 is not None:
            f1s.append(f1)
        elif precision is not None or recall is not None:
            f1s.append(0.0)
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}
    macro_f1 = sum(f1s) / len(f1s) if f1s else None

    # Top-2 accuracy where class cardinality > 2.
    top2: float | None = None
    if len(classes) > 2:
        hits = 0
        total = 0
        for gold_label, prob_map in zip(y_true, prob_vectors, strict=False):
            if not prob_map:
                continue
            top2_labels = sorted(prob_map, key=lambda k: prob_map[k], reverse=True)[:2]
            hits += 1 if gold_label in top2_labels else 0
            total += 1
        top2 = hits / total if total else None

    # Multiclass log loss / Brier over cases with probability vectors.
    log_loss = brier = None
    vec = [(t, p) for t, p in zip(y_true, prob_vectors, strict=False) if p]
    if vec:
        ll = 0.0
        br = 0.0
        for t, probs in vec:
            p_t = min(max(probs.get(t, 0.0), 1e-15), 1 - 1e-15)
            ll += -math.log(p_t)
            for c in classes:
                target = 1.0 if c == t else 0.0
                br += (probs.get(c, 0.0) - target) ** 2
        log_loss = ll / len(vec)
        brier = br / len(vec)

    return MulticlassMetrics(
        n=n,
        accuracy=accuracy,
        macro_f1=macro_f1,
        top2_accuracy=top2,
        log_loss=log_loss,
        brier=brier,
        per_class=per_class,
        confusion=confusion,
        classes=classes,
    )
