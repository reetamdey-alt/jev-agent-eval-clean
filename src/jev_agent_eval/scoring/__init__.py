"""Scoring aggregation: turns CaseResults into per-capability metrics.

This module owns the mapping from parsed JEV answers to scored predictions,
applying probability validation (spec 34) and error taxonomy (spec 33).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from jev_agent_eval.schemas.result import CaseResult, CaseStatus, FailureCategory
from jev_agent_eval.scoring.calibration import (
    CalibrationReport,
    calibration_report,
    p_positive,
    validate_probability_vector,
)
from jev_agent_eval.scoring.categorical import BinaryMetrics, binary_scores, multiclass_scores
from jev_agent_eval.scoring.latency import latency_stats  # noqa: F401  (re-export)

__all__ = [
    "BinaryMetrics",
    "CalibrationReport",
    "calibration_report",
    "p_positive",
    "score_capability",
    "validate_probability_vector",
]


def _gold_as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return 1
        if s in ("false", "no", "0"):
            return 0
    return None


def score_capability(
    results: list[CaseResult], epsilon: float = 1e-3, ece_bins: int = 10
) -> dict[str, Any]:
    """Compute all metrics for one capability's results.

    Handles noul (binary), choice (multiclass), and score (ordinal) questions
    mixed within the capability, keyed by question id.
    """
    by_question: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(
        lambda: {
            "y_true": [],
            "y_pred": [],
            "p_pred": [],
            "prob_vectors": [],
            "correct": [],
            "confidences": [],
            "score_true": [],
            "score_pred": [],
            "types": [],
        }
    )
    # Per-question count of invalid probability vectors (spec 34).
    q_invalid: dict[str, int] = defaultdict(int)
    failure_counts: dict[str, int] = defaultdict(int)
    latencies: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    statuses: dict[str, int] = defaultdict(int)
    n_invalid_probability = 0
    # Categories are counted AFTER each result is scored, not before:
    # scoring itself appends categories (invalid_probability,
    # annotation_error, missing_answer, ...) to the result, and counting
    # at the top of the iteration would miss everything appended below —
    # the failures taxonomy reported {} while n_invalid_probability > 0.
    counted_categories: set[tuple[int, str]] = set()

    def _count_categories(result: Any) -> None:
        for cat in result.failure_categories:
            key = (id(result), cat.value)
            if key not in counted_categories:
                counted_categories.add(key)
                failure_counts[cat.value] += 1

    for r in results:
        statuses[r.status.value] += 1
        if r.transport:
            if r.transport.latency_ms is not None and not r.transport.from_cache:
                latencies.append(r.transport.latency_ms)
            if r.transport.input_tokens:
                input_tokens.append(r.transport.input_tokens)
            if r.transport.output_tokens:
                output_tokens.append(r.transport.output_tokens)
        if r.status != CaseStatus.SUCCESS:
            _count_categories(r)
            continue
        for qid, pred in r.predictions.items():
            # Predictions must be restricted to the questions the case
            # actually asked: a corrupted or adversarial response answering
            # a fabricated question id would otherwise create phantom
            # metric entries (questions.ghost-question) and distort n.
            if qid not in r.question_ids:
                failure_counts["unexpected_answer"] += 1
                continue
            gold = _find_gold(r, qid)
            qtype = pred.get("type")
            case_correct = None
            if qtype not in ("noul", "choice", "score"):
                # Unknown answer type (spec 33): record it in the error
                # taxonomy, never let it silently vanish — a provider that
                # answers every question with garbage types must not report
                # a clean run with zero scored questions.
                # The type is NOT recorded in slot["types"]: the aggregation
                # pass picks the first observed type to route the question
                # to a scorer, and a garbage type recorded here would route
                # every subsequent valid answer into no scorer at all,
                # silently zeroing accuracy for the whole question.
                failure_counts["unknown_answer_type"] += 1
                if FailureCategory.SCHEMA_ERROR not in r.failure_categories:
                    r.failure_categories.append(FailureCategory.SCHEMA_ERROR)
                continue
            gold_declared = _declared_type(r, qid)
            if gold_declared is not None and qtype != gold_declared:
                # The answer's self-declared type disagrees with the
                # question it answers (e.g. a 'score' answer to a noul
                # question): routing it into the question's scorer would
                # compare against a coerced gold. Categorize, don't guess.
                # (Type excluded from slot["types"] for the same reason as
                # above: a mismatched type must not corrupt routing.)
                failure_counts["answer_type_mismatch"] += 1
                if FailureCategory.SCHEMA_ERROR not in r.failure_categories:
                    r.failure_categories.append(FailureCategory.SCHEMA_ERROR)
                continue
            slot = by_question[(qid, qtype)]
            slot["types"].append(qtype)
            if qtype == "noul":
                gold_int = _gold_as_int(gold)
                if gold_int is None and gold is not None:
                    # Gold exists but cannot be coerced to binary: it must
                    # not silently vanish from n_successful vs n_scored.
                    if FailureCategory.ANNOTATION_ERROR not in r.failure_categories:
                        r.failure_categories.append(FailureCategory.ANNOTATION_ERROR)
                p = p_positive(pred)
                # Spec 34: validate the probability before use. An invalid
                # vector corrupts Brier/NLL and silently vanishes from ECE
                # bins — treat it as no-probability and count it.
                p_invalid = p is not None and not (0.0 <= p <= 1.0)
                if p_invalid:
                    n_invalid_probability += 1
                    q_invalid[qid] += 1
                    if FailureCategory.INVALID_PROBABILITY not in r.failure_categories:
                        r.failure_categories.append(FailureCategory.INVALID_PROBABILITY)
                    p = None
                if gold_int is None:
                    continue
                if p is not None:
                    # Label and probability must describe the SAME
                    # prediction: a raw noul field and probabilities["true"]
                    # that disagree would score two different answers for
                    # one case.
                    pred_int = 1 if p >= 0.5 else 0
                elif pred.get("noul") is not None:
                    pred_int = 1 if float(pred["noul"]) >= 0.5 else 0
                else:
                    # Neither a probability nor a label: unanswerable. Do
                    # not append anything — y_true/y_pred/p_pred must stay
                    # index-aligned (a misaligned zip pairs the wrong gold
                    # with the wrong prediction and corrupts every metric).
                    if FailureCategory.MISSING_ANSWER not in r.failure_categories:
                        r.failure_categories.append(FailureCategory.MISSING_ANSWER)
                    continue
                slot["y_true"].append(gold_int)
                slot["y_pred"].append(pred_int)
                slot["p_pred"].append(p)
                case_correct = pred_int == gold_int
                slot["correct"].append(case_correct)
                # Calibration confidence is the probability assigned to
                # the PREDICTED class, not p_positive: a confident
                # negative (p_positive=0.0, gold=0, correct) must land
                # in the high-confidence bin, not the 0.0 bin. Appended
                # in lockstep with `correct` so calibration pairs can
                # never mix one case's confidence with another's outcome.
                slot["confidences"].append(
                    (p if pred_int == 1 else 1.0 - p) if p is not None else None
                )
                r.correct[qid] = case_correct
            elif qtype == "choice":
                if not isinstance(gold, str):
                    if gold is not None:
                        if FailureCategory.ANNOTATION_ERROR not in r.failure_categories:
                            r.failure_categories.append(FailureCategory.ANNOTATION_ERROR)
                    continue
                # A schema-valid choice answer may still carry a null label
                # (JEVAnswer.choice is optional). Score it as an incorrect
                # prediction (counts in n and the accuracy denominator);
                # multiclass_scores excludes None from the class set so
                # sorted() never sees it.
                probs = pred.get("probabilities")
                valid_probs = None
                if isinstance(probs, dict):
                    if validate_probability_vector(probs, epsilon):
                        valid_probs = probs
                    else:
                        n_invalid_probability += 1
                        q_invalid[qid] += 1
                slot["y_true"].append(gold)
                slot["y_pred"].append(pred.get("choice"))
                slot["prob_vectors"].append(valid_probs)
                case_correct = pred.get("choice") == gold
                slot["correct"].append(case_correct)
                # Confidence and correctness are recorded as aligned pairs:
                # a case without a validated vector contributes no
                # calibration point (None placeholder keeps indices aligned).
                slot["confidences"].append(max(valid_probs.values()) if valid_probs else None)
                r.correct[qid] = case_correct
            elif qtype == "score":
                if not isinstance(gold, (int, float)):
                    if gold is not None:
                        if FailureCategory.ANNOTATION_ERROR not in r.failure_categories:
                            r.failure_categories.append(FailureCategory.ANNOTATION_ERROR)
                    continue
                if pred.get("score") is not None:
                    slot["score_true"].append(float(gold))
                    slot["score_pred"].append(float(pred["score"]))
                    case_correct = abs(float(pred["score"]) - float(gold)) < 0.5
                    slot["correct"].append(case_correct)
                    r.correct[qid] = case_correct
        # Count this result's categories now that scoring has appended
        # everything it will (see the comment at _count_categories).
        _count_categories(r)

    question_metrics: dict[str, Any] = {}
    all_correct: list[bool] = []
    calibration_pairs: list[tuple[float, bool]] = []
    # The same question id can appear with different answer types across a
    # capability's cases (e.g. "claim_supported" as noul in the synthetic
    # claim family and as choice in the contradiction family). Slots are
    # keyed (qid, type); a single-typed question keeps its bare id, a
    # mixed one gets "qid:type" entries so neither family's golds pollute
    # the other's scorer (previously: mixed str/int labels crashed
    # multiclass_scores' sorted() at report time).
    qid_type_counts: Counter = Counter(qid for qid, _t in by_question)
    for (qid, qtype), slot in by_question.items():
        entry_name = qid if qid_type_counts[qid] == 1 else f"{qid}:{qtype}"
        entry: dict[str, Any] = {"type": qtype, "n": len(slot["y_true"]) or len(slot["score_true"])}
        if qtype == "noul":
            bm = binary_scores(slot["y_true"], slot["y_pred"], slot["p_pred"])
            entry.update(_binary_to_dict(bm))
            entry["n_invalid_probability"] = q_invalid[qid]
            all_correct.extend(slot["correct"])
        elif qtype == "choice":
            mm = multiclass_scores(slot["y_true"], slot["y_pred"], slot["prob_vectors"])
            entry.update(
                accuracy=mm.accuracy,
                macro_f1=mm.macro_f1,
                top2_accuracy=mm.top2_accuracy,
                log_loss=mm.log_loss,
                brier=mm.brier,
                per_class=mm.per_class,
                confusion=mm.confusion,
                n=mm.n,
            )
            entry["n_invalid_probability"] = q_invalid[qid]
            all_correct.extend(slot["correct"])
        elif qtype == "score":
            from jev_agent_eval.scoring.ordinal import score_scores

            sm = score_scores(slot["score_true"], slot["score_pred"])
            entry.update(
                mae=sm.mae,
                rmse=sm.rmse,
                spearman=sm.spearman,
                mean_signed_error=sm.mean_signed_error,
                within_one_level=sm.within_one_level,
                n=sm.n,
            )
            all_correct.extend(slot["correct"])
        # Calibration pairs are collected as atomic (confidence, correct)
        # tuples so a desynced pair of lists can never pair one case's
        # confidence with another's outcome. Cases with no validated
        # probability contribute no calibration point.
        if qtype in ("noul", "choice"):
            calibration_pairs.extend(
                (c, ok)
                for c, ok in zip(slot["confidences"], slot["correct"], strict=True)
                if c is not None
            )
        question_metrics[entry_name] = entry

    calibration = (
        calibration_report(
            [c for c, _ in calibration_pairs],
            [ok for _, ok in calibration_pairs],
            bins=ece_bins,
        )
        if calibration_pairs
        else CalibrationReport()
    )

    n_cases = len(results)
    n_success = statuses.get("success", 0)
    accuracy = (sum(1 for c in all_correct if c) / len(all_correct)) if all_correct else None

    return {
        "n_cases": n_cases,
        "n_successful": n_success,
        "n_scored_questions": len(all_correct),
        "accuracy": accuracy,
        "questions": question_metrics,
        "calibration": {
            "ece": calibration.ece,
            "adaptive_ece": calibration.adaptive_ece,
            "brier": calibration.brier,
            "nll": calibration.nll,
            "n": calibration.n,
            "n_invalid_probability": n_invalid_probability,
            "reliability": [b.__dict__ for b in calibration.reliability],
        },
        "failures": dict(failure_counts),
        "statuses": dict(statuses),
        "latency_ms_p50": None,  # filled by caller with latency_stats
        "_latencies": latencies,
        "_input_tokens": input_tokens,
        "_output_tokens": output_tokens,
    }


def _binary_to_dict(bm: BinaryMetrics) -> dict[str, Any]:
    return {
        "n": bm.n,
        "accuracy": bm.accuracy,
        "precision": bm.precision,
        "recall": bm.recall,
        "f1": bm.f1,
        "roc_auc": bm.roc_auc,
        "pr_auc": bm.pr_auc,
        "brier": bm.brier,
        "log_loss": bm.log_loss,
        "base_rate": bm.base_rate,
    }


def _declared_type(r: CaseResult, qid: str) -> str | None:
    """The question's declared type from the gold snapshot, if present."""
    gold_snapshot = getattr(r, "gold_snapshot", None)
    if isinstance(gold_snapshot, dict):
        answer = gold_snapshot.get(qid)
        if answer is not None:
            return getattr(answer, "type", None)
    return None


def _find_gold(r: CaseResult, qid: str) -> Any:
    # Gold values ride along in the runner's scored payload; CaseResult
    # predictions are compared against gold attached by the runner via
    # `gold_snapshot` on the result (kept out of the JEV input).
    gold_snapshot = getattr(r, "gold_snapshot", None)
    if isinstance(gold_snapshot, dict):
        answer = gold_snapshot.get(qid)
        if answer is not None:
            return getattr(answer, "value", answer)
    return None


def classify_failure(r: CaseResult, gold: Any, prediction: Any) -> FailureCategory | None:
    """Assign a failure category to a failed case (spec 33)."""
    if r.status == CaseStatus.TRANSPORT_ERROR:
        if r.transport and r.transport.http_status == 429:
            return FailureCategory.RATE_LIMITED
        if r.transport and r.transport.http_status and r.transport.http_status >= 500:
            return FailureCategory.PROVIDER_5XX
        return FailureCategory.TRANSPORT_ERROR
    if r.status == CaseStatus.SCHEMA_ERROR:
        if r.error_detail and "invalid_probabilit" in str(r.error_detail):
            return FailureCategory.INVALID_PROBABILITY
        if r.error_detail and "missing_answer" in str(r.error_detail):
            return FailureCategory.MISSING_ANSWER
        return FailureCategory.SCHEMA_ERROR
    if prediction is None:
        return FailureCategory.MISSING_ANSWER
    return None
