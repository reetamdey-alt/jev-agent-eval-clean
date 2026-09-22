"""Local JSONL dataset loader with validation and leakage checks (spec 67)."""

from __future__ import annotations

import re
from pathlib import Path

from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.utils.jsonl import iter_jsonl

# Keys that must never appear inside the JEV input (spec section 45).
FORBIDDEN_STATE_KEYS = frozenset(
    {
        "answer",
        "gold",
        "label",
        "patch",
        "test_patch",
        "reference_answer",
        "adjudication",
        "gold_patch",
        "expected",
    }
)

# High-signal keys whose BARE "key:" form is almost always leakage when it
# appears in a JEV-bound string. Generic words like "expected:", "answer:"
# occur naturally in prose (bug reports: "Expected: HTTP 200") and are only
# flagged in their quoted/structured form (spec 8.1 documented exception:
# genuine user-visible state text).
HIGH_SIGNAL_KEYS = frozenset(
    {
        "gold",
        "test_patch",
        "reference_answer",
        "adjudication",
        "gold_patch",
        "benchmark_solution",
        "hidden_test",
    }
)

# Spec 60 defines input-length buckets up to "64K+" TOKENS, so states of
# 64K+ tokens (~256K+ chars) must be representable and testable — the cap
# only guards against unbounded memory, it must not make the top bucket
# unreachable.
MAX_STATE_CHARS = 400_000


class LeakageError(ValueError):
    """Raised when a case would leak the gold label into the JEV input."""


def validate_case(
    case: CanonicalCase,
    *,
    max_state_chars: int = MAX_STATE_CHARS,
    strict_secrets: bool = True,
) -> list[str]:
    """Return a list of validation problems (empty = valid).

    Checks the invariants from spec sections 9.1 and 67, including the gold
    leakage validator which inspects every string that reaches the JEV
    input — the state AND each question's instructions/criteria — for
    forbidden field markers in any common quoting/escaping.
    """
    problems: list[str] = []
    if not case.case_id:
        problems.append("case_id is empty")
    if not case.state:
        problems.append("state is empty")
    if len(case.state) > max_state_chars:
        problems.append(f"state exceeds {max_state_chars} chars")
    for qid in case.gold:
        if qid not in case.questions:
            problems.append(f"gold for unknown question {qid!r}")
    for qid, q in case.questions.items():
        if qid not in case.gold:
            problems.append(f"question {qid!r} has no gold answer")
        if isinstance(q, dict):
            if q.get("type") not in ("noul", "choice", "score"):
                problems.append(f"question {qid!r} has unsupported type {q.get('type')!r}")
            if not (q.get("instructions") or "").strip():
                problems.append(f"question {qid!r} has empty instructions")

    # Gold leakage: every JEV-input-bound string is scanned, with common
    # quoting/escaping normalized so a marker cannot hide behind \" or \\
    # encodings. Bare `gold:`/`answer:`/`expected:` key patterns are
    # checked too, not only quoted markers.
    jev_bound_strings = [case.state]
    for q in case.questions.values():
        if isinstance(q, dict):
            if isinstance(q.get("instructions"), str):
                jev_bound_strings.append(q["instructions"])
            criteria = q.get("criteria")
            if isinstance(criteria, dict):
                jev_bound_strings.extend(str(v) for v in criteria.values())
            elif isinstance(criteria, list):
                jev_bound_strings.extend(str(v) for v in criteria)
    for text in jev_bound_strings:
        normalized = (
            text.replace('\\"', '"').replace("\\\\", "\\").replace("'", '"').lower()
        )
        for key in FORBIDDEN_STATE_KEYS:
            quoted = f'"{key}"'
            bare = f"{key}:"
            # Quoted marker is flagged only when it is a serialized mapping
            # key ('"gold": 1' in JSON/YAML embedded in state) — a quoted
            # word appearing in code (e.g. r['gold'] subscript access on a
            # CSV column in real trace content) or prose is legitimate
            # state text (spec 8.1 documented exception).
            if re.search(re.escape(quoted) + r"\s*:", normalized):
                problems.append(f"JEV-input text contains forbidden field marker {quoted}")
                break
            # Bare "key:" is flagged for high-signal keys only; prose words
            # like "expected:" in real bug reports are legitimate state
            # (spec 8.1 documented exception).
            if bare in normalized and key in HIGH_SIGNAL_KEYS:
                problems.append(f"JEV-input text contains forbidden field pattern {bare}")
                break

    # Gold VALUE leakage: a state that embeds the answer itself evades the
    # field-marker scan above ("the correct claim_supported answer is 0"
    # contains no forbidden key). Detect a question's id and its gold
    # value co-occurring in one JEV-bound string — the realistic leak
    # vector for benchmark contamination, and the co-occurrence is what
    # carries the signal: a bare 0/1 in prose means nothing, but
    # "claim_supported = 0" next to a claim_supported question is the
    # answer. Only very short ALPHANUMERIC word values are skipped when
    # they are also one of the question's allowed choices (a choice
    # question legitimately names its options in instructions).
    for qid, g in case.gold.items():
        value = getattr(g, "value", None)
        if value is None or not isinstance(value, (int, float, str, bool)):
            continue
        text_value = str(value).strip().lower()
        if not text_value:
            continue
        q = case.questions.get(qid)
        allowed_choices: set[str] = set()
        if isinstance(q, dict) and isinstance(q.get("choices"), list):
            allowed_choices = {str(c).strip().lower() for c in q["choices"]}
        qid_lower = qid.lower()
        for text in jev_bound_strings:
            normalized = (
                text.replace('\\"', '"').replace("\\\\", "\\").replace("'", '"').lower()
            )
            if qid_lower not in normalized or text_value not in normalized:
                continue
            if text_value in allowed_choices and qid_lower not in _LEAK_TRIGGER_WORDS:
                # A value that is an allowed choice appears in every
                # instruction that lists the options; require an explicit
                # trigger word near the question id (e.g. "answer is X")
                # to avoid making every choice question unvalidatable.
                if not _has_leak_trigger(normalized, qid_lower):
                    continue
            problems.append(
                f"JEV-input text may leak gold value for question {qid!r} "
                f"(question id and answer value both present)"
            )
            break

    # Secret patterns: a state embedding credentials must never reach the
    # provider or persist into artifacts.
    if strict_secrets:
        for pattern in _contains_secret_patterns(case.state):
            problems.append(f"state contains secret pattern {pattern!r}")
    return problems


# Phrases that turn a co-occurrence into an unmistakable leak even when
# the gold value is also an allowed choice (choice questions legitimately
# enumerate their options in instructions).
_LEAK_TRIGGER_WORDS = (
    "answer is",
    "answer:",
    "correct",
    "gold",
    "solution",
    "expected value",
)


def _has_leak_trigger(normalized_text: str, _qid: str) -> bool:
    return any(trigger in normalized_text for trigger in _LEAK_TRIGGER_WORDS)


def _contains_secret_patterns(text: str) -> list[str]:
    from jev_agent_eval.client.redaction import PATTERNS

    hits: list[str] = []
    for pattern, _replacement in PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern[:40])
    return hits


class LocalJSONLDataset:
    """Streaming loader for canonical cases from JSONL files."""

    name = "local_jsonl"
    version = "1.0"

    def __init__(self, paths: list[str | Path]):
        self.paths = [Path(p) for p in paths]

    def iter_cases(self, *, validate: bool = True) -> list[CanonicalCase]:
        cases: list[CanonicalCase] = []
        seen_ids: set[str] = set()
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(f"dataset file not found: {path}")
            for i, obj in enumerate(iter_jsonl(path), 1):
                try:
                    case = CanonicalCase.model_validate(obj)
                except Exception as e:
                    raise ValueError(f"{path}:{i}: invalid canonical case: {e}") from e
                if case.case_id in seen_ids:
                    raise ValueError(f"{path}:{i}: duplicate case_id {case.case_id!r}")
                seen_ids.add(case.case_id)
                if validate:
                    problems = validate_case(case)
                    if problems:
                        raise LeakageError(
                            f"{path}:{i}: case {case.case_id!r} failed validation: "
                            + "; ".join(problems)
                        )
                cases.append(case)
        return cases
