"""Non-JEV baselines (spec section 39).

Three mandatory baselines run alongside JEV so the release can answer
"does JEV add value beyond trivial or deterministic logic?":

- Baseline A — random: sanity check for the scoring pipeline.
- Baseline B — majority class: exposes class imbalance.
- Baseline C — deterministic heuristic: capability-specific rules
  (shell-risk regex, test-count completion, keyword routing, simple
  instruction-constraint verifier, evidence contradiction checker).

If JEV cannot beat Baseline C on a capability, the spec requires an
investigation into whether that capability should be handled by
deterministic logic instead.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from jev_agent_eval.schemas.case import CanonicalCase


@dataclass
class BaselineResult:
    """Predictions from one baseline over the dataset."""

    name: str
    # case_id -> {question_id: predicted value} (noul: 0/1/0.5, choice: str, score: float)
    predictions: dict[str, dict[str, Any]]
    description: str = ""


# Baseline A ------------------------------------------------------------------


def random_baseline(
    cases: list[CanonicalCase],
    seed: int = 0,
) -> BaselineResult:
    """Uniform random over the answer space (sanity check only).

    For noul questions the prediction alternates deterministically per
    repeat index so repeated evaluation doesn't need per-call randomness;
    here each case gets one independent draw.
    """
    rng = random.Random(seed)
    predictions: dict[str, dict[str, Any]] = {}
    for case in cases:
        preds: dict[str, Any] = {}
        for qid, question in case.questions.items():
            if not isinstance(question, dict):
                continue
            qtype = question.get("type")
            if qtype == "noul":
                preds[qid] = rng.choice([0.0, 1.0])
            elif qtype == "choice":
                options = _choice_options(question)
                preds[qid] = rng.choice(options) if options else None
            elif qtype == "score":
                preds[qid] = rng.random()
        predictions[case.case_id] = preds
    return BaselineResult(
        name="random",
        predictions=predictions,
        description="Baseline A: uniform random over the answer space (sanity check)",
    )


# Baseline B ------------------------------------------------------------------


def majority_baseline(
    cases: list[CanonicalCase],
    training_cases: list[CanonicalCase] | None = None,
) -> BaselineResult:
    """Majority class per question id.

    The majority is computed over `training_cases` when given (honest
    train/test separation); otherwise over `cases` itself (the spec's
    intent is to expose class imbalance, and the dataset-level majority
    is that diagnostic).
    """
    source = training_cases or cases
    counts: dict[str, dict[Any, int]] = {}
    for case in source:
        for qid, gold in case.gold.items():
            value = getattr(gold, "value", None)
            if value is None:
                continue
            counts.setdefault(qid, {})
            counts[qid][value] = counts[qid].get(value, 0) + 1
    majority: dict[str, Any] = {}
    for qid, dist in counts.items():
        # Ties break toward the lexically smallest repr for determinism.
        majority[qid] = min(dist.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]

    predictions: dict[str, dict[str, Any]] = {}
    for case in cases:
        preds: dict[str, Any] = {}
        for qid, question in case.questions.items():
            if not isinstance(question, dict):
                continue
            if qid in majority:
                preds[qid] = majority[qid]
            elif question.get("type") == "noul":
                # Unseen question: default to the negative class.
                preds[qid] = 0.0
            elif question.get("type") == "score":
                preds[qid] = 0.5
        predictions[case.case_id] = preds
    return BaselineResult(
        name="majority",
        predictions=predictions,
        description="Baseline B: majority gold class per question id (exposes imbalance)",
    )


# Baseline C ------------------------------------------------------------------

# Obvious shell-risk regex (§39 capability-specific example).
_SHELL_RISK_RE = re.compile(
    r"(?:rm\s+-rf?\s+(?![-./\w]*\./)|sudo\s|chmod\s+777|dd\s+if=|mkfs|"
    r"drop\s+table|truncate\s+table|git\s+push\s+--force|git\s+reset\s+--hard|"
    r"npm\s+publish|aws\s+s3\s+rm|kubectl\s+delete\s+namespace|"
    r">\s*/dev/sd|curl[^|]*\|\s*(?:ba)?sh|wget[^|]*\|\s*(?:ba)?sh|"
    r"shutdown|reboot|:(){ :\|:& };:)",
    re.IGNORECASE,
)
# Scoped rm -rf on a local rebuildable directory is the §14.1 minimal-pair
# design: `rm -rf ./build` is low-risk by construction.
_SHELL_SCOPED_SAFE_RE = re.compile(r"rm\s+-rf\s+\./[\w.-]*")
# Read-only commands: strong benign signal. Matches after a proposal
# marker, a line break, or sentence punctuation — the command must start
# a command position, not appear inside a word like "concatenate".
_SHELL_BENIGN_RE = re.compile(
    r"(?:^|(?<=[:\n.]))\s*(?:cat|ls|pwd|head|tail|wc|grep|which|find|file|stat|git\s+(?:status|diff|log|show)|"
    r"docker\s+ps|kubectl\s+get|npm\s+(?:test|run\s+test)|pytest|python\s+-c\s+['\"]import)\s",
    re.IGNORECASE,
)
# Explicit test-count completion evidence.
_TEST_COUNT_RE = re.compile(
    r"(?:(\d+)\s*/\s*(\d+)\s+tests?\s+pass|(?:all|suite)\s+(?:green|pass(?:ing)?)|"
    r"\b(\d+)\s+passed\b|\bpassed:\s*(\d+)|\bfail\w*['\"]?\s*[:=]\s*['\"]?0\b)",
    re.IGNORECASE,
)
_FAILING_RE = re.compile(r"(?:\d+\s+failing|\bfail(?:ed|ure)?s?\b|✗|FAILED)", re.IGNORECASE)
_FAIL_ZERO_RE = re.compile(r"fail\w*['\"]?\s*[:=]\s*['\"]?0\b", re.IGNORECASE)
# Keyword routing: which tool family a request needs.
_ROUTING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "edit_files": ("edit", "modify", "change", "update", "fix", "refactor", "rename", "patch"),
    "run_tests": ("test", "pytest", "unittest", "suite", "coverage"),
    "run_command": ("run", "execute", "command", "shell", "script", "bash"),
    "search_code": ("search", "find", "grep", "locate", "where", "grep for"),
    "read_files": ("read", "open", "show", "view", "inspect", "cat"),
    "browser": ("browser", "click", "navigate", "webpage", "url", "screenshot"),
    "ask_user": ("ask", "clarify", "ambiguous", "unclear", "confirm with"),
}
# Instruction-constraint verifier: directive sentences in instructions.
_CONSTRAINT_RE = re.compile(
    r"(?:must|should|do not|don't|never|always|required to|has to|only)\b[^.;]{3,80}",
    re.IGNORECASE,
)
# Evidence contradiction markers (agent claim vs state evidence).
_CONTRADICTION_MARKERS = (
    "not applied",
    "no changes",
    "still reproduces",
    "signature changed",
    "not run",
    "failed to",
    "error:",
    "traceback",
    "8 months ago",
    "the opposite outcome holds",
)


def heuristic_answer(question: dict[str, Any], case: CanonicalCase) -> Any:
    """Baseline C: one deterministic heuristic answer for a question.

    Returns the predicted value, or None when no heuristic fires (the
    question is out of this baseline's coverage).
    """
    qtype = question.get("type")
    instructions = (question.get("instructions") or "").lower()
    qid = (question.get("id") or "").lower()
    signal = f"{instructions} {qid}"
    state = case.state or ""

    # Scope the command judgment to the proposed action when present —
    # surrounding context must not change the risk read. Markers match
    # case-insensitively (real states capitalize "Proposed tool call:").
    tail = state
    lowered_state = state.lower()
    for marker in (
        "proposed tool call:",
        "proposed action:",
        "intends to run:",
        "the agent runs:",
        "executed tool:",
    ):
        if marker in lowered_state:
            tail = lowered_state.split(marker)[-1]
            break
    tail = tail.lower()

    if qtype == "noul":
        if "high risk" in signal or "destructive" in signal or "risk" in qid:
            if _SHELL_SCOPED_SAFE_RE.search(tail):
                return 0.0
            if _SHELL_RISK_RE.search(tail):
                return 1.0
            if _SHELL_BENIGN_RE.match(tail.strip()) or _SHELL_BENIGN_RE.search(tail):
                return 0.0
            return None
        if "goal complete" in signal or "claim supported" in signal or "supported by" in signal:
            sanitized = _FAIL_ZERO_RE.sub("", state)
            m = _TEST_COUNT_RE.search(sanitized)
            if m:
                # "N/M tests pass" completes only when N == M; a bare
                # "failed: 0" or "N passed" with no failing evidence
                # counts as completion evidence.
                a, b = m.group(1), m.group(2)
                if a is not None and b is not None:
                    return 1.0 if a == b else 0.0
                return 1.0
            if _FAILING_RE.search(sanitized):
                return 0.0
            return None
        if "stuck" in signal or "progress" in signal:
            retry_language = any(
                w in state.lower()
                for w in ("again", "a third time", "third time", "re-running", "retrying", "still")
            )
            return 1.0 if (retry_language and _FAILING_RE.search(sanitized)) else 0.0
        if "in scope" in signal or "within the scope" in signal:
            return (
                1.0
                if (_SHELL_BENIGN_RE.search(tail) or _SHELL_SCOPED_SAFE_RE.search(tail))
                else 0.0
            )
        if "constraint" in signal or "compliance" in signal or "instruct" in signal:
            return None  # handled by the choice/verifier path below when applicable
        return None
    if qtype == "choice":
        options = _choice_options(question)
        if not options:
            return None
        lowered = {str(o).lower() for o in options}
        if {"safe", "low_risk", "high_risk", "forbidden", "irreversible", "sensitive"} & lowered:
            if _SHELL_SCOPED_SAFE_RE.search(tail):
                choice = "low_risk" if "low_risk" in lowered else "safe"
            elif "rm -rf /" in tail or "chmod 777" in tail:
                choice = "forbidden" if "forbidden" in lowered else "high_risk"
            elif _SHELL_RISK_RE.search(tail):
                choice = "high_risk" if "high_risk" in lowered else "irreversible"
            elif _SHELL_BENIGN_RE.search(tail):
                choice = "safe" if "safe" in lowered else "low_risk"
            else:
                return None
            return choice if choice in lowered else None
        if "routing" in signal or "next tool" in signal or "tool" in signal:
            text = f"{state[:2000]} {instructions}".lower()
            for tool_name, keywords in _ROUTING_KEYWORDS.items():
                if any(k in text for k in keywords) and tool_name in lowered:
                    return tool_name
            return None
        return None
    if qtype == "score":
        return None
    return None


def heuristic_baseline(cases: list[CanonicalCase]) -> BaselineResult:
    """Baseline C: capability-specific deterministic heuristics.

    Questions with no heuristic coverage are omitted from the prediction
    map — coverage is part of the report (a heuristic that fires on 5% of
    questions and is right on all of them has not replaced JEV).
    """
    predictions: dict[str, dict[str, Any]] = {}
    covered = 0
    total = 0
    for case in cases:
        preds: dict[str, Any] = {}
        for qid, question in case.questions.items():
            if not isinstance(question, dict):
                continue
            total += 1
            answer = heuristic_answer(question, case)
            if answer is not None:
                preds[qid] = answer
                covered += 1
        predictions[case.case_id] = preds
    return BaselineResult(
        name="heuristic",
        predictions=predictions,
        description=(
            f"Baseline C: deterministic capability-specific heuristics "
            f"(covered {covered}/{total} questions)"
        ),
    )


def _choice_options(question: dict[str, Any]) -> list[str]:
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        return list(criteria.keys())
    if isinstance(criteria, list):
        return [str(c) for c in criteria]
    return []


BASELINES = {
    "random": random_baseline,
    "majority": majority_baseline,
    "heuristic": heuristic_baseline,
}
