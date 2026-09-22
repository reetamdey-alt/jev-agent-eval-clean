"""Trace runner (spec sections 13.3, 20).

Consumes normalized coding-agent trace states (JSONL per spec section 20)
and evaluates JEV control-plane decisions against gold labels embedded in
the trace records. The evaluator projects any agent's trace into the
normalized schema; no framework-specific logic lives here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TraceToolResult(BaseModel):
    status: str | None = None
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None


class TraceStep(BaseModel):
    """Normalized trace schema (spec section 20), framework-independent."""

    session_id: str | None = None
    task_id: str | None = None
    step: int = 0
    timestamp: str | None = None
    user_request: str | None = None
    agent_message: str | None = None
    tool_call: TraceToolCall | None = None
    tool_result: TraceToolResult | None = None
    files_changed: list[str] = Field(default_factory=list)
    tests: dict[str, int] | None = None
    git: dict[str, Any] | None = None


def project_trace_to_state(step: TraceStep) -> str:
    """Render a trace step as natural-language state for JEV (spec 22)."""
    parts: list[str] = []
    if step.user_request:
        parts.append(f"User request: {step.user_request}")
    if step.agent_message:
        parts.append(f"Agent message: {step.agent_message}")
    if step.tool_call:
        args = json.dumps(step.tool_call.arguments, ensure_ascii=False)[:2000]
        parts.append(f"Proposed/executed tool: {step.tool_call.name}({args})")
    if step.tool_result:
        outcome = step.tool_result.status or ("error" if step.tool_result.exit_code else "ok")
        detail = (step.tool_result.stderr or step.tool_result.stdout or "")[:2000]
        parts.append(f"Tool result: {outcome}\n{detail}")
    if step.files_changed:
        parts.append("Files changed: " + ", ".join(step.files_changed[:20]))
    if step.tests:
        parts.append(f"Tests: {step.tests}")
    if step.git:
        parts.append(f"Git: {step.git}")
    parts.append(f"Step number: {step.step}")
    return "\n\n".join(parts)


def iter_trace_steps(path: str | Path) -> Iterable[TraceStep]:
    from jev_agent_eval.utils.jsonl import iter_jsonl

    for obj in iter_jsonl(path):
        yield TraceStep.model_validate(obj)


def trace_to_cases(path: str | Path, dataset_version: str = "trace-1.0") -> list[Any]:
    """Build canonical cases from a trace file.

    Gold labels come from optional `gold` fields on each trace record; steps
    without gold are not evaluated (shadow mode).
    """
    from typing import cast

    from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer
    from jev_agent_eval.schemas.request import JEVQuestion
    from jev_agent_eval.utils.jsonl import iter_jsonl

    QuestionType = Literal["noul", "choice", "score"]
    cases = []
    seen_ids: set[str] = set()
    for obj in iter_jsonl(path):
        gold_fields = obj.get("gold") or {}
        if not gold_fields:
            continue
        step = TraceStep.model_validate({k: v for k, v in obj.items() if k != "gold"})
        state = project_trace_to_state(step)
        questions: dict[str, JEVQuestion] = {}
        gold: dict[str, GoldAnswer] = {}
        for qid, g in gold_fields.items():
            if isinstance(g, dict):
                instructions = g.get("instructions", "")
                # Accept both question_type and type (JEVQuestion's own
                # field name); a gold dict using 'type' must not be
                # silently coerced to noul.
                qtype_raw = g.get("question_type") or g.get("type") or "noul"
                if qtype_raw not in ("noul", "choice", "score"):
                    raise ValueError(
                        f"trace record gold for question {qid!r} has unsupported "
                        f"type {qtype_raw!r} (session={obj.get('session_id', '?')}, step={step.step})"
                    )
                qtype = cast(QuestionType, qtype_raw)
                value = g.get("value")
                if value is None:
                    # A missing value is a partial annotation, not gold=0:
                    # fabricating a negative label corrupts accuracy.
                    raise ValueError(
                        f"trace record gold for question {qid!r} is missing 'value' "
                        f"(session={obj.get('session_id', '?')}, step={step.step})"
                    )
            else:
                instructions, qtype, value = "", cast(QuestionType, "noul"), g
            questions[qid] = JEVQuestion(type=qtype, instructions=instructions)
            gold[qid] = GoldAnswer(type=qtype, value=value)
        case_id = f"trace-{obj.get('session_id', 's')}-{step.step:04d}"
        if case_id in seen_ids:
            raise ValueError(
                f"duplicate trace case_id {case_id!r} (records missing both "
                "session_id and step collide; give each record unique identifiers)"
            )
        seen_ids.add(case_id)
        cases.append(
            CanonicalCase(
                case_id=case_id,
                dataset="agent-trace",
                dataset_version=dataset_version,
                capability=obj.get("capability", "goal_completion"),
                tags=["trace"],
                state=state,
                questions={qid: q.model_dump() for qid, q in questions.items()},
                gold=gold,
                metadata={"source": "trace", "source_ref": str(path)},
            )
        )
    return cases
