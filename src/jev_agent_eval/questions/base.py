"""Question pack abstraction (spec section 10).

A question pack converts a canonical case into one or more JEV questions.
Packs are pure functions of the canonical case and never make network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer
from jev_agent_eval.schemas.request import JEVQuestion


@dataclass
class QuestionPackResult:
    questions: dict[str, JEVQuestion]
    gold: dict[str, GoldAnswer]
    capability: str
    tags: list[str] = field(default_factory=list)


@runtime_checkable
class QuestionPack(Protocol):
    name: str
    version: str

    def build(self, case: CanonicalCase) -> QuestionPackResult: ...


class PreBakedQuestionPack:
    """Default pack: uses questions and gold pre-baked into the case itself.

    This is how `jev-agent-core` and transformed public-benchmark cases are
    authored: the question wording ships with the dataset (versioned as part
    of the dataset), and this pack simply projects it into the request
    schema. It is a pure function of the case.
    """

    name = "prebaked"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        questions: dict[str, JEVQuestion] = {}
        gold: dict[str, GoldAnswer] = dict(case.gold)
        for qid, raw in case.questions.items():
            if isinstance(raw, JEVQuestion):
                questions[qid] = raw
            elif isinstance(raw, dict):
                questions[qid] = JEVQuestion.model_validate(raw)
            else:
                raise ValueError(f"case {case.case_id}: question {qid!r} is not a question object")
            if qid not in gold:
                raise ValueError(f"case {case.case_id}: question {qid!r} has no gold answer")
        return QuestionPackResult(
            questions=questions,
            gold=gold,
            capability=case.capability,
            tags=list(case.tags),
        )
