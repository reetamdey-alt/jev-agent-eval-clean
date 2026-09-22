"""Model escalation question pack (spec sections 5.I, 39)."""

from __future__ import annotations

from jev_agent_eval.questions.base import QuestionPackResult
from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.request import JEVQuestion

ESCALATION_LABELS = [
    "deterministic",
    "jev_only",
    "fast_llm",
    "strong_reasoning_model",
    "human_authorization",
]

QUESTION_ID = "escalation_tier"


class EscalationQuestionPack:
    name = "escalation"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        questions = {
            QUESTION_ID: JEVQuestion(
                type="choice",
                instructions=(
                    "What is the minimum capability tier needed to handle this "
                    "task correctly, given the state?"
                ),
                criteria={
                    "deterministic": "A fixed rule or script suffices.",
                    "jev_only": "A single structured JEV decision suffices.",
                    "fast_llm": "A fast general model suffices.",
                    "strong_reasoning_model": "Deep multi-step reasoning is required.",
                    "human_authorization": "A human must approve or perform the work.",
                },
            )
        }
        if QUESTION_ID not in case.gold:
            raise ValueError(
                f"case {case.case_id}: escalation pack requires gold for {QUESTION_ID!r}"
            )
        return QuestionPackResult(
            questions=questions,
            gold={QUESTION_ID: case.gold[QUESTION_ID]},
            capability="escalation",
            tags=list(case.tags),
        )
