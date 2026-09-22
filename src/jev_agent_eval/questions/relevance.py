"""Context relevance question packs (spec sections 5.B tool relevance, 5.J context relevance, 25)."""

from __future__ import annotations

from jev_agent_eval.questions.base import QuestionPackResult
from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.request import JEVQuestion

TOOL_RELEVANCE_QUESTION_ID = "tool_relevant"
CONTEXT_RELEVANCE_QUESTION_ID = "evidence_importance"

RELEVANCE_SCALE = ["irrelevant", "low", "medium", "high", "critical"]


class ToolRelevanceQuestionPack:
    """Is the proposed tool appropriate for the current state? (spec 5.B)"""

    name = "tool_relevance"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        questions = {
            TOOL_RELEVANCE_QUESTION_ID: JEVQuestion(
                type="noul",
                instructions=(
                    "Is calling this tool an appropriate next step for the "
                    "current state of the task?"
                ),
            )
        }
        if TOOL_RELEVANCE_QUESTION_ID not in case.gold:
            raise ValueError(
                f"case {case.case_id}: tool relevance pack requires gold for {TOOL_RELEVANCE_QUESTION_ID!r}"
            )
        return QuestionPackResult(
            questions=questions,
            gold={TOOL_RELEVANCE_QUESTION_ID: case.gold[TOOL_RELEVANCE_QUESTION_ID]},
            capability="tool_relevance",
            tags=list(case.tags),
        )


class ContextRelevanceQuestionPack:
    """How important is one evidence item for the current decision? (spec 5.J, 25)"""

    name = "context_relevance"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        questions = {
            CONTEXT_RELEVANCE_QUESTION_ID: JEVQuestion(
                type="choice",
                instructions=(
                    "How important is this evidence for deciding whether the "
                    "current coding task is complete?"
                ),
                criteria={
                    "irrelevant": "No bearing on the current task.",
                    "low": "Background information only.",
                    "medium": "Useful context but not decisive.",
                    "high": "Strongly informs the decision.",
                    "critical": "Decisive for the decision.",
                },
            )
        }
        if CONTEXT_RELEVANCE_QUESTION_ID not in case.gold:
            raise ValueError(
                f"case {case.case_id}: context relevance pack requires gold for {CONTEXT_RELEVANCE_QUESTION_ID!r}"
            )
        return QuestionPackResult(
            questions=questions,
            gold={CONTEXT_RELEVANCE_QUESTION_ID: case.gold[CONTEXT_RELEVANCE_QUESTION_ID]},
            capability="context_relevance",
            tags=list(case.tags),
        )
