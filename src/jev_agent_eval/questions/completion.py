"""Goal completion question pack (spec sections 5.E, 21)."""

from __future__ import annotations

from jev_agent_eval.questions.base import QuestionPackResult
from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.request import JEVQuestion

QUESTION_ID = "goal_complete"


class CompletionQuestionPack:
    name = "goal_completion"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        questions = {
            QUESTION_ID: JEVQuestion(
                type="noul",
                instructions=(
                    "Has the requested objective actually been satisfied given the "
                    "observed evidence in the state, independent of the agent's own claim?"
                ),
            )
        }
        if QUESTION_ID not in case.gold:
            raise ValueError(
                f"case {case.case_id}: completion pack requires gold for {QUESTION_ID!r}"
            )
        return QuestionPackResult(
            questions=questions,
            gold={QUESTION_ID: case.gold[QUESTION_ID]},
            capability="goal_completion",
            tags=list(case.tags),
        )
