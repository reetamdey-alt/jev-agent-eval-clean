"""Task routing question pack (spec sections 5.A, 62-63).

Question wording is part of the versioned evaluation surface: any wording
change requires a question-pack version change.
"""

from __future__ import annotations

from jev_agent_eval.questions.base import QuestionPackResult
from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer, GoldProvenance
from jev_agent_eval.schemas.request import JEVQuestion

ROUTING_LABELS = [
    "implementation",
    "bug_fix",
    "debugging",
    "refactor",
    "research",
    "code_review",
    "documentation",
    "explanation",
    "configuration",
    "investigation",
    "planning",
    "unknown",
]

QUESTION_ID = "task_type"


class RoutingQuestionPack:
    name = "routing"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        criteria = {
            label: f"The request asks for {label.replace('_', ' ')}." for label in ROUTING_LABELS
        }
        questions = {
            QUESTION_ID: JEVQuestion(
                type="choice",
                instructions=(
                    "Classify the kind of work the coding agent is being asked "
                    "to perform based on the user request in the state."
                ),
                criteria=criteria,
            )
        }
        gold = case.gold.get(QUESTION_ID)
        if gold is None:
            raise ValueError(f"case {case.case_id}: routing pack requires gold for {QUESTION_ID!r}")
        return QuestionPackResult(
            questions=questions,
            gold={QUESTION_ID: gold},
            capability="task_routing",
            tags=list(case.tags),
        )


def make_routing_gold(
    label: str, provenance: GoldProvenance = GoldProvenance.RULE_DERIVED
) -> GoldAnswer:
    return GoldAnswer(type="choice", value=label, provenance=provenance)
