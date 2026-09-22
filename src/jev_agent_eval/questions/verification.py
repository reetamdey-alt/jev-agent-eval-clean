"""Claim-evidence verification question pack (spec sections 5.H, 24)."""

from __future__ import annotations

from jev_agent_eval.questions.base import QuestionPackResult
from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.request import JEVQuestion

QUESTION_ID = "claim_supported"


class VerificationQuestionPack:
    name = "claim_evidence"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        questions = {
            QUESTION_ID: JEVQuestion(
                type="noul",
                instructions=("Is the agent's claim supported by the provided evidence?"),
            )
        }
        if QUESTION_ID not in case.gold:
            raise ValueError(
                f"case {case.case_id}: verification pack requires gold for {QUESTION_ID!r}"
            )
        return QuestionPackResult(
            questions=questions,
            gold={QUESTION_ID: case.gold[QUESTION_ID]},
            capability="claim_evidence",
            tags=list(case.tags),
        )
