"""Stop/continue and stuck/progress detection question packs (spec sections 5.F, 5.G, 22)."""

from __future__ import annotations

from jev_agent_eval.questions.base import QuestionPackResult
from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.request import JEVQuestion

STOP_LABELS = ["stop", "continue", "verify", "ask_user", "escalate"]
STUCK_LABELS = [
    "progressing",
    "slow_but_progressing",
    "stuck",
    "blocked",
    "needs_user_input",
    "needs_escalation",
]

STOP_QUESTION_ID = "stop_decision"
STUCK_QUESTION_ID = "progress_state"


class StopContinueQuestionPack:
    name = "stop_continue"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        questions = {
            STOP_QUESTION_ID: JEVQuestion(
                type="choice",
                instructions=(
                    "Given the current state of the work, should the agent "
                    "terminate, continue working, verify the result, ask the "
                    "user, or escalate?"
                ),
                criteria={
                    "stop": "The objective is satisfied and verified.",
                    "continue": "Meaningful work remains and progress is being made.",
                    "verify": "The work appears done but verification is missing.",
                    "ask_user": "A decision only the user can make is blocking.",
                    "escalate": "The task exceeds the current model's capability.",
                },
            )
        }
        if STOP_QUESTION_ID not in case.gold:
            raise ValueError(
                f"case {case.case_id}: stop/continue pack requires gold for {STOP_QUESTION_ID!r}"
            )
        return QuestionPackResult(
            questions=questions,
            gold={STOP_QUESTION_ID: case.gold[STOP_QUESTION_ID]},
            capability="stop_continue",
            tags=list(case.tags),
        )


class StuckDetectionQuestionPack:
    name = "stuck_detection"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        questions = {
            STUCK_QUESTION_ID: JEVQuestion(
                type="choice",
                instructions=(
                    "Classify the agent's progress over the recent window of "
                    "steps shown in the state."
                ),
                criteria={
                    "progressing": "Steps are producing meaningful movement toward the goal.",
                    "slow_but_progressing": "Progress is real but slow.",
                    "stuck": "Repeating the same failing actions without progress.",
                    "blocked": "An external obstacle prevents any progress.",
                    "needs_user_input": "Progress requires user clarification.",
                    "needs_escalation": "Progress requires stronger reasoning or privileges.",
                },
            )
        }
        if STUCK_QUESTION_ID not in case.gold:
            raise ValueError(
                f"case {case.case_id}: stuck pack requires gold for {STUCK_QUESTION_ID!r}"
            )
        return QuestionPackResult(
            questions=questions,
            gold={STUCK_QUESTION_ID: case.gold[STUCK_QUESTION_ID]},
            capability="stuck_detection",
            tags=list(case.tags),
        )
