from jev_agent_eval.questions.base import PreBakedQuestionPack, QuestionPack, QuestionPackResult
from jev_agent_eval.questions.completion import CompletionQuestionPack
from jev_agent_eval.questions.escalation import ESCALATION_LABELS, EscalationQuestionPack
from jev_agent_eval.questions.injection import InjectionQuestionPack
from jev_agent_eval.questions.registry import get_pack, register_pack, registered_packs
from jev_agent_eval.questions.relevance import (
    RELEVANCE_SCALE,
    ContextRelevanceQuestionPack,
    ToolRelevanceQuestionPack,
)
from jev_agent_eval.questions.routing import ROUTING_LABELS, RoutingQuestionPack
from jev_agent_eval.questions.stuck import (
    STOP_LABELS,
    STUCK_LABELS,
    StopContinueQuestionPack,
    StuckDetectionQuestionPack,
)
from jev_agent_eval.questions.tool_risk import RISK_LABELS, ToolRiskQuestionPack
from jev_agent_eval.questions.verification import VerificationQuestionPack

_CAPABILITY_PACKS: list[QuestionPack] = [
    RoutingQuestionPack(),
    ToolRiskQuestionPack(),
    CompletionQuestionPack(),
    StopContinueQuestionPack(),
    StuckDetectionQuestionPack(),
    VerificationQuestionPack(),
    EscalationQuestionPack(),
    ToolRelevanceQuestionPack(),
    ContextRelevanceQuestionPack(),
    InjectionQuestionPack(),
]

for _pack in _CAPABILITY_PACKS:
    register_pack(_pack)

__all__ = [
    "ESCALATION_LABELS",
    "PreBakedQuestionPack",
    "QuestionPack",
    "QuestionPackResult",
    "RELEVANCE_SCALE",
    "RISK_LABELS",
    "ROUTING_LABELS",
    "STOP_LABELS",
    "STUCK_LABELS",
    "CompletionQuestionPack",
    "ContextRelevanceQuestionPack",
    "EscalationQuestionPack",
    "InjectionQuestionPack",
    "RoutingQuestionPack",
    "StuckDetectionQuestionPack",
    "StopContinueQuestionPack",
    "ToolRelevanceQuestionPack",
    "ToolRiskQuestionPack",
    "VerificationQuestionPack",
    "get_pack",
    "register_pack",
    "registered_packs",
]
