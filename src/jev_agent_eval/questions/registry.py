"""Question pack registry: adding a new pack never requires touching the runner."""

from __future__ import annotations

from jev_agent_eval.questions.base import PreBakedQuestionPack, QuestionPack

_PACKS: dict[str, QuestionPack] = {}


def register_pack(pack: QuestionPack) -> QuestionPack:
    _PACKS[pack.name] = pack
    return pack


def get_pack(name: str) -> QuestionPack:
    if name not in _PACKS:
        raise KeyError(f"unknown question pack {name!r}; registered: {sorted(_PACKS)}")
    return _PACKS[name]


def registered_packs() -> list[str]:
    return sorted(_PACKS)


# Built-in default.
register_pack(PreBakedQuestionPack())
