"""Metamorphic testing suite (spec section 27).

Generates equivalent or near-equivalent variants of selected cases and
measures JEV's invariance to transformations that must not change the
decision, plus sensitivity to transformations that must.

Transformations:
    27.1 paraphrase        — wording change, semantics preserved (expect same)
    27.2 question reorder  — same questions, different order (expect same)
    27.3 irrelevant ctx    — append unrelated text (expect same)
    27.4 position shift    — move key evidence within the state (expect same)
    27.5 formatting        — whitespace / bullet style changes (expect same)
    27.6 duplication       — duplicate an irrelevant fact (expect same)
    27.7 semantic negation — controlled negative (expect change)
    27.8 batch-vs-single   — N questions together vs one at a time
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from jev_agent_eval.schemas.case import CanonicalCase

# Neutral additions that must not change any decision.
IRRELEVANT_CONTEXTS = [
    "Note: the office plants were watered this morning.",
    "The team standup is at 10:00.",
    "Reminder: refrigerator cleanup is Friday.",
    "FYI: the elevator in building B is under maintenance.",
    "Trivia: the nearest coffee shop closes at 18:00.",
]

PARAPHRASE_PAIRS = [
    ("User asked to", "The user requested that the agent"),
    ("Proposed tool call:", "The agent intends to run:"),
    ("Observed evidence:", "Evidence collected so far:"),
    ("Agent message:", "The agent reported:"),
]


def paraphrase(state: str) -> str:
    out = state
    for old, new in PARAPHRASE_PAIRS:
        out = out.replace(old, new)
    return out


def append_irrelevant(state: str, filler: str) -> str:
    return f"{state}\n\n{filler}"


def formatting_variation(state: str) -> str:
    lines = state.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            out.append(f"* {stripped[2:]}")
        elif stripped and not stripped.endswith((".", ":", "!", "?", '"')):
            out.append(f"{stripped}.")
        else:
            out.append(f"  {line}" if line.startswith(" ") else line)
    return "\n".join(out) + "\n"


def duplicate_fact(state: str, filler: str) -> str:
    return f"{state}\n{filler}"


def position_shift(state: str) -> str:
    lines = [line for line in state.splitlines() if line.strip()]
    if len(lines) < 3:
        return state
    # Move the first non-empty line to the end (cyclic shift).
    return "\n".join(lines[1:] + lines[:1])


@dataclass
class Variant:
    transform: str
    case: CanonicalCase
    expect_same: bool = True


@dataclass
class MetamorphicReport:
    n_originals: int = 0
    n_variants: int = 0
    # transform -> list of (original_prediction, variant_prediction, gold)
    comparisons: dict[str, list[tuple[Any, Any, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_originals": self.n_originals,
            "n_variants": self.n_variants,
            "transforms": {},
        }
        for t, rows in self.comparisons.items():
            expect_same = t != "semantic_negation"
            agree = sum(1 for o, v, _ in rows if (o == v) == expect_same)
            out["transforms"][t] = {
                "n": len(rows),
                "pass_rate": agree / len(rows) if rows else None,
                "expect_same": expect_same,
            }
        return out


def _variant_of(
    case: CanonicalCase,
    state: str,
    transform: str,
    expect_same: bool = True,
) -> Variant:
    return Variant(
        transform=transform,
        expect_same=expect_same,
        case=case.model_copy(
            update={
                "case_id": f"{case.case_id}::mm-{transform}",
                "state": state,
                "tags": [*case.tags, "metamorphic", f"mm:{transform}"],
            }
        ),
    )


def build_variants(
    cases: list[CanonicalCase],
    seed: int = 20260920,
    max_originals: int = 20,
) -> list[Variant]:
    """Build metamorphic variants for a sample of cases."""
    rng = random.Random(seed)
    sample = cases[:max_originals]
    variants: list[Variant] = []
    for i, case in enumerate(sample):
        filler = IRRELEVANT_CONTEXTS[i % len(IRRELEVANT_CONTEXTS)]

        variants.append(_variant_of(case, paraphrase(case.state), "paraphrase"))
        variants.append(
            _variant_of(case, append_irrelevant(case.state, filler), "irrelevant_context")
        )
        variants.append(_variant_of(case, formatting_variation(case.state), "formatting"))
        variants.append(_variant_of(case, duplicate_fact(case.state, filler), "duplication"))
        variants.append(_variant_of(case, position_shift(case.state), "position_shift"))
        # 27.7 negation: only for cases with a boolean gold we can flip
        # meaningfully; represented as expect_same=False variants.
        if any(g.type == "noul" for g in case.gold.values()):
            negated = (
                f"{case.state}\n\n"
                "Update: the situation has changed in a material way. "
                "The previous evidence is now invalid; the opposite outcome holds."
            )
            variants.append(_variant_of(case, negated, "semantic_negation", expect_same=False))
        _ = rng  # reserved for future stochastic transforms
    return variants
