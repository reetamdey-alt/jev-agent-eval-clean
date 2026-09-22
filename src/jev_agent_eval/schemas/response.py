"""JEV response schema (spec section 11.2).

The parser must tolerate optional provider fields while enforcing the fields
needed to score. Validation of probability vectors happens in scoring, not
here, so that invalid responses can be recorded and categorized rather than
dropped.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class JEVAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    legend: dict[str, str] | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


class JEVUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None


class JEVResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, JEVAnswer] = Field(default_factory=dict)
    usage: JEVUsage | None = None
