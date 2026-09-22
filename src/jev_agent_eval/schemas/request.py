"""JEV request schema (spec section 11.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class JEVQuestion(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str
    # Live contract: `choice` criteria are a label->description mapping;
    # `score` criteria are an ordered list of level descriptions.
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _criteria_shape(self) -> JEVQuestion:
        if (
            self.type == "score"
            and self.criteria is not None
            and not isinstance(self.criteria, list)
        ):
            raise ValueError("score criteria must be a list")
        if (
            self.type == "choice"
            and self.criteria is not None
            and not isinstance(self.criteria, dict)
        ):
            raise ValueError("choice criteria must be a mapping")
        return self


class JEVRequest(BaseModel):
    state: str
    model: str
    questions: dict[str, JEVQuestion] = Field(default_factory=dict)
