"""Benchmark adapter contract (spec sections 56, 58).

Adapters never write into evaluation reports; they only produce canonical
cases. The synthetic CaseGenerator protocol is also defined here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from jev_agent_eval.schemas.case import CanonicalCase


class RawCase(BaseModel):
    """Minimally-typed upstream record before transformation."""

    source_id: str
    data: dict[str, Any] = Field(default_factory=dict)


class DatasetMetadata(BaseModel):
    name: str
    version: str
    upstream_repo: str | None = None
    upstream_revision: str | None = None
    license: str | None = None
    transformation_version: str | None = None
    download_timestamp: str | None = None


@runtime_checkable
class DatasetAdapter(Protocol):
    name: str
    version: str

    def download(self) -> None:
        """Fetch/pin upstream data (no-op for purely local adapters)."""
        ...

    def enumerate_cases(self) -> Iterable[RawCase]:
        ...

    def transform(self, raw: RawCase) -> list[CanonicalCase]:
        ...

    def metadata(self) -> DatasetMetadata:
        ...


@runtime_checkable
class CaseGenerator(Protocol):
    """Deterministic synthetic generator (spec section 58). All generation is seeded."""

    name: str
    version: str

    def generate(self, seed: int, n: int) -> Iterable[CanonicalCase]:
        ...
