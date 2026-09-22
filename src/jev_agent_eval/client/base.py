"""Provider protocol (spec section 48): JEV is the first implementation.

A provider maps a canonical inference request to an inference response,
recording transport telemetry. The evaluation core never talks to HTTP
directly, which keeps the runner testable and allows comparing other
structured-decision models with identical cases.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from jev_agent_eval.schemas.request import JEVRequest
from jev_agent_eval.schemas.response import JEVResponse


class InferenceResult(BaseModel):
    """Outcome of one inference call, including transport telemetry."""

    response: JEVResponse | None = None
    raw_body: str | None = None
    # Body of a non-200 response: the server's own reason (e.g. an auth
    # error message), kept for error_detail in errors.jsonl so a failed
    # run is diagnosable from artifacts alone.
    error_body: str | None = None
    http_status: int | None = None
    network_error_class: str | None = None
    parse_error: str | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    retry_history: list[dict[str, Any]] = Field(default_factory=list)
    from_cache: bool = False


@runtime_checkable
class InferenceProvider(Protocol):
    """Generic structured-decision inference provider."""

    name: str

    def infer(self, request: JEVRequest) -> InferenceResult:
        """Execute one inference request synchronously."""
        ...

    async def infer_async(self, request: JEVRequest) -> InferenceResult:
        """Execute one inference request asynchronously."""
        ...
