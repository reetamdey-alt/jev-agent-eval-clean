"""Run result schemas: transport records, per-case results, run manifest (spec 11.3, 33, 36)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CaseStatus(StrEnum):
    SUCCESS = "success"
    TRANSPORT_ERROR = "transport_error"
    SCHEMA_ERROR = "schema_error"
    SCORING_ERROR = "scoring_error"


class FailureCategory(StrEnum):
    """Failure taxonomy (spec section 33)."""

    TRANSPORT_ERROR = "transport_error"
    RATE_LIMITED = "rate_limited"
    PROVIDER_5XX = "provider_5xx"
    SCHEMA_ERROR = "schema_error"
    MISSING_ANSWER = "missing_answer"
    INVALID_PROBABILITY = "invalid_probability"
    INVALID_CHOICE = "invalid_choice"
    WRONG_LABEL = "wrong_label"
    WRONG_SCORE = "wrong_score"
    CONFIDENCE_MISCALIBRATION = "confidence_miscalibration"
    THRESHOLD_FAILURE = "threshold_failure"
    CONTEXT_SENSITIVITY = "context_sensitivity"
    PARAPHRASE_INSTABILITY = "paraphrase_instability"
    QUESTION_INTERFERENCE = "question_interference"
    PROMPT_INJECTION_FAILURE = "prompt_injection_failure"
    ANNOTATION_ERROR = "annotation_error"
    BENCHMARK_ADAPTER_ERROR = "benchmark_adapter_error"


class TransportRecord(BaseModel):
    """Everything we know about one request/response exchange (spec 11.3)."""

    request_start: str
    request_end: str | None = None
    connect_time_ms: float | None = None
    latency_ms: float | None = None
    http_status: int | None = None
    network_error_class: str | None = None
    retry_count: int = 0
    retry_history: list[dict[str, Any]] = Field(default_factory=list)
    parse_error: str | None = None
    server_model: str | None = None
    from_cache: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None


class CaseResult(BaseModel):
    """Result of evaluating one canonical case."""

    case_id: str
    dataset: str
    dataset_version: str
    capability: str
    repeat_index: int = 0
    status: CaseStatus
    request_hash: str | None = None
    question_ids: list[str] = Field(default_factory=list)
    # question_id -> parsed answer dict (selected label, probabilities, ...)
    predictions: dict[str, Any] = Field(default_factory=dict)
    # question_id -> correctness booleans for gold-scored questions
    correct: dict[str, bool] = Field(default_factory=dict)
    failure_categories: list[FailureCategory] = Field(default_factory=list)
    error_detail: str | None = None
    transport: TransportRecord | None = None
    raw_response: dict[str, Any] | None = None
    # Slice dimensions (spec 32) captured for aggregation.
    difficulty: str | None = None
    adversarial: bool = False
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    state_chars: int = 0
    # Generated variant (spec 64 invariants 1-2): included in run results
    # for counterfactual/metamorphic pair analysis, never counted in
    # primary accuracy or sample counts.
    is_variant: bool = False


class RunManifest(BaseModel):
    """Self-describing run identity (spec section 36). Never stores API keys."""

    run_id: str
    created_at: str
    evaluator_version: str
    git_commit: str | None = None
    python_version: str
    platform: str
    model: str
    provider_url: str
    dataset_manifest: str
    dataset_hash: str
    config_hash: str
    seed: int
    repeats: int
    cache_enabled: bool
    suite: str | None = None
    completed: bool = False
    case_count: int = 0


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
