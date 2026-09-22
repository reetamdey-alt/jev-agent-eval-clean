"""Episode / trajectory schema (v2 spec section 13).

An episode chains JEV checkpoints across the states of one agent session:
initial -> pre-tool -> post-tool-success/failure -> ... -> pre-stop.
Episode gold describes trajectory-level outcomes (task solved, unsafe
action attempted, premature stop, ...).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CheckpointKind(StrEnum):
    """Episode checkpoint states (v2 spec 13.1)."""

    INITIAL = "initial"
    PRE_TOOL = "pre_tool"
    POST_TOOL_SUCCESS = "post_tool_success"
    POST_TOOL_FAILURE = "post_tool_failure"
    POST_EDIT = "post_edit"
    POST_TEST = "post_test"
    POST_RETRY = "post_retry"
    POST_USER_CORRECTION = "post_user_correction"
    PRE_STOP = "pre_stop"
    POST_STOP = "post_stop"


class EpisodeCheckpoint(BaseModel):
    """One JEV evaluation point inside an episode."""

    kind: CheckpointKind
    step_index: int
    # Canonical state string handed to JEV at this checkpoint.
    state: str
    # Question ids asked at this checkpoint (pack resolved at build time).
    question_ids: list[str] = Field(default_factory=list)
    # Gold control decision for this checkpoint, keyed by question id.
    gold: dict[str, Any] = Field(default_factory=dict)


class EpisodeGold(BaseModel):
    """Trajectory-level gold outcomes (v2 spec 13.2)."""

    task_solved: bool
    unsafe_action_attempted: bool = False
    unnecessary_escalation: bool = False
    premature_stop: bool = False
    delayed_stop: bool = False
    unnecessary_tool_use: bool = False
    successful_recovery: bool = False
    final_claim_correct: bool | None = None
    n_steps: int = 0
    provenance: str = "rule_derived"


class Episode(BaseModel):
    """A trajectory evaluation unit: ordered checkpoints + outcome gold."""

    episode_id: str
    dataset: str
    dataset_version: str
    capability: str = "trajectory_control"
    task_description: str
    checkpoints: list[EpisodeCheckpoint] = Field(default_factory=list)
    gold: EpisodeGold | None = None
    # Horizon bucket (v2 spec 71): 5-10 / 10-25 / 25-50 / 50+ steps.
    horizon_bucket: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeStepResult(BaseModel):
    """JEV decision observed at one checkpoint during execution."""

    checkpoint_kind: CheckpointKind
    step_index: int
    question_id: str
    prediction: Any = None
    probability: float | None = None
    confidence: float | None = None
    correct: bool | None = None
    latency_ms: float | None = None


class EpisodeResult(BaseModel):
    """Full evaluation result for one episode."""

    episode_id: str
    n_checkpoints: int = 0
    step_results: list[EpisodeStepResult] = Field(default_factory=list)
    # Derived episode-level rates (v2 spec 13.3).
    decision_agreement: float | None = None
    unsafe_action_rate: float | None = None
    premature_stop_rate: float | None = None
    delayed_stop_rate: float | None = None
    missed_escalation_rate: float | None = None
    unnecessary_escalation_rate: float | None = None
    unnecessary_tool_rate: float | None = None
    recovery_success_rate: float | None = None
    unsupported_success_acceptance_rate: float | None = None
    mean_decision_latency_ms: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    status: str = "success"
