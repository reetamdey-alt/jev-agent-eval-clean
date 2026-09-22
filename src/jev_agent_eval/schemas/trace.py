"""Normalized trace schema (v2 spec section 12).

Framework-independent representation of one step of a coding-agent session.
Importers for different harnesses normalize into this shape; the episode
builder turns sequences of steps into JEV evaluation checkpoints.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TraceStepType(StrEnum):
    """Kind of step, used for checkpoint selection (v2 spec 13.1)."""

    USER_REQUEST = "user_request"
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_EDIT = "file_edit"
    TEST_RUN = "test_run"
    RETRY = "retry"
    USER_CORRECTION = "user_correction"
    MODEL_SWITCH = "model_switch"
    COMPACTION = "compaction"
    PERMISSION_DECISION = "permission_decision"
    STOP = "stop"


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultRecord(BaseModel):
    status: str = "success"  # success | error | timeout | denied
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None


class TestSummary(BaseModel):
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0


class GitState(BaseModel):
    dirty: bool | None = None
    diff_lines_added: int | None = None
    diff_lines_removed: int | None = None


class TraceStep(BaseModel):
    """One normalized step in an agent session (v2 spec 12)."""

    session_id: str
    task_id: str
    episode_id: str
    step: int
    timestamp: str | None = None
    step_type: TraceStepType | None = None
    user_request: str | None = None
    agent_message: str | None = None
    tool_call: ToolCallRecord | None = None
    tool_result: ToolResultRecord | None = None
    files_changed: list[str] = Field(default_factory=list)
    tests: TestSummary | None = None
    git: GitState | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    # Redacted/anonymized free-form extras from the source harness.
    extras: dict[str, Any] = Field(default_factory=dict)


class TraceSession(BaseModel):
    """A full normalized session: ordered steps plus provenance."""

    session_id: str
    task_id: str
    episode_id: str
    steps: list[TraceStep] = Field(default_factory=list)
    # Ingestion provenance (v2 spec 29): source classification + hash.
    source_class: str = "internal_trace"  # trace | benchmark | synthetic
    source_ref: str | None = None
    anonymized: bool = True
    trace_hash: str | None = None

    def compute_hash(self) -> str:
        """Stable content hash over the canonical session dump."""
        from jev_agent_eval.utils.hashing import sha256_hex

        self.trace_hash = "sha256:" + sha256_hex(self.model_dump_json())
        return self.trace_hash


class TraceIngestionReport(BaseModel):
    """Result of ingesting a trace file (v2 spec 29)."""

    n_sessions: int = 0
    n_steps: int = 0
    n_rejected: int = 0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    secret_hits: list[str] = Field(default_factory=list)
    source_classes: dict[str, int] = Field(default_factory=dict)
