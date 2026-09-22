"""Configuration loading and validation (spec section 8).

Credentials never appear here; they come exclusively from the JEV_API_KEY
environment variable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _ValidatedModel(BaseModel):
    """Base config model: field constraints re-check on every assignment.

    CLI overrides mutate the loaded config in place (`cfg.run.repeats = 0`);
    without validate_assignment those assignments bypass the Field
    constraints entirely, so an invalid override silently runs with the
    value instead of failing as a config error (exit 2).
    """

    model_config = ConfigDict(validate_assignment=True)


class ProviderConfig(_ValidatedModel):
    base_url: str = "https://grid.ai.juspay.net/v1/systemone"
    model: str = "jev-latest"
    timeout_s: float = 30
    connect_timeout_s: float = 5
    read_timeout_s: float = 30
    max_retries: int = 4
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    concurrency: int = 8
    requests_per_second: float | None = 10


class RunConfig(_ValidatedModel):
    seed: int = 20260920
    # repeats=0 evaluates nothing and silently reports an empty success;
    # that is a config typo, not a run.
    repeats: int = Field(default=1, ge=1)
    fail_fast: bool = False
    cache: bool = True
    save_raw: bool = True
    redact_secrets: bool = True


class ScoringConfig(_ValidatedModel):
    # An impossible or nonsensical scoring setup (negative samples, a
    # confidence level outside (0,1), zero bins) must fail at config load
    # (exit 2), not surface later as a crash or silently wrong statistics.
    bootstrap_samples: int = Field(default=2000, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    ece_bins: int = Field(default=10, ge=1)
    multiclass_ece: bool = True
    probability_epsilon: float = Field(default=0.001, gt=0.0)


class ReportsConfig(_ValidatedModel):
    output_dir: str = "reports"
    formats: list[str] = Field(default_factory=lambda: ["json", "markdown", "html"])


class ThresholdsConfig(_ValidatedModel):
    # Rates are proportions in [0, 1]; latency is positive milliseconds. A
    # negative threshold makes the gate unsatisfiable (0.0 > -1.0 fires on
    # every run) and would permanently red the pipeline as a "quality"
    # failure when it is really a config typo.
    transport_error_rate_max: float = Field(default=0.01, ge=0.0, le=1.0)
    schema_error_rate_max: float = Field(default=0.01, ge=0.0, le=1.0)
    p95_latency_ms_max: float = Field(default=1500, gt=0.0)


class PricingConfig(_ValidatedModel):
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None


class LimitsConfig(_ValidatedModel):
    max_state_chars: int = 200_000
    max_response_body_bytes: int = 10_485_760


class EvalConfig(_ValidatedModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    quality_gates: dict[str, dict[str, float]] = Field(default_factory=dict)


class ConfigError(Exception):
    pass


# Repo-level default config, used when no --config is given so that quality
# gates and thresholds always come from the shipped configuration rather
# than silently degrading to "no gates".
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> EvalConfig:
    """Load config from YAML, falling back to defaults for missing sections.

    With no explicit path, the repo's configs/default.yaml is loaded (quality
    gates must never silently vanish); if that file is absent, pure defaults
    apply (gates empty — appropriate for programmatic use, not CLI runs).
    """
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if path is not None and not p.exists():
        raise ConfigError(f"config file not found: {path}")
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {p}: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"config root must be a mapping: {p}")
    # Quality gates must never silently vanish: a suite config that omits
    # the quality_gates section (smoke/pr/nightly/... all configure rates
    # and thresholds but no gates) inherits the repo default gates instead
    # of running ungated — observed live when a smoke run printed "All
    # quality gates passed" + exit 0 while its own scorecard said FAIL.
    # An explicitly empty mapping (quality_gates: {}) opts out on purpose.
    if p != DEFAULT_CONFIG_PATH and "quality_gates" not in data and DEFAULT_CONFIG_PATH.exists():
        try:
            with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
                defaults = yaml.safe_load(f) or {}
            if isinstance(defaults, dict) and defaults.get("quality_gates"):
                data["quality_gates"] = defaults["quality_gates"]
        except yaml.YAMLError:
            pass  # unreadable defaults must not break loading the real config
    try:
        return EvalConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"invalid config: {e}") from e


def config_hash(config: EvalConfig) -> str:
    from jev_agent_eval.utils.hashing import sha256_hex

    return sha256_hex(config.model_dump_json(exclude_none=True))
