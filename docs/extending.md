# Extending JEV-Agent-Eval

This guide explains how to add a new capability, dataset, scorer, question
pack, or provider. Everything in the framework is a registered plugin behind
a small protocol, so each extension is one module plus one registration.

## Adding a new capability (question pack)

A capability is a mapping from a `CanonicalCase` to typed JEV questions and
gold answers. Packs are pure functions of the case — no network, no clocks,
no randomness.

1. Create `src/jev_agent_eval/questions/<capability>.py`:

```python
from dataclasses import dataclass, field

from jev_agent_eval.questions.base import QuestionPack, QuestionPackResult
from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer, GoldProvenance
from jev_agent_eval.schemas.request import JEVQuestion


@dataclass
class MyCapabilityQuestionPack(QuestionPackResult): ...


class MyQuestionPack:
    """Implements the QuestionPack protocol."""

    name = "my_capability"
    version = "1.0"

    def build(self, case: CanonicalCase) -> QuestionPackResult:
        # Read the case's pre-baked questions if present, else author them here.
        questions = {
            "my_decision": JEVQuestion(
                type="noul",
                instructions="Decide X given the state.",
            )
        }
        gold = {
            k: GoldAnswer(type="noul", value=v.value, provenance=v.provenance)
            for k, v in case.gold.items()
            if k in questions
        }
        return QuestionPackResult(
            questions=questions,
            gold=gold,
            capability="my_capability",
            tags=list(case.tags),
        )
```

2. Register it in `src/jev_agent_eval/questions/__init__.py` (add to
   `_CAPABILITY_PACKS`).

3. Author wording carefully (spec sections 62–63):
   - neutral phrasing, no cues to the gold label
   - the question must be answerable from the state alone
   - never mention the gold, dataset name, or expected distribution

4. Add cases for the capability to a dataset (below), or extend
   `src/jev_agent_eval/datasets/synthetic.py` with a generator and rebuild
   via `scripts/build_internal_cases.py`.

## Adding a new dataset

Datasets enter the framework as `CanonicalCase` JSONL files referenced by a
manifest with integrity hashes.

1. **Internal / synthetic data**: write a deterministic generator (seeded,
   reproducible) in `src/jev_agent_eval/datasets/synthetic.py`, then run
   `uv run python scripts/build_internal_cases.py` to emit
   `datasets/internal/<name>.jsonl` and refresh the manifest.

2. **Public benchmark**: create `src/jev_agent_eval/datasets/<name>.py`
   implementing the `DatasetAdapter` protocol:

```python
class MyAdapter:
    name = "my_benchmark"
    UPSTREAM_REPO = "https://github.com/org/repo"
    UPSTREAM_REVISION = "v1.2.3"  # always pin an exact revision

    def __init__(self, data_path: str, dataset_version: str = "1.0.0"): ...

    def download(self) -> None:
        # Fetches are handled by scripts/download_benchmarks.py so the
        # adapter stays offline-safe in CI; raise NotImplementedError here.
        ...

    def metadata(self) -> DatasetMetadata: ...

    def enumerate_cases(self) -> Iterable[RawCase]:
        # Stream upstream records (JSONL) as RawCase(source_id=..., data=...).
        ...

    def transform(self, raw: RawCase) -> list[CanonicalCase]:
        # Transform to META-DECISION cases only (spec 57): we evaluate JEV's
        # decisions about the agent, never the agent's raw output quality.
        # Gold labels never enter `state` — the loader's leakage validator
        # (FORBIDDEN_STATE_KEYS) rejects them at load time.
        ...
```

3. Reference the JSONL in a manifest under `datasets/manifests/` and run
   `uv run jev-eval datasets validate <manifest>` — this checks schema,
   integrity, case invariants, and gold leakage.

## Adding a new scorer

Scorers live in `src/jev_agent_eval/scoring/` as pure functions over
prediction/gold arrays.

1. Add the function with a small dataclass result type.
2. Wire it into the per-question loop in `scoring/__init__.py`
   (`score_capability`) and into `reports/json_report.py::build_metrics`
   if it should appear in every report.
3. Add a golden test in `tests/unit/test_scoring.py` with hand-computed
   values.

## Adding a new provider

Providers implement the `InferenceProvider` protocol in
`client/base.py` (`infer` and `infer_async`, returning `InferenceResult`).

- Credentials must come from an environment variable only — never
  constructor arguments, config files, or datasets.
- Bound retries to transient conditions (429/5xx/timeouts) with exponential
  backoff and jitter.
- Cap response body size; parse tolerantly (`extra="allow"`).
- Record latency, retries, and error class so reports stay comparable.

## Adding a new report format

Add a renderer module in `src/jev_agent_eval/reports/` that consumes the
metrics dict produced by `build_metrics`, then expose it in the `report`
CLI command.

## Checklist for every extension

- [ ] Deterministic: same inputs → same outputs, no wall-clock or RNG without a seed
- [ ] Gold labels never appear in `state` or question text
- [ ] Unit test with hand-computed expectations
- [ ] Docs updated (this file / README)
- [ ] `uv run pytest -q`, `uv run ruff check src tests scripts`, `uv run mypy src` all pass
