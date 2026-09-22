"""BFCL adapter (spec sections 4.1, 57).

Uses Berkeley Function Calling Leaderboard scenarios to construct JEV
meta-decisions (should a tool be called, is tool X relevant, is the failure
recoverable), never to ask JEV to emit function calls. Original BFCL ground
truth stays outside the JEV input.

Upstream: https://github.com/ShishirPatil/gorilla (Berkeley Function Calling
Leaderboard). Revision is pinned per dataset snapshot.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jev_agent_eval.datasets.base import DatasetMetadata, RawCase
from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer, GoldProvenance
from jev_agent_eval.utils.jsonl import iter_jsonl


class BFCLAdapter:
    name = "bfcl"
    version = "1.0"

    # Pinned upstream revision; update deliberately and record in manifest.
    UPSTREAM_REPO = "https://github.com/ShishirPatil/gorilla"
    UPSTREAM_REVISION = "main"
    UPSTREAM_TAG = "bfcl-v4"

    def __init__(self, data_path: str, dataset_version: str = "1.0.0"):
        self.data_path = data_path
        self.dataset_version = dataset_version

    def download(self) -> None:
        # Downloads are handled by scripts/download_benchmarks.py so that the
        # adapter itself stays offline-safe in CI.
        raise NotImplementedError(
            "use scripts/download_benchmarks.py to pin and fetch BFCL data"
        )

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=self.name,
            version=self.version,
            upstream_repo=self.UPSTREAM_REVISION and self.UPSTREAM_REPO,
            upstream_revision=self.UPSTREAM_TAG,
            license="Apache-2.0 (upstream)",
            transformation_version=self.version,
        )

    def enumerate_cases(self) -> Iterable[RawCase]:
        for i, obj in enumerate(iter_jsonl(self.data_path)):
            # BFCL records carry 'question' plus optional function definitions,
            # execution results, and initial/config states (V3/V4).
            question = obj.get("question") or obj.get("user_message") or ""
            yield RawCase(
                source_id=f"bfcl-{i:06d}",
                data={
                    "question": question,
                    "functions": obj.get("functions", []),
                    "initial_config": obj.get("initial_config", {}),
                    "execution_result": obj.get("execution_result"),
                    "behavior": obj.get("behavior"),  # V3/V4 multi-turn behavior tag
                    "ground_truth": obj.get("ground_truth"),  # expected calls (or no-call marker)
                },
            )

    def transform(self, raw: RawCase) -> list[CanonicalCase]:
        data: dict[str, Any] = raw.data
        question = str(data.get("question", ""))[:4000]
        functions = data.get("functions") or []
        tool_names = ", ".join(f.get("name", "?") for f in functions[:20]) or "none"
        state = (
            "User query to a coding assistant:\n"
            f"{question}\n\n"
            f"Available tools: {tool_names}\n"
        )
        # Execution outcome is only used to derive gold; it is kept OUT of the
        # state when it would reveal whether a call succeeded.
        exec_result = data.get("execution_result")
        cases: list[CanonicalCase] = []

        # Meta-question 1: does this query require a tool call at all?
        # Upstream ground_truth is authoritative when present (BFCL irrelevance
        # records carry function lists yet expect NO call); the function/question
        # presence heuristic is only a fallback and is rule-derived, not
        # benchmark gold.
        ground_truth = data.get("ground_truth")
        if ground_truth is not None:
            needs_tool = bool(ground_truth)
            provenance = GoldProvenance.BENCHMARK_GOLD
        else:
            needs_tool = bool(functions) and bool(question.strip())
            provenance = GoldProvenance.RULE_DERIVED
        cases.append(
            CanonicalCase(
                case_id=f"{raw.source_id}-tool-needed",
                dataset="bfcl-derived",
                dataset_version=self.dataset_version,
                capability="tool_relevance",
                tags=["bfcl", "function_calling"],
                state=state,
                questions={
                    "tool_relevant": {
                        "type": "noul",
                        "instructions": (
                            "Does answering this query require calling one of the "
                            "available tools?"
                        ),
                    }
                },
                gold={
                    "tool_relevant": GoldAnswer(
                        type="noul",
                        value=1 if needs_tool else 0,
                        provenance=provenance,
                    )
                },
                metadata={
                    "source": "bfcl",
                    "source_ref": raw.source_id,
                    "upstream_revision": self.UPSTREAM_TAG,
                },
            )
        )

        # Meta-question 2 (only when execution evidence exists): is the latest
        # failure recoverable?
        if isinstance(exec_result, str) and ("error" in exec_result.lower() or "exception" in exec_result.lower()):
            recoverable = "argument" in exec_result.lower() or "type" in exec_result.lower()
            cases.append(
                CanonicalCase(
                    case_id=f"{raw.source_id}-recoverable",
                    dataset="bfcl-derived",
                    dataset_version=self.dataset_version,
                    capability="stuck_detection",
                    tags=["bfcl", "error_recovery"],
                    state=(
                        "A coding agent called a tool in response to a user query "
                        "and received an error.\n\n"
                        f"Tool error output:\n{exec_result[:2000]}"
                    ),
                    questions={
                        "recoverable": {
                            "type": "noul",
                            "instructions": (
                                "Is this tool failure plausibly recoverable by "
                                "adjusting the arguments and retrying?"
                            ),
                        }
                    },
                    gold={
                        "recoverable": GoldAnswer(
                            type="noul",
                            value=1 if recoverable else 0,
                            provenance=GoldProvenance.RULE_DERIVED,
                        )
                    },
                    metadata={
                        "source": "bfcl",
                        "source_ref": raw.source_id,
                        "upstream_revision": self.UPSTREAM_TAG,
                    },
                )
            )
        return cases
