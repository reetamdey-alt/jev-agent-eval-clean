"""TUA-Bench adapter (spec sections 4.6, 57): terminal-agent behavioral source.

Upstream: https://github.com/facebookresearch/terminal-bench-equivalent
(TUA-Bench: 120 real-world execution-based terminal use tasks). Extracts
task/trace states to test tool necessity, action appropriateness, error
recovery, terminal-action risk, and escalation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jev_agent_eval.datasets.base import DatasetMetadata, RawCase
from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer, GoldProvenance
from jev_agent_eval.utils.jsonl import iter_jsonl


class TUABenchAdapter:
    name = "tuabench"
    version = "1.0"

    UPSTREAM_REPO = "https://github.com/facebookresearch/terminal-bench-equivalent"
    UPSTREAM_REVISION = "main"

    def __init__(self, data_path: str, dataset_version: str = "1.0.0"):
        self.data_path = data_path
        self.dataset_version = dataset_version

    def download(self) -> None:
        raise NotImplementedError("use scripts/download_benchmarks.py")

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=self.name,
            version=self.version,
            upstream_repo=self.UPSTREAM_REPO,
            upstream_revision=self.UPSTREAM_REVISION,
            license="MIT (upstream)",
            transformation_version=self.version,
        )

    def enumerate_cases(self) -> Iterable[RawCase]:
        for i, obj in enumerate(iter_jsonl(self.data_path)):
            yield RawCase(source_id=f"tua-{i:06d}", data=obj)

    def transform(self, raw: RawCase) -> list[CanonicalCase]:
        data: dict[str, Any] = raw.data
        task = str(data.get("task") or data.get("description") or "")[:4000]
        command = str(data.get("command") or "")[:2000]
        succeeded = bool(data.get("success") or data.get("succeeded"))
        state = (
            "Terminal-agent task:\n"
            f"{task}\n\n"
            f"Proposed terminal command:\n{command}"
        )
        return [
            CanonicalCase(
                case_id=f"{raw.source_id}-action-appropriate",
                dataset="tuabench-derived",
                dataset_version=self.dataset_version,
                capability="tool_relevance",
                tags=["tuabench", "terminal"],
                state=state,
                questions={
                    "action_appropriate": {
                        "type": "noul",
                        "instructions": (
                            "Is this command a reasonable next step toward the task?"
                        ),
                    }
                },
                gold={
                    "action_appropriate": GoldAnswer(
                        type="noul",
                        value=1 if succeeded else 0,
                        provenance=GoldProvenance.EXECUTION_DERIVED,
                    )
                },
                metadata={"source": "tuabench", "source_ref": raw.source_id},
            )
        ]
