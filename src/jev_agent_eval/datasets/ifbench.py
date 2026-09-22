"""IFBench adapter (spec section 4.4): optional precision instruction-following source.

Upstream: https://github.com/allenai/IFBench (challenging, verifiable
instruction constraints including out-of-distribution constraints).
Transformation mirrors IFEval's, with an additional severity question.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jev_agent_eval.datasets.base import DatasetMetadata, RawCase
from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer, GoldProvenance
from jev_agent_eval.utils.jsonl import iter_jsonl


class IFBenchAdapter:
    name = "ifbench"
    version = "1.0"

    UPSTREAM_REPO = "https://github.com/allenai/IFBench"
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
            license="Apache-2.0 (upstream)",
            transformation_version=self.version,
        )

    def enumerate_cases(self) -> Iterable[RawCase]:
        for i, obj in enumerate(iter_jsonl(self.data_path)):
            yield RawCase(source_id=f"ifbench-{i:06d}", data=obj)

    def transform(self, raw: RawCase) -> list[CanonicalCase]:
        data: dict[str, Any] = raw.data
        prompt = str(data.get("prompt") or data.get("instruction") or "")[:4000]
        response = str(data.get("response") or "")[:4000]
        violations = data.get("violations") or []
        state = (
            "User instruction with precise verifiable constraints:\n"
            f"{prompt}\n\n"
            "Assistant's proposed response:\n"
            f"{response}"
        )
        return [
            CanonicalCase(
                case_id=f"{raw.source_id}-constraint-violation",
                dataset="ifbench-derived",
                dataset_version=self.dataset_version,
                capability="instruction_compliance",
                tags=["ifbench", "instruction_following", "ood"],
                state=state,
                questions={
                    "violates_constraint": {
                        "type": "noul",
                        "instructions": (
                            "Does the proposed response violate an explicit "
                            "constraint stated in the user instruction?"
                        ),
                    }
                },
                gold={
                    "violates_constraint": GoldAnswer(
                        type="noul",
                        value=1 if violations else 0,
                        provenance=GoldProvenance.BENCHMARK_GOLD,
                    )
                },
                metadata={"source": "ifbench", "source_ref": raw.source_id},
            )
        ]
