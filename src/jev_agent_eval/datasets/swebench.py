"""SWE-bench adapter (spec sections 4.2, 57).

Uses task statements and repository metadata to build coding-agent control
plane decisions. The gold patch and test patch are never included in the JEV
state; they exist only in the upstream record and are used solely to derive
structural gold labels (e.g. multi-file vs single-file).

Upstream: https://github.com/SWE-bench/SWE-bench
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jev_agent_eval.datasets.base import DatasetMetadata, RawCase
from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer, GoldProvenance


class SWEBenchAdapter:
    name = "swebench"
    version = "1.0"

    UPSTREAM_REPO = "https://github.com/SWE-bench/SWE-bench"
    UPSTREAM_REVISION = "v2.0.0"
    # Documented sizes: Full 2294, Lite 534, Verified 500 instances.

    def __init__(self, data_path: str, dataset_version: str = "1.0.0"):
        self.data_path = data_path
        self.dataset_version = dataset_version

    def download(self) -> None:
        raise NotImplementedError(
            "use scripts/download_benchmarks.py to pin and fetch SWE-bench data"
        )

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=self.name,
            version=self.version,
            upstream_repo=self.UPSTREAM_REPO,
            upstream_revision=self.UPSTREAM_REVISION,
            license="MIT (upstream dataset MIT/other per instance)",
            transformation_version=self.version,
        )

    def enumerate_cases(self) -> Iterable[RawCase]:
        from jev_agent_eval.utils.jsonl import iter_jsonl

        for i, obj in enumerate(iter_jsonl(self.data_path)):
            yield RawCase(source_id=str(obj.get("instance_id") or f"swe-{i:06d}"), data=obj)

    def transform(self, raw: RawCase) -> list[CanonicalCase]:
        data: dict[str, Any] = raw.data
        statement = str(data.get("problem_statement", ""))[:6000]
        repo = str(data.get("repo", "unknown"))
        # Gold patch used ONLY to derive structural labels; never enters state.
        patch = str(data.get("patch") or "")
        n_files = len([ln for ln in patch.split("\n") if ln.startswith("--- ")]) if patch else 0

        state = (
            f"Repository: {repo}\n\n"
            f"Issue:\n{statement}"
        )
        cases = [
            CanonicalCase(
                case_id=f"{raw.source_id}-needs-code-change",
                dataset="swebench-derived",
                dataset_version=self.dataset_version,
                capability="task_routing",
                tags=["swebench", "coding"],
                state=state,
                questions={
                    "requires_code_modification": {
                        "type": "noul",
                        "instructions": (
                            "Does resolving this issue require modifying source code "
                            "(as opposed to docs-only or no change)?"
                        ),
                    }
                },
                gold={
                    "requires_code_modification": GoldAnswer(
                        type="noul",
                        # SWE-bench instances are code changes by construction;
                        # a missing patch only means the record is truncated.
                        value=1,
                        provenance=GoldProvenance.BENCHMARK_GOLD,
                    )
                },
                metadata={"source": "swebench", "source_ref": raw.source_id},
            ),
            CanonicalCase(
                case_id=f"{raw.source_id}-multi-file",
                dataset="swebench-derived",
                dataset_version=self.dataset_version,
                capability="task_routing",
                tags=["swebench", "planning"],
                state=state,
                questions={
                    "likely_multi_file": {
                        "type": "noul",
                        "instructions": (
                            "Is this change likely to require edits across more than "
                            "one source file?"
                        ),
                    }
                },
                gold={
                    "likely_multi_file": GoldAnswer(
                        type="noul",
                        value=1 if n_files > 1 else 0,
                        provenance=GoldProvenance.EXECUTION_DERIVED,
                    )
                },
                metadata={"source": "swebench", "source_ref": raw.source_id},
            ),
        ]
        return cases
