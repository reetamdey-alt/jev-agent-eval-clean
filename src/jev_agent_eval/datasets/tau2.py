"""tau2-bench adapter (spec sections 4.5, 57).

Converts conversational agent/tool/user interaction states into intent
classification, policy compliance, tool relevance, and escalation decisions.

Upstream: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jev_agent_eval.datasets.base import DatasetMetadata, RawCase
from jev_agent_eval.schemas.case import CanonicalCase, GoldAnswer, GoldProvenance
from jev_agent_eval.utils.jsonl import iter_jsonl


class Tau2Adapter:
    name = "tau2"
    version = "1.0"

    UPSTREAM_REPO = "https://github.com/sierra-research/tau2-bench"
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
            yield RawCase(source_id=f"tau2-{i:06d}", data=obj)

    def transform(self, raw: RawCase) -> list[CanonicalCase]:
        data: dict[str, Any] = raw.data

        def _render_conversation(value: Any) -> str:
            # Conversations arrive either as a list of message dicts or a plain
            # string; str() on a list would embed Python repr in the state.
            if isinstance(value, list):
                lines = []
                for msg in value:
                    if isinstance(msg, dict):
                        role = str(msg.get("role", "?"))
                        content = str(msg.get("content", ""))
                        lines.append(f"{role}: {content}")
                    else:
                        lines.append(str(msg))
                return "\n".join(lines)[:6000]
            return str(value or "")[:6000]

        conversation = _render_conversation(
            data.get("conversation") or data.get("messages") or ""
        )
        policy = str(data.get("policy") or "")[:3000]
        action = str(data.get("action") or "")[:2000]
        policy_compliant = bool(data.get("policy_compliant"))
        state = (
            "Conversation between a user and a service agent:\n"
            f"{conversation}\n\n"
            f"Agent policy:\n{policy}\n\n"
            f"Proposed next action:\n{action}"
        )
        return [
            CanonicalCase(
                case_id=f"{raw.source_id}-policy-compliance",
                dataset="tau2-derived",
                dataset_version=self.dataset_version,
                capability="instruction_compliance",
                tags=["tau2", "policy"],
                state=state,
                questions={
                    "policy_compliant": {
                        "type": "noul",
                        "instructions": (
                            "Is the proposed action compliant with the agent policy?"
                        ),
                    }
                },
                gold={
                    "policy_compliant": GoldAnswer(
                        type="noul",
                        value=1 if policy_compliant else 0,
                        provenance=GoldProvenance.BENCHMARK_GOLD,
                    )
                },
                metadata={"source": "tau2", "source_ref": raw.source_id},
            )
        ]
