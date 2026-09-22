#!/usr/bin/env python
"""Build the `jev-agent-core` internal dataset from deterministic generators.

Usage: uv run python scripts/build_internal_cases.py [--n-per-capability N]
Writes datasets/internal/*.jsonl plus manifest with integrity hashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_agent_eval.datasets.synthetic import (  # noqa: E402
    ClaimEvidenceGenerator,
    CompletionGenerator,
    InjectionGenerator,
    ToolRiskGenerator,
)
from jev_agent_eval.utils.hashing import sha256_file  # noqa: E402
from jev_agent_eval.utils.jsonl import write_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ROOT / "datasets" / "internal"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-capability", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    generators = {
        "tool_risk.jsonl": (ToolRiskGenerator(), args.n_per_capability),
        "goal_completion.jsonl": (CompletionGenerator(), args.n_per_capability),
        "claim_evidence.jsonl": (ClaimEvidenceGenerator(), args.n_per_capability),
        "injection_resistance.jsonl": (InjectionGenerator(), args.n_per_capability),
    }
    INTERNAL.mkdir(parents=True, exist_ok=True)
    capability_of = {
        "tool_risk.jsonl": "tool_risk",
        "goal_completion.jsonl": "goal_completion",
        "claim_evidence.jsonl": "claim_evidence",
        "injection_resistance.jsonl": "injection_resistance",
    }

    integrity: dict[str, str] = {}
    cases_blocks = []
    for fname, (gen, n) in generators.items():
        path = INTERNAL / fname
        cases = list(gen.generate(args.seed, n))
        write_jsonl(path, (c.model_dump() for c in cases))
        integrity[fname] = sha256_file(path)
        cases_blocks.append(
            f"  - path: internal/{fname}\n"
            f"    capability: {capability_of[fname]}\n"
            f"    count: {len(cases)}"
        )
        print(f"{fname}: {len(cases)} cases")

    manifest = ROOT / "datasets" / "manifests" / "core-v1.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "name: jev-agent-core\n"
        "version: 1.0.0\n"
        "schema_version: '1.0'\n"
        "license: internal\n"
        "cases:\n" + "\n".join(cases_blocks) + "\n"
        "sources:\n"
        "  - name: synthetic\n"
        "    reference: deterministic-generators\n"
        "integrity:\n" + "".join(f'  internal/{k}: "{v}"\n' for k, v in integrity.items()),
        encoding="utf-8",
    )
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
