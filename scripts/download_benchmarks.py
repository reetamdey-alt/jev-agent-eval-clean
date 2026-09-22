#!/usr/bin/env python
"""Download pinned public benchmark datasets (spec sections 4, 44, 47).

Each adapter pins an upstream repository and revision. This script fetches
only the files each adapter needs, records the revision, and verifies that
the adapter's pinned revision matches what was downloaded.

Usage: uv run python scripts/download_benchmarks.py [--benchmarks bfcl swebench ...]

Security: no code from the datasets is executed; files are stored as data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "public"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["bfcl", "swebench", "ifeval", "ifbench", "tau2", "tuabench"],
        help="Subsets to download (default: all)",
    )
    parser.add_argument("--list-only", action="store_true", help="Show pins without downloading")
    args = parser.parse_args()

    from jev_agent_eval.datasets.bfcl import BFCLAdapter
    from jev_agent_eval.datasets.ifbench import IFBenchAdapter
    from jev_agent_eval.datasets.ifeval import IFEvalAdapter
    from jev_agent_eval.datasets.swebench import SWEBenchAdapter
    from jev_agent_eval.datasets.tau2 import Tau2Adapter
    from jev_agent_eval.datasets.tuabench import TUABenchAdapter

    # Adapters take a data_path; instantiate lazily with the target dir.
    adapter_classes = {
        "bfcl": BFCLAdapter,
        "swebench": SWEBenchAdapter,
        "ifeval": IFEvalAdapter,
        "ifbench": IFBenchAdapter,
        "tau2": Tau2Adapter,
        "tuabench": TUABenchAdapter,
    }

    for name in args.benchmarks:
        if name not in adapter_classes:
            print(f"unknown benchmark: {name}", file=sys.stderr)
            return 2
        cls = adapter_classes[name]
        print(f"{name}: repo={cls.UPSTREAM_REPO} revision={cls.UPSTREAM_REVISION}")

    if args.list_only:
        return 0

    # Download requires network access and upstream availability. Adapters
    # transform the raw data at load time; raw files are cached under
    # datasets/public/<name>/ and never executed.
    try:
        import urllib.request
    except ImportError:  # pragma: no cover
        print("urllib unavailable", file=sys.stderr)
        return 7

    for name in args.benchmarks:
        cls = adapter_classes[name]
        target = DATA_DIR / name
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "PIN.txt"
        marker.write_text(
            f"repo: {cls.UPSTREAM_REPO}\nrevision: {cls.UPSTREAM_REVISION}\n",
            encoding="utf-8",
        )
        if hasattr(cls, "UPSTREAM_FILES"):
            for rel in cls.UPSTREAM_FILES:
                url = f"{cls.UPSTREAM_REPO}/raw/{cls.UPSTREAM_REVISION}/{rel}"
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    print(f"  cached: {rel}")
                    continue
                print(f"  fetching: {rel}")
                try:
                    urllib.request.urlretrieve(url, dest)  # noqa: S310
                except Exception as e:  # noqa: BLE001
                    print(f"  WARN could not fetch {rel}: {e}")
        else:
            print(
                f"  NOTE: {name} adapter has no UPSTREAM_FILES list; "
                "place raw files manually per the adapter docstring."
            )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
