#!/usr/bin/env python
"""Compare two run directories and print a regression report (spec 48, 54).

Usage:
    uv run python scripts/compare_runs.py <baseline_run_dir> <candidate_run_dir> [--md out.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--md", type=Path, default=None, help="Write markdown here")
    args = parser.parse_args()

    from jev_agent_eval.reports.compare import compare_runs, render_comparison_markdown

    for d in (args.baseline, args.candidate):
        if not (d / "manifest.json").exists():
            print(f"not a run directory: {d}", file=sys.stderr)
            return 3

    comparison = compare_runs(args.baseline, args.candidate)
    md = render_comparison_markdown(comparison)
    print(md)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md, encoding="utf-8")
        print(f"\nwrote {args.md}")

    # Exit 1 when a regression is detected (quality-gate semantics).
    # compare_runs() keys per-capability results under "capabilities" —
    # a wrong key here would silently turn every regression into exit 0.
    regressions = [
        name
        for name, c in comparison.get("capabilities", {}).items()
        if c.get("classification") == "regressed"
    ]
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
