"""Run manifest store: registry of runs for listing and comparison (spec 36, 54)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestStore:
    """Indexes run directories under a runs root for listing and comparison."""

    def __init__(self, runs_root: str | Path):
        self.runs_root = Path(runs_root)

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        if not self.runs_root.exists():
            return runs
        for d in sorted(self.runs_root.iterdir()):
            mf = d / "manifest.json"
            if mf.exists():
                try:
                    with open(mf, encoding="utf-8") as f:
                        runs.append(json.load(f))
                except json.JSONDecodeError:
                    continue
        return runs

    def load_summary(self, run_dir: str | Path) -> dict[str, Any]:
        with open(Path(run_dir) / "summary.json", encoding="utf-8") as f:
            return json.load(f)

    def load_metrics(self, run_dir: str | Path) -> dict[str, Any]:
        with open(Path(run_dir) / "metrics.json", encoding="utf-8") as f:
            return json.load(f)

    def load_scored(self, run_dir: str | Path) -> list[dict[str, Any]]:
        path = Path(run_dir) / "scored.jsonl"
        records: list[dict[str, Any]] = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        return records
