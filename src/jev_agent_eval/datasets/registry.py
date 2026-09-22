"""Dataset registry: resolving manifests to loaders without touching the runner."""

from __future__ import annotations

from pathlib import Path

import yaml

from jev_agent_eval.datasets.local_jsonl import LocalJSONLDataset
from jev_agent_eval.schemas.case import CanonicalCase
from jev_agent_eval.schemas.manifest import DatasetManifest
from jev_agent_eval.utils.hashing import sha256_file

# Dataset paths must stay inside the repo root; reject traversal (spec 43).
# Evaluated lazily: a caller that chdir()s (tests, embedded use) must have
# its CURRENT cwd honored, not the cwd at module import time.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _allowed_roots() -> list[Path]:
    return [Path.cwd().resolve(), _REPO_ROOT]


def _safe_resolve(path: Path) -> Path:
    resolved = path.resolve()
    # Component containment, not string-prefix matching: a sibling like
    # /repo-malicious does not live inside /repo.
    if not any(resolved.is_relative_to(root) for root in _allowed_roots()):
        raise ValueError(f"dataset path escapes allowed roots: {path}")
    if ".." in path.parts:
        raise ValueError(f"dataset path contains '..': {path}")
    return resolved


def _resolve_entry(datasets_root: Path, entry_path: str) -> Path:
    # Reject absolute manifest entries: `root / "/abs/path"` silently
    # discards the root.
    if Path(entry_path).is_absolute():
        raise ValueError(f"dataset path must be relative: {entry_path}")
    return _safe_resolve(datasets_root / entry_path)


def load_manifest(path: str | Path) -> DatasetManifest:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return DatasetManifest.model_validate(data)


def load_manifest_cases(manifest: DatasetManifest, manifest_dir: Path) -> list[CanonicalCase]:
    """Load and integrity-check all cases referenced by a manifest.

    Case paths are relative to the datasets root (the parent of the manifests
    directory), so a manifest at datasets/manifests/core-v1.yaml referencing
    `internal/tool_risk.jsonl` resolves to datasets/internal/tool_risk.jsonl.
    """
    datasets_root = manifest_dir.parent
    all_cases: list[CanonicalCase] = []
    seen_ids: dict[str, str] = {}  # case_id -> file that first defined it
    for entry in manifest.cases:
        path = _resolve_entry(datasets_root, entry.path)
        if not path.exists():
            raise FileNotFoundError(f"manifest references missing file: {entry.path}")
        expected = manifest.integrity.get(entry.path)
        if expected is not None:
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"integrity mismatch for {entry.path}: expected {expected}, got {actual}"
                )
        dataset = LocalJSONLDataset([path])
        for case in dataset.iter_cases():
            if case.case_id in seen_ids:
                raise ValueError(
                    f"duplicate case_id {case.case_id!r} across manifest files "
                    f"{seen_ids[case.case_id]} and {entry.path}"
                )
            seen_ids[case.case_id] = entry.path
            all_cases.append(case)
    return all_cases


def compute_integrity(manifest_dir: Path, manifest: DatasetManifest) -> dict[str, str]:
    """Compute per-file sha256 for a manifest's case files.

    Case paths resolve against the datasets root (the manifest
    directory's parent), exactly like load_manifest_cases.
    """
    integrity: dict[str, str] = {}
    for entry in manifest.cases:
        path = _resolve_entry(manifest_dir.parent, entry.path)
        integrity[entry.path] = sha256_file(path)
    return integrity
