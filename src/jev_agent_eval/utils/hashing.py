"""Deterministic hashing for requests, files, and configs (spec section 12).

Canonicalization procedure:
1. Parse JSON.
2. Remove authorization headers and secret material.
3. Normalize dictionary ordering (recursively sort keys).
4. Serialize with stable separators.
5. SHA-256 hash the canonical request.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Keys that are removed before hashing; they never influence request identity.
SECRET_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "x-api-key",
        "token",
        "secret",
        "password",
        "credential",
    }
)


def _strip_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_secrets(v) for k, v in obj.items() if k.lower() not in SECRET_KEYS}
    if isinstance(obj, list):
        return [_strip_secrets(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Stable JSON serialization: sorted keys, no whitespace, secrets stripped."""
    return json.dumps(
        _strip_secrets(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def request_hash(payload: dict[str, Any]) -> str:
    """Deterministic hash of a JEV request payload (secrets already absent)."""
    return sha256_hex(canonical_json(payload))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_files(paths: list[str | Path]) -> str:
    """Composite hash over several files: each contributes its path (as
    recorded) and content, so a case file changing under an unchanged
    manifest changes the dataset hash."""
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
        with open(str(p), "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        h.update(b"\0")
    return "sha256:" + h.hexdigest()
