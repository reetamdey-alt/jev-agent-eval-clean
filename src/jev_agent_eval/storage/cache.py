"""Deterministic response cache (spec section 12).

Cache key: provider/model + request_hash. Cached latency is never used as a
production latency metric; replayed measurements are flagged in reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_agent_eval.schemas.request import JEVRequest
from jev_agent_eval.utils.hashing import canonical_json, sha256_hex


def cache_key(model: str, request: JEVRequest) -> str:
    payload = {"model": model, "request": request.model_dump(exclude_none=True)}
    return sha256_hex(canonical_json(payload))


class ResponseCache:
    """JSONL-file-backed append cache; safe for concurrent appends per run."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._entries[rec["request_hash"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate torn writes

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(
        self,
        key: str,
        *,
        model: str,
        response: dict[str, Any],
        latency_ms: float | None,
        status: str,
    ) -> None:
        from jev_agent_eval.utils.timing import utc_now_iso

        rec = {
            "request_hash": key,
            "model": model,
            "created_at": utc_now_iso(),
            "response": response,
            "latency_ms": latency_ms,
            "status": status,
        }
        self._entries[key] = rec
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def __len__(self) -> int:
        return len(self._entries)
