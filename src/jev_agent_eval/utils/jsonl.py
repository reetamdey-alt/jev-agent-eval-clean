"""Streaming JSONL helpers (spec section 70: never load large datasets fully into memory)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed objects from a JSONL file, one line at a time."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def append_jsonl(path: str | Path, obj: Any) -> None:
    """Append one object as a JSONL line (incremental artifact writes)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: str | Path, objs: list[Any] | Iterator[Any]) -> None:
    """Write objects as JSONL. Truncates existing content."""
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
