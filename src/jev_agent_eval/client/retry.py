"""Retry policy (spec section 11.4).

Retry only transient conditions: connection reset, timeout, 429, 500, 502,
503, 504. Never blindly retry semantic or schema errors. Exponential backoff
with jitter. The original error and retry history are preserved in the run
artifact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def is_transient_status(status: int) -> bool:
    return status in TRANSIENT_STATUS_CODES


@dataclass
class RetryPolicy:
    max_retries: int = 4
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    jitter: bool = True
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def should_retry(
        self, attempt: int, *, http_status: int | None, error_class: str | None
    ) -> bool:
        """Whether another retry is permitted for this failure signature."""
        if attempt >= self.max_retries:
            return False
        if http_status is not None:
            return is_transient_status(http_status)
        # Network-level transient error classes (httpx exception names).
        if error_class in {
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
        }:
            return True
        return False

    def backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped at backoff_max_s.

        The cap applies AFTER jitter: backoff_max_s is a wall-clock
        ceiling on any single retry delay, not a pre-jitter target —
        otherwise the actual sleep can exceed the configured maximum by
        the full jitter spread (1.5x).
        """
        delay = min(self.backoff_base_s * (2**attempt), self.backoff_max_s)
        if self.jitter:
            delay *= 0.5 + self._rng.random()
        return min(delay, self.backoff_max_s)
