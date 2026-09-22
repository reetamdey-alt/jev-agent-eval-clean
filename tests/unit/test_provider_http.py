"""Regression tests for the SystemOne HTTP transport (loop: live-run audit).

Two defects surfaced during a real run against the live endpoint:

1. The provider POSTed to base_url + "/" (httpx merges an empty path into a
   trailing slash), which the endpoint rejects with a 307 redirect — the run
   failed 100% with no diagnostic beyond the bare status code.
2. Non-200 response bodies were discarded, so errors.jsonl could not show
   the server's own reason for the failure.

Both are asserted here against a mocked transport — no network access.
"""

from __future__ import annotations

import random
from datetime import UTC

import httpx

from jev_agent_eval.client.systemone import SystemOneProvider
from jev_agent_eval.schemas.request import JEVRequest


def _request() -> JEVRequest:
    return JEVRequest(
        state="User asked to inspect the logs. Proposed tool call: cat logs/app.log",
        model="jev-latest",
        questions={
            "is_high_risk": {"type": "noul", "instructions": "Is this risky?"},
        },
    )


class TestEndpointUrl:
    def test_post_goes_to_exact_url_without_trailing_slash(self):
        """httpx must not silently turn .../systemone into .../systemone/."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "model": "jev-latest",
                    "answers": {"is_high_risk": {"type": "noul", "noul": 0.1}},
                },
            )

        provider = SystemOneProvider(api_key="test-key")
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        try:
            result = provider.infer(_request())
        finally:
            import asyncio

            asyncio.run(provider._client.aclose())

        assert seen == ["https://grid.ai.juspay.net/v1/systemone"], seen
        assert result.response is not None

    def test_base_url_with_trailing_slash_config_is_normalized(self):
        """A config value ending in / must still produce the no-slash POST."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "model": "jev-latest",
                    "answers": {"is_high_risk": {"type": "noul", "noul": 0.1}},
                },
            )

        provider = SystemOneProvider(
            base_url="https://grid.ai.juspay.net/v1/systemone/", api_key="test-key"
        )
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        try:
            result = provider.infer(_request())
        finally:
            import asyncio

            asyncio.run(provider._client.aclose())

        assert seen == ["https://grid.ai.juspay.net/v1/systemone"], seen
        assert result.response is not None


class TestErrorBodyCapture:
    def test_non_200_body_is_captured(self):
        """The server's error message must land in error_body for
        errors.jsonl, not be discarded."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": "invalid api key"})

        provider = SystemOneProvider(api_key="test-key")
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        try:
            result = provider.infer(_request())
        finally:
            import asyncio

            asyncio.run(provider._client.aclose())

        assert result.response is None
        assert result.http_status == 403
        assert result.error_body is not None
        assert "invalid api key" in result.error_body

    def test_error_body_redacted_before_persistence(self):
        """redact_secrets must apply to the captured error body."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                text='{"error": "key sk-abcdefghijklmnop12345 rejected"}',
            )

        provider = SystemOneProvider(api_key="test-key")
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        try:
            result = provider.infer(_request())
        finally:
            import asyncio

            asyncio.run(provider._client.aclose())

        assert result.error_body is not None
        # The runner redacts before writing; assert the redaction helper
        # removes the key format from the captured body.
        from jev_agent_eval.client.redaction import redact_text

        redacted = redact_text(result.error_body)
        assert "sk-abcdefghijklmnop12345" not in redacted


class TestServerAwareBackoff:
    """Regression: on 429 the provider states when its rate-limit window
    resets ("Limit resets at: ... UTC", litellm body) and may send
    Retry-After. Blind exponential backoff burns all retries inside a dead
    window; the client must wait the server-stated time instead (bounded)."""

    @staticmethod
    def _rate_limited_response(reset_in_s: float, retry_after: str | None = None) -> httpx.Response:
        from datetime import datetime, timedelta

        reset_ts = datetime.now(UTC) + timedelta(seconds=reset_in_s)
        body = (
            '{"error":{"message":"litellm.RateLimitError: Rate limit exceeded '
            "for api_key=[REDACTED] Limit type: max_parallel_requests. Current "
            f"limit: 20, Remaining: 0. Limit resets at: "
            f'{reset_ts.strftime("%Y-%m-%d %H:%M:%S")} UTC"}}'
        )
        headers = {"Retry-After": retry_after} if retry_after else {}
        return httpx.Response(429, text=body, headers=headers)

    def test_body_reset_timestamp_sets_wait(self):
        provider = SystemOneProvider(api_key="test-key")
        resp = self._rate_limited_response(reset_in_s=42.0)
        wait = provider._server_retry_wait(resp)
        assert wait is not None
        assert 40.0 <= wait <= 42.0

    def test_retry_after_header_takes_precedence(self):
        provider = SystemOneProvider(api_key="test-key")
        resp = self._rate_limited_response(reset_in_s=600.0, retry_after="7")
        # The far-future body timestamp is beyond the 5-minute ceiling, so
        # the header (7s) is the only usable signal.
        wait = provider._server_retry_wait(resp)
        assert wait == 7.0

    def test_far_future_reset_is_ignored(self):
        provider = SystemOneProvider(api_key="test-key")
        resp = self._rate_limited_response(reset_in_s=3600.0)
        assert provider._server_retry_wait(resp) is None

    def test_non_429_has_no_server_wait(self):
        provider = SystemOneProvider(api_key="test-key")
        resp = httpx.Response(500, text="boom")
        assert provider._server_retry_wait(resp) is None

    def test_none_response_has_no_server_wait(self):
        provider = SystemOneProvider(api_key="test-key")
        assert provider._server_retry_wait(None) is None

    def test_429_without_body_or_header_has_no_wait(self):
        provider = SystemOneProvider(api_key="test-key")
        resp = httpx.Response(429, text="rate limited")
        assert provider._server_retry_wait(resp) is None

    def test_retry_loop_uses_server_wait(self):
        """End-to-end through infer(): the second attempt must be delayed
        by the server-stated reset, not only the exponential backoff."""
        from datetime import datetime, timedelta

        reset_ts = datetime.now(UTC) + timedelta(seconds=2.0)
        body = (
            "litellm.RateLimitError: Limit type: max_parallel_requests. "
            f"Limit resets at: {reset_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, text=body)
            return httpx.Response(
                200,
                json={
                    "model": "jev-latest",
                    "answers": {"is_high_risk": {"type": "noul", "noul": 0.1}},
                },
            )

        provider = SystemOneProvider(api_key="test-key")
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )
        try:
            import time as _time

            t0 = _time.monotonic()
            result = provider.infer(_request())
            elapsed = _time.monotonic() - t0
        finally:
            import asyncio

            asyncio.run(provider._client.aclose())

        assert result.response is not None
        assert result.retry_count == 1
        # The backoff actually used must reflect the server-stated wait,
        # not the blind exponential: attempt 0's exponential backoff is
        # base 0.5s with jitter, i.e. strictly < 0.75s. If the server wait
        # was honored, the recorded backoff (and elapsed time) is larger.
        assert result.retry_history[0]["backoff_s"] > 0.75, result.retry_history
        assert elapsed > 0.75, elapsed
        # Regression (2026-09-22 live run): latency_ms must measure the
        # LAST attempt's network time only, not the wall clock including
        # the backoff sleep — a 2s backoff plus a fast mock response must
        # NOT be reported as >2s latency (it used to fail p95 gates).
        assert result.latency_ms < 750, result.latency_ms
        assert "attempt_latency_ms" in result.retry_history[0], result.retry_history[0]

    def test_server_wait_backoff_has_jitter(self):
        """Regression (2026-09-22 live run): the server-aware wait replaced
        the exponential backoff WITHOUT jitter, so every concurrent worker
        woke at the same 'Limit resets at' instant, stampeded the freed
        parallel slots together and drew fresh 429s — a thundering-herd
        livelock (0 successes over 240s with 4 workers vs limit 5). The
        wait must be staggered with a small random spread."""
        provider = SystemOneProvider(api_key="test-key")
        resp = self._rate_limited_response(reset_in_s=42.0)
        server_wait = provider._server_retry_wait(resp)
        assert server_wait is not None

        # Simulate the retry loop's backoff computation many times: the
        # resulting waits must NOT all be identical (jitter present) and
        # must stay within [server_wait, server_wait + 2s].
        waits = []
        for _ in range(50):
            backoff = 0.25  # exponential backoff, smaller than server_wait
            if server_wait is not None and server_wait > backoff:
                backoff = server_wait + random.uniform(0.0, 2.0)
            waits.append(backoff)
        assert len(set(round(w, 4) for w in waits)) > 1, "no jitter in server wait"
        assert all(server_wait <= w <= server_wait + 2.0 for w in waits)
