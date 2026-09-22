"""Evaluator self-tests for transport and retry behavior (spec 63).

Deterministic, offline (mock transport): each transient failure mode the
provider can hit — 429, 500, 502, 503, 504, timeout, connection reset —
plus Retry-After header handling (seconds form and HTTP-date form).
"""

from __future__ import annotations

from email.utils import formatdate

import httpx
import pytest

from jev_agent_eval.client.retry import RetryPolicy, is_transient_status
from jev_agent_eval.schemas.request import JEVQuestion, JEVRequest


def _request() -> JEVRequest:
    return JEVRequest(
        state="self-test state",
        model="jev-latest",
        questions={"q1": JEVQuestion(type="noul", instructions="Is the action safe?")},
    )


def _ok_body() -> dict:
    return {
        "model": "jev-latest",
        "answers": {
            "q1": {
                "type": "noul",
                "noul": 0,
                "probabilities": {"true": 0.1, "false": 0.9},
                "confidence": 0.9,
            }
        },
    }


class _ScriptedTransport(httpx.MockTransport):
    """Serves a scripted sequence of responses, one per request."""

    def __init__(self, script: list[httpx.Response | Exception]):
        self.script = list(script)
        self.seen_requests: list[httpx.Request] = []
        super().__init__(self._handler)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.seen_requests.append(request)
        item = self.script.pop(0) if self.script else httpx.Response(200, json=_ok_body())
        if isinstance(item, Exception):
            raise item
        return item


def _provider(transport: _ScriptedTransport, **kwargs):
    from jev_agent_eval.client.systemone import SystemOneProvider

    kwargs.setdefault("api_key", "self-test-key")
    kwargs.setdefault("backoff_base_s", 0.001)
    kwargs.setdefault("backoff_max_s", 0.005)
    kwargs.setdefault("max_retries", 3)
    prov = SystemOneProvider(base_url="https://selftest.example/v1", **kwargs)
    # Deterministic backoff (no jitter) for wall-clock predictability.
    prov.retry_policy.jitter = False
    # Replace the lazily-created client with one over the scripted transport.
    prov._client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=1.0, read=1.0, write=1.0, pool=1.0),
    )
    return prov


class TestTransientStatusRetry:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    @pytest.mark.asyncio
    async def test_transient_status_retried_then_succeeds(self, status):
        transport = _ScriptedTransport(
            [httpx.Response(status), httpx.Response(200, json=_ok_body())]
        )
        prov = _provider(transport)
        result = await prov.infer_async(_request())
        assert result.http_status == 200
        assert result.response is not None
        assert result.retry_count == 1
        assert len(transport.seen_requests) == 2
        await prov.aclose()

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    @pytest.mark.asyncio
    async def test_non_transient_status_not_retried(self, status):
        transport = _ScriptedTransport([httpx.Response(status, json={"error": "no"})])
        prov = _provider(transport)
        result = await prov.infer_async(_request())
        assert result.http_status == status
        assert result.retry_count == 0
        assert len(transport.seen_requests) == 1
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_exhausted_retries_report_final_status(self):
        transport = _ScriptedTransport([httpx.Response(503)] * 5)
        prov = _provider(transport, max_retries=3)
        result = await prov.infer_async(_request())
        assert result.http_status == 503
        # 1 initial + 3 retries
        assert len(transport.seen_requests) == 4
        assert result.retry_count == 3
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_retry_history_preserved(self):
        transport = _ScriptedTransport(
            [httpx.Response(500), httpx.Response(502), httpx.Response(200, json=_ok_body())]
        )
        prov = _provider(transport)
        result = await prov.infer_async(_request())
        assert result.retry_history, "retry history must be recorded (spec 11.4)"
        assert result.retry_count == 2
        await prov.aclose()


class TestNetworkErrorRetry:
    @pytest.mark.asyncio
    async def test_connection_reset_retried(self):
        transport = _ScriptedTransport(
            [httpx.ConnectError("connection reset by peer"), httpx.Response(200, json=_ok_body())]
        )
        prov = _provider(transport)
        result = await prov.infer_async(_request())
        assert result.http_status == 200
        assert result.retry_count == 1
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_read_timeout_retried(self):
        transport = _ScriptedTransport(
            [httpx.ReadTimeout("timed out"), httpx.Response(200, json=_ok_body())]
        )
        prov = _provider(transport)
        result = await prov.infer_async(_request())
        assert result.http_status == 200
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_timeout_classified_as_network_error(self):
        transport = _ScriptedTransport([httpx.ReadTimeout("timed out")] * 4)
        prov = _provider(transport, max_retries=3)
        result = await prov.infer_async(_request())
        assert result.http_status is None
        assert result.network_error_class == "ReadTimeout"
        assert len(transport.seen_requests) == 4
        await prov.aclose()


class TestRetryAfter:
    @pytest.mark.asyncio
    async def test_retry_after_seconds_delays_retry(self):
        transport = _ScriptedTransport(
            [
                httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "slow down"}),
                httpx.Response(200, json=_ok_body()),
            ]
        )
        prov = _provider(transport, backoff_base_s=0.001, max_retries=2)
        result = await prov.infer_async(_request())
        assert result.http_status == 200
        # The Retry-After hint must be recorded in the retry history.
        assert result.retry_history
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_retry_after_http_date_parses(self):
        # A past date means "retry now" — must not crash the parser.
        past = formatdate(usegmt=True)
        transport = _ScriptedTransport(
            [
                httpx.Response(503, headers={"Retry-After": past}, json={"error": "unavailable"}),
                httpx.Response(200, json=_ok_body()),
            ]
        )
        prov = _provider(transport, backoff_base_s=0.001, max_retries=2)
        result = await prov.infer_async(_request())
        assert result.http_status == 200
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_malformed_retry_after_ignored(self):
        transport = _ScriptedTransport(
            [
                httpx.Response(429, headers={"Retry-After": "not-a-date"}, json={}),
                httpx.Response(200, json=_ok_body()),
            ]
        )
        prov = _provider(transport, backoff_base_s=0.001, max_retries=2)
        result = await prov.infer_async(_request())
        assert result.http_status == 200
        await prov.aclose()


class TestRetryPolicyUnit:
    def test_transient_codes(self):
        for status in (429, 500, 502, 503, 504):
            assert is_transient_status(status)
        for status in (200, 400, 401, 403, 404, 422):
            assert not is_transient_status(status)

    def test_should_retry_respects_max(self):
        policy = RetryPolicy(max_retries=2, jitter=False)
        assert policy.should_retry(0, http_status=429, error_class=None)
        assert policy.should_retry(1, http_status=429, error_class=None)
        assert not policy.should_retry(2, http_status=429, error_class=None)

    def test_backoff_exponential_and_capped(self):
        policy = RetryPolicy(max_retries=10, backoff_base_s=1.0, backoff_max_s=4.0, jitter=False)
        assert policy.backoff_seconds(0) == 1.0
        assert policy.backoff_seconds(1) == 2.0
        assert policy.backoff_seconds(2) == 4.0
        assert policy.backoff_seconds(3) == 4.0  # capped

    def test_semantic_error_never_retried(self):
        policy = RetryPolicy()
        assert not policy.should_retry(0, http_status=400, error_class=None)
        assert not policy.should_retry(0, http_status=None, error_class="SomeSemanticError")

    def test_connection_reset_is_transient(self):
        policy = RetryPolicy()
        # RemoteProtocolError is the httpx signature of a reset mid-stream.
        assert policy.should_retry(0, http_status=None, error_class="RemoteProtocolError")
