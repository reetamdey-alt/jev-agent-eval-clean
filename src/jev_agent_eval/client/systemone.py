"""Typed JEV SystemOne client (spec section 11).

Synchronous wrapper around an async httpx client. Handles:
- auth from the JEV_API_KEY environment variable only
- bounded retries on transient failures with exponential backoff + jitter
- token-bucket rate limiting
- response-size capping
- tolerant response parsing with structured error reporting
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import ValidationError

from jev_agent_eval.client.base import InferenceResult
from jev_agent_eval.client.rate_limit import TokenBucketRateLimiter
from jev_agent_eval.client.redaction import redact_text
from jev_agent_eval.client.retry import RetryPolicy
from jev_agent_eval.schemas.request import JEVRequest
from jev_agent_eval.schemas.response import JEVAnswer, JEVResponse, JEVUsage

DEFAULT_BASE_URL = "https://grid.ai.juspay.net/v1/systemone"
API_KEY_ENV = "JEV_API_KEY"


class SystemOneProvider:
    """JEV provider over POST /v1/systemone."""

    name = "systemone"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "jev-latest",
        api_key: str | None = None,
        timeout_s: float = 30.0,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 30.0,
        max_retries: int = 4,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 8.0,
        requests_per_second: float | None = None,
        max_response_body_bytes: int = 10_485_760,
    ):
        # rstrip so the provider always POSTs to ".../systemone", never
        # ".../systemone/": httpx joins base_url + "/" (empty path) into a
        # trailing-slash URL, which this endpoint rejects with a 307
        # redirect. The canonical URL is the config value verbatim.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(
                f"JEV API key not set: export {API_KEY_ENV}='<key>' "
                "(credentials are never stored in configs or artifacts)"
            )
        self.timeout = httpx.Timeout(
            connect=connect_timeout_s, read=read_timeout_s, write=timeout_s, pool=timeout_s
        )
        self.retry_policy = RetryPolicy(
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
        )
        self.rate_limiter = (
            TokenBucketRateLimiter(requests_per_second)
            if requests_per_second and requests_per_second > 0
            else None
        )
        self.max_response_body_bytes = max_response_body_bytes
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def _parse_response(self, body: str) -> tuple[JEVResponse | None, str | None]:
        """Parse a response body tolerantly; return (response, error)."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            return None, f"malformed_json: {e}"
        if not isinstance(data, dict):
            return None, "response_not_object"
        try:
            response = JEVResponse.model_validate(data)
        except ValidationError as e:
            return None, f"schema_error: {e.errors(include_url=False)[:3]}"
        return response, None

    async def _infer_once(
        self, client: httpx.AsyncClient, request: JEVRequest
    ) -> tuple[httpx.Response | None, str | None, str | None]:
        """One HTTP attempt. Returns (response, error_class, error_detail)."""
        payload = request.model_dump(exclude_none=True)
        try:
            # Full URL, not base_url + "": httpx merges an empty path into
            # base_url + "/", sending the POST to ".../systemone/" — a
            # different resource on this endpoint (307 redirect, never a
            # valid answer).
            resp = await client.post(self.base_url, json=payload)
        except httpx.TimeoutException as e:
            return None, type(e).__name__, str(e)
        except httpx.TransportError as e:
            return None, type(e).__name__, str(e)
        return resp, None, None

    async def _infer_once_timed(
        self, client: httpx.AsyncClient, request: JEVRequest
    ) -> tuple[httpx.Response | None, str | None, str | None, float]:
        """One HTTP attempt with per-attempt latency.

        Latency measured HERE excludes client-side retry backoff sleeps —
        the wall-clock around the whole retry loop (what latency_ms used
        to be) counts our own throttling against the provider's latency
        gates: a case that waited 240s in backoffs and answered in 0.3s
        was reported as 240s and failed the p95 gate.
        """
        t0 = time.perf_counter()
        resp, err_class, err_detail = await self._infer_once(client, request)
        return resp, err_class, err_detail, (time.perf_counter() - t0) * 1000.0

    async def _read_body(self, resp: httpx.Response) -> tuple[str | None, str | None]:
        """Read the response body as text, enforcing the size cap in BYTES.

        Declared Content-Length is checked before reading the body so an
        oversized response is rejected without downloading it; the actual
        byte length is re-checked after decoding (UTF-8 multibyte content
        can exceed len(body) in characters).
        """
        if resp.status_code != 200:
            # Error bodies are small and useful for diagnostics; read them.
            return resp.text, None
        try:
            declared = int(resp.headers.get("content-length", ""))
        except ValueError:
            declared = None
        if declared is not None and declared > self.max_response_body_bytes:
            return None, "response_body_too_large"
        try:
            raw = await resp.aread()
        except httpx.TransportError as e:
            return None, f"body_read_error: {type(e).__name__}"
        if len(raw) > self.max_response_body_bytes:
            return None, "response_body_too_large"
        return raw.decode(resp.encoding or "utf-8", errors="replace"), None

    def _server_retry_wait(self, resp: httpx.Response | None) -> float | None:
        """Seconds the server asked us to wait before retrying, if it said.

        Recognizes, in order of preference:
        - ``Retry-After`` header (seconds form; the delta-seconds variant is
          the only one litellm emits, and the HTTP-date form is rare here).
        - The litellm 429 body text ``Limit resets at: YYYY-MM-DD HH:MM:SS UTC``
          parsed against the wall clock.

        Returns None when neither is present/parsable, or when the wait
        exceeds a sane ceiling (5 minutes) — a far-future or garbage
        timestamp must not translate into an unbounded sleep.
        """
        if resp is None or resp.status_code != 429:
            return None

        ceiling_s = 300.0
        header = resp.headers.get("retry-after")
        if header:
            try:
                wait = float(header)
                if 0 <= wait <= ceiling_s:
                    return wait
            except ValueError:
                pass  # HTTP-date form or garbage; fall through to the body.
        # The body may not be cheap to read twice, but this path only runs
        # on a 429 (already-read small error body) — resp.text is buffered
        # by httpx after the first read inside _infer_once's error path.
        text = resp.text or ""
        match = re.search(r"Limit resets at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", text)
        if match:
            try:
                reset_dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=UTC
                )
            except ValueError:
                return None
            now = datetime.now(UTC)
            wait = (reset_dt - now).total_seconds()
            if 0 <= wait <= ceiling_s:
                return wait
        return None

    async def infer_async(self, request: JEVRequest) -> InferenceResult:
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        client = self._get_client()
        retry_history: list[dict[str, Any]] = []
        attempt = 0
        result = InferenceResult()

        while True:
            resp, err_class, err_detail, attempt_latency_ms = await self._infer_once_timed(
                client, request
            )
            status = resp.status_code if resp is not None else None

            transient = self.retry_policy.should_retry(
                attempt, http_status=status, error_class=err_class
            )
            if not transient:
                break
            backoff = self.retry_policy.backoff_seconds(attempt)
            # Server-aware backoff: on 429 the provider says when its
            # rate-limit window resets — the litellm body carries
            # "Limit resets at: 2026-09-22 00:42:03 UTC", and the HTTP
            # layer may send a Retry-After header. Honoring them waits out
            # the actual window instead of burning all retries blindly
            # inside a dead window (each 429 costs a full retry attempt
            # while slots remain at "Remaining: 0").
            server_wait = self._server_retry_wait(resp)
            if server_wait is not None and server_wait > backoff:
                # Wait out the server's window, but stagger the wake-up:
                # without jitter every concurrent worker wakes at the same
                # "Limit resets at" instant, stampedes the (provider-side
                # bounded) parallel slots together and draws a fresh 429 —
                # a thundering-herd livelock observed live on 2026-09-22
                # (4 workers, provider limit 5, 0 successes over 240s).
                # A small random spread (<= 2s, never past the window by
                # more than 2s) staggers arrivals into the freed slots.
                backoff = server_wait + random.uniform(0.0, 2.0)
            retry_history.append(
                {
                    "attempt": attempt + 1,
                    "http_status": status,
                    "error_class": err_class,
                    "error_detail": err_detail,
                    "attempt_latency_ms": attempt_latency_ms,
                    "backoff_s": backoff,
                }
            )
            attempt += 1
            await asyncio.sleep(backoff)

        # latency_ms is the LAST attempt's network time (excluding our own
        # backoff sleeps) — the provider's latency gates must not measure
        # client-side throttling. The retry-loop wall clock stays available
        # via retry_history (attempt latencies + backoffs sum to it).
        result.latency_ms = attempt_latency_ms
        result.http_status = status if resp is not None else None
        result.network_error_class = err_class
        result.retry_count = attempt
        result.retry_history = retry_history

        if resp is None:
            return result

        if resp.status_code != 200:
            # Read the error body for diagnostics — errors.jsonl must show
            # the server's reason (auth failure, bad request, ...), not
            # just the bare status code.
            body, _ = await self._read_body(resp)
            result.error_body = (body or "")[:1000] or None
            return result

        body, body_error = await self._read_body(resp)
        if body is None:
            result.parse_error = body_error
            return result

        parsed, parse_error = self._parse_response(body)
        if parsed is None:
            result.parse_error = parse_error
            result.raw_body = body[:10000]
            return result
        result.response = parsed
        result.raw_body = body
        return result

    def infer(self, request: JEVRequest) -> InferenceResult:
        """Synchronous entry point; runs the async client in a private loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise RuntimeError(
                "SystemOneProvider.infer() cannot be called from a running event loop; use infer_async()"
            )
        return asyncio.run(self._guarded(request))

    async def _guarded(self, request: JEVRequest) -> InferenceResult:
        try:
            return await self.infer_async(request)
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


class MockJEVProvider:
    """Deterministic offline provider used for tests, CI, and smoke runs.

    Simulates the JEV contract without network access. Answers are derived
    from deterministic keyword heuristics over the question instructions and
    the case state; this exercises the full pipeline end-to-end and provides
    the "simple deterministic heuristic" baseline (spec section 49).
    """

    name = "mock"

    def __init__(self, model: str = "jev-latest", latency_ms: float = 0.0):
        self.model = model
        self.latency_ms = latency_ms

    def _answer_noul(self, question_id: str, instructions: str, state: str) -> dict[str, Any]:
        text = (state + " " + question_id).lower()
        signal = (instructions + " " + question_id).lower()
        risk_words = (
            "rm -rf",
            "delete",
            "drop table",
            "force push",
            "push --force",
            "sudo",
            "chmod 777",
            "production",
            "deploy",
            "secret",
            "credential",
            "password",
            "exfiltrat",
            "curl -x post",
            "destroy",
            "truncate",
            "reboot",
            "shutdown",
            "dd if=",
            "npm publish",
            "reset --hard",
            "s3 rm",
            "prune",
            "id_rsa",
            "find /",
            "-exec",
        )
        complete_words = (
            "all tests pass",
            "42/42",
            "all green",
            "acceptance criteria",
            "tests pass",
            "suite green",
            "now passes",
            "implemented and tested",
            "updated with",
        )
        # The proposed command is everything after the last proposal marker.
        # Risk must be judged on the COMMAND, not the surrounding state: the
        # request context ("verify credentials") and injected README text
        # must not make a benign command look risky. The paraphrase marker
        # ("The agent intends to run:") is honored too: risk scoping must
        # survive wording changes, or the metamorphic paraphrase relation
        # reports spurious flips.
        tail = text
        for marker in (
            "proposed tool call:",
            "proposed action:",
            "intends to run:",
            "the agent runs:",
            "executed tool:",
        ):
            tail = tail.split(marker)[-1]
        has_risk = any(w in tail for w in risk_words)
        # Scoped destructive commands are low-risk by design (minimal pairs):
        # `rm -rf ./build` targets a local, rebuildable directory while
        # `rm -rf /` and `rm -rf logs/` are not scoped that way.
        scoped_safe = "rm -rf ./" in tail
        if scoped_safe:
            has_risk = False
        # Completion evidence must come from the agent's REPORTED outcome, not
        # the user's request: "make the suite green" is a goal statement, not
        # evidence the goal was reached. The request line precedes the agent's
        # message, so completion words are searched in the agent portion.
        agent_portion = text
        for marker in ("user request:", "agent message:"):
            if marker in agent_portion:
                agent_portion = agent_portion.split(marker, 1)[1]
        has_complete = any(w in agent_portion for w in complete_words)
        # Benign read-only commands: strong evidence of NOT high risk when
        # the question is about risk and no risk word fired.
        benign_commands = (
            "cat ",
            "git status",
            "git diff",
            "git log",
            "npm test",
            "grep ",
            "ls ",
            "docker ps",
            "curl -s http",
            "aws sts get-caller-identity",
            "pwd",
            "head ",
            "tail ",
            "wc ",
            "find ",
            "which ",
            "edit src",
            "chmod +x",
            "psql -c 'select",
            "kubectl get",
            "cat ~/.ssh/config",
        )
        has_benign = scoped_safe or any(b in tail for b in benign_commands)
        # Contradicting evidence: failing tests, missing runs, stale markers.
        # A failure that has since been fixed ("failing test now passes") is
        # NOT a contradiction, and neither is an explicit zero-failure count
        # ("failed: 0", "failures: 0") — 'fail' as a bare substring would
        # fire on a fully passing test summary. Zero-count mentions are
        # stripped before any fail-word is searched.
        import re as _re

        sanitized = _re.sub(r"fail\w*['\"]?\s*[:=]\s*['\"]?0\b", "", text)
        has_fail_word = "fail" in sanitized
        has_contradiction = (
            has_fail_word
            or any(
                w in text
                for w in (
                    "tests not run",
                    "not applied",
                    "not written",
                    "no changes in git diff",
                    "8 months ago",
                    "still reproduces",
                    "signature changed",
                    "methods removed",
                )
            )
        ) and "now passes" not in text
        if not has_risk and not has_complete and not has_contradiction and not has_benign:
            value = 0.5
        else:
            # Polarity of the question decides how the evidence maps to the
            # answer: "is high risk?" and "is the claim supported?" /
            # "is the goal complete?" read the same state oppositely.
            # Positive-polarity questions (risk, stuck, violation, ...) say
            # yes on risky evidence; completion/support questions say yes on
            # completion evidence and no on risk or contradicting evidence.
            # Scope questions (injection resistance) judge whether the
            # PROPOSED ACTION matches the user request — injected scary text
            # in surrounding content is irrelevant by construction.
            if "in scope" in signal or "within the scope" in signal:
                # The proposed action line follows "Proposed action:"; the
                # injected content lives in the README block. If the state's
                # post-request action is one of the benign verbs, in scope.
                value = 1.0 if has_benign else 0.0
            elif any(
                w in signal
                for w in ("goal complete", "satisfied", "claim supported", "supported by")
            ):
                value = 1.0 if has_complete and not has_risk and not has_contradiction else 0.0
            elif "high risk" in signal or "destructive" in signal:
                value = 1.0 if has_risk else (0.0 if has_benign else 0.5)
            elif "stuck" in signal:
                # Stuck = repeating the same failing action without progress:
                # retry-count language ("a third time", "again", "maybe it
                # will pass now") plus an error result. A single failure
                # with no retry language is NOT stuck — the agent may simply
                # be mid-diagnosis.
                retry_language = any(
                    w in text
                    for w in (
                        "a third time",
                        "third time",
                        "again",
                        "still",
                        "same command",
                        "same curl",
                        "re-running",
                        "retrying",
                        "re-running the",
                        "maybe it will pass",
                    )
                )
                value = 1.0 if (retry_language and has_contradiction) else 0.0
            else:
                value = 1.0 if has_risk else 0.0
        # Metamorphic semantic_negation variants append a marked contradiction
        # ("the previous evidence is now invalid; the opposite outcome holds").
        # The heuristic honors the marker by flipping its verdict — an
        # evaluator that ignores a state update it cannot refute would report
        # perfect invariance on every negation (pass_rate 0.0 forever).
        if "the opposite outcome holds" in text:
            value = 1.0 - value
        return {
            "type": "noul",
            "noul": value,
            "probabilities": {"true": value, "false": 1.0 - value},
            "confidence": round(max(value, 1.0 - value), 4),
        }

    def _answer_choice(
        self, question_id: str, instructions: str, state: str, criteria: Any
    ) -> dict[str, Any]:
        options: list[str] = []
        if isinstance(criteria, dict):
            options = list(criteria.keys())
        elif isinstance(criteria, list):
            options = list(criteria)
        if not options:
            options = ["unknown"]
        text = (state + " " + question_id).lower()
        tail = text.split("proposed tool call:")[-1].split("proposed action:")[-1]
        choice: str | None = None
        if {"safe", "low_risk", "high_risk", "forbidden"} & set(options):
            # Risk taxonomy: classify the proposed command, from the most
            # severe signatures down to benign read-only commands.
            if "rm -rf ./" in tail:
                choice = "low_risk"
            elif "rm -rf /" in tail:
                choice = "forbidden"
            elif "rm -rf" in tail:
                choice = "high_risk"
            elif (
                "chmod 777" in tail
                or ("find /" in tail and "-exec" in tail)
                or "evil.example.com" in tail
            ):
                choice = "forbidden"
            elif (
                "drop table" in tail
                or "s3 rm" in tail
                or "reset --hard" in tail
                or ("push --force" in tail or "force" in tail and "push" in tail)
                or "delete namespace" in tail
                or "npm publish" in tail
            ):
                choice = "irreversible"
            elif "prune" in tail:
                choice = "high_risk"
            elif ".ssh" in tail or "aws " in tail or "kubectl" in tail:
                choice = "sensitive"
            elif "chmod +x" in tail or "curl -s" in tail:
                choice = "low_risk"
            elif any(
                b in tail
                for b in (
                    "cat ",
                    "git status",
                    "git diff",
                    "npm test",
                    "grep ",
                    "ls ",
                    "docker ps",
                    "pwd",
                    "psql",
                    "find ",
                    "which ",
                )
            ):
                choice = "safe"
        heuristic_matched = choice is not None
        if choice is None or choice not in options:
            # Deterministic pick from a hash of the state: exercises scoring paths.
            h = sum(ord(c) for c in (state + question_id)) % max(1, len(options))
            choice = options[h]
            heuristic_matched = False
        # A heuristic-driven classification is deterministic and correct;
        # the hash fallback is a guess. Only the guess should carry mid
        # confidence — otherwise the mock miscalibrates itself and fails
        # its own ece_max gate.
        conf = 0.95 if heuristic_matched else 0.7
        probabilities = {
            opt: conf if opt == choice else round((1.0 - conf) / max(1, len(options) - 1), 4)
            for opt in options
        }
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": probabilities,
            "confidence": conf,
        }

    def _answer_score(self, question_id: str, instructions: str, state: str) -> dict[str, Any]:
        text = (state + " " + question_id).lower()
        if any(w in text for w in ("all tests pass", "42/42", "exact failing", "critical")):
            value = 0.9
        elif any(w in text for w in ("unrelated", "readme", "old issue")):
            value = 0.1
        else:
            value = 0.5
        return {"type": "score", "score": value, "confidence": 0.6}

    def infer(self, request: JEVRequest) -> InferenceResult:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        return self._infer_body(request)

    def _infer_body(self, request: JEVRequest) -> InferenceResult:
        answers: dict[str, JEVAnswer] = {}
        for qid, question in request.questions.items():
            if question.type == "noul":
                data = self._answer_noul(qid, question.instructions, request.state)
            elif question.type == "choice":
                data = self._answer_choice(
                    qid, question.instructions, request.state, question.criteria
                )
            else:
                data = self._answer_score(qid, question.instructions, request.state)
            answers[qid] = JEVAnswer.model_validate(data)
        response = JEVResponse(
            model=self.model,
            answers=answers,
            usage=JEVUsage(
                input_tokens=max(1, len(request.state) // 4),
                output_tokens=max(1, len(answers)),
            ),
        )
        return InferenceResult(
            response=response,
            raw_body=response.model_dump_json(),
            http_status=200,
            latency_ms=self.latency_ms or 1.0,
        )

    async def infer_async(self, request: JEVRequest) -> InferenceResult:
        # Async path must yield to the event loop: a coroutine that never
        # awaits runs to completion without re-entering the loop, so a
        # concurrent `asyncio.sleep` timer (a soak duration, a stop event's
        # waiter) can never fire and worker loops starve the loop forever.
        # Simulated latency is async too — `time.sleep` here would block
        # every other coroutine.
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        await asyncio.sleep(0)
        return self._infer_body(request)


def redacted_raw_body(result: InferenceResult) -> str | None:
    """Redacted raw body for artifact storage (spec 43.1)."""
    if result.raw_body is None:
        return None
    return redact_text(result.raw_body)
