"""Retries, timeouts and a circuit breaker around any :class:`Provider`.

Eval runs fire hundreds of concurrent requests at a model endpoint. Without
backpressure a degraded endpoint turns into a retry storm and a run full of
noise. The wrapper below gives:

* per-call timeout (``asyncio.timeout``),
* exponential backoff with *full jitter* for retryable errors only,
* a shared circuit breaker that fails fast once the endpoint looks unhealthy
  and lets a single probe through after a cool-down (half-open state).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from gauntlet.errors import CircuitOpenError, ProviderError
from gauntlet.models import AgentStep, Message, ToolSpec
from gauntlet.providers.base import Provider


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    timeout_s: float = 60.0

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """Full-jitter backoff: uniform(0, min(cap, base * 2**attempt))."""
        return rng.uniform(0.0, min(self.max_delay_s, self.base_delay_s * 2**attempt))


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - self._opened_at >= self.reset_timeout_s:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allow(self) -> bool:
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._probe_in_flight or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
        self._probe_in_flight = False


class ResilientProvider:
    """Decorates a provider with timeout, retry and circuit-breaking."""

    def __init__(
        self,
        inner: Provider,
        *,
        policy: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        seed: int | None = None,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.policy = policy or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()
        self._sleep = sleep
        self._rng = random.Random(seed)
        self.retries = 0

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> AgentStep:
        last_error: ProviderError | None = None
        for attempt in range(self.policy.max_attempts):
            if not self.breaker.allow():
                raise CircuitOpenError(
                    f"circuit open for {self.name} after repeated failures"
                    + (f"; last error: {last_error}" if last_error else "")
                )
            try:
                async with asyncio.timeout(self.policy.timeout_s):
                    step = await self.inner.complete(system=system, messages=messages, tools=tools)
            except TimeoutError:
                last_error = ProviderError(
                    f"{self.name} timed out after {self.policy.timeout_s}s", retryable=True
                )
            except ProviderError as exc:
                if not exc.retryable:
                    # Client-side errors say nothing about endpoint health.
                    self.breaker.record_success()
                    raise
                last_error = exc
            else:
                self.breaker.record_success()
                return step

            self.breaker.record_failure()
            if attempt + 1 < self.policy.max_attempts:
                self.retries += 1
                await self._sleep(self.policy.backoff(attempt, self._rng))

        assert last_error is not None
        raise ProviderError(
            f"giving up after {self.policy.max_attempts} attempts: {last_error}", retryable=False
        ) from last_error
