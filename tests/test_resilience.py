from __future__ import annotations

import asyncio

import pytest

from conftest import FlakyProvider
from gauntlet import ProviderError
from gauntlet.errors import CircuitOpenError
from gauntlet.resilience import BreakerState, CircuitBreaker, ResilientProvider, RetryPolicy

sleeps: list[float] = []


async def _no_sleep(seconds: float) -> None:
    sleeps.append(seconds)


def _wrap(inner, **kw) -> ResilientProvider:
    return ResilientProvider(inner, sleep=_no_sleep, seed=1, **kw)


async def _complete(p: ResilientProvider):
    return await p.complete(system="", messages=[], tools=[])


async def test_retries_then_succeeds() -> None:
    inner = FlakyProvider(failures=2)
    provider = _wrap(inner, policy=RetryPolicy(max_attempts=4))
    step = await _complete(provider)
    assert step.text == "ok" and inner.calls == 3 and provider.retries == 2


async def test_gives_up_after_max_attempts() -> None:
    provider = _wrap(FlakyProvider(failures=10), policy=RetryPolicy(max_attempts=3))
    with pytest.raises(ProviderError, match="giving up after 3 attempts"):
        await _complete(provider)


async def test_non_retryable_errors_are_not_retried() -> None:
    inner = FlakyProvider(failures=5, error=ProviderError("bad request", retryable=False))
    with pytest.raises(ProviderError, match="bad request"):
        await _complete(_wrap(inner))
    assert inner.calls == 1


async def test_timeout_is_retryable() -> None:
    class Slow:
        name = "slow"
        calls = 0

        async def complete(self, **_):
            self.calls += 1
            await asyncio.sleep(1)

    slow = Slow()
    provider = _wrap(slow, policy=RetryPolicy(max_attempts=2, timeout_s=0.01))
    with pytest.raises(ProviderError, match="timed out"):
        await _complete(provider)
    assert slow.calls == 2


def test_backoff_is_bounded_full_jitter() -> None:
    import random

    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=4.0)
    rng = random.Random(0)
    for attempt in range(10):
        delay = policy.backoff(attempt, rng)
        assert 0.0 <= delay <= min(4.0, 0.5 * 2**attempt)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_half_opens_and_closes() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_s=10, clock=clock)
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN and not breaker.allow()

    clock.now = 10
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow()  # the single probe
    assert not breaker.allow()  # concurrent callers still fail fast
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_failed_probe_reopens() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=5, clock=clock)
    breaker.record_failure()
    clock.now = 5
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN


async def test_open_circuit_fails_fast() -> None:
    inner = FlakyProvider(failures=100)
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_s=60)
    provider = _wrap(inner, policy=RetryPolicy(max_attempts=5), breaker=breaker)
    with pytest.raises(CircuitOpenError):
        await _complete(provider)
    assert inner.calls == 2  # stopped hammering the endpoint
