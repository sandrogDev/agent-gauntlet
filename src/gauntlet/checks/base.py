"""Check registry and execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from gauntlet.models import CheckResult, CheckSpec, Scenario, Trace
from gauntlet.providers.base import Provider
from gauntlet.tools import ToolRegistry


@dataclass(frozen=True)
class CheckOutcome:
    passed: bool
    message: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class CheckContext:
    scenario: Scenario
    trace: Trace
    registry: ToolRegistry
    judge: Provider | None = None


CheckFn = Callable[[CheckContext, dict[str, Any]], Awaitable[CheckOutcome]]

CHECKS: dict[str, CheckFn] = {}


def register_check(name: str) -> Callable[[CheckFn], CheckFn]:
    """Register a check. Third-party code can add its own checks the same way."""

    def decorator(fn: CheckFn) -> CheckFn:
        if name in CHECKS:
            raise ValueError(f"check '{name}' already registered")
        CHECKS[name] = fn
        return fn

    return decorator


def require(params: dict[str, Any], key: str) -> Any:
    if key not in params:
        raise ValueError(f"missing required parameter '{key}'")
    return params[key]


async def run_checks(specs: Sequence[CheckSpec], ctx: CheckContext) -> list[CheckResult]:
    results: list[CheckResult] = []
    for spec in specs:
        fn = CHECKS.get(spec.check)
        if fn is None:
            outcome = CheckOutcome(False, f"unknown check '{spec.check}'")
        else:
            try:
                outcome = await fn(ctx, spec.params)
            except (KeyError, TypeError, ValueError) as exc:
                outcome = CheckOutcome(False, f"misconfigured check: {exc}")
        results.append(
            CheckResult(
                check=spec.check,
                label=spec.label,
                passed=outcome.passed,
                weight=spec.weight,
                severity=spec.severity,
                message=outcome.message,
                skipped=outcome.skipped,
            )
        )
    return results
