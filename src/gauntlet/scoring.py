"""Scenario scoring and run-level metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from statistics import fmean

from gauntlet.models import CheckResult, RunMetrics, ScenarioResult, Severity, Trace


def score_scenario(
    checks: Sequence[CheckResult], trace: Trace, threshold: float
) -> tuple[float, bool]:
    """Weighted score in [0, 1]; pass requires no critical failure and score >= threshold.

    Skipped checks (e.g. judge checks with no judge configured) are excluded
    from both numerator and denominator rather than counted as failures.
    """
    if trace.error:
        return 0.0, False
    counted = [c for c in checks if not c.skipped]
    total = sum(c.weight for c in counted)
    score = sum(c.weight for c in counted if c.passed) / total if total else 1.0
    critical = any(not c.passed and c.severity is Severity.CRITICAL for c in counted)
    return round(score, 4), (not critical and score >= threshold)


def _rate(results: Sequence[ScenarioResult]) -> float | None:
    return round(sum(r.passed for r in results) / len(results), 4) if results else None


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def compute_metrics(results: Sequence[ScenarioResult]) -> RunMetrics:
    if not results:
        return RunMetrics()

    calls = [c for r in results for c in r.trace.tool_calls]
    organic = [c for c in calls if not c.injected_fault]
    latencies = [r.trace.latency_s for r in results]
    baseline = [r for r in results if not r.scenario.is_adversarial]
    adversarial = [r for r in results if r.scenario.is_adversarial]

    by_mutator: dict[str, list[ScenarioResult]] = {}
    for r in adversarial:
        by_mutator.setdefault(r.scenario.mutator or "?", []).append(r)

    failures = Counter(c.check for r in results for c in r.failures)

    return RunMetrics(
        scenarios=len(results),
        passed=sum(r.passed for r in results),
        pass_rate=_rate(results) or 0.0,
        mean_score=round(fmean(r.score for r in results), 4),
        baseline_pass_rate=_rate(baseline),
        adversarial_pass_rate=_rate(adversarial),
        tool_calls=len(calls),
        tool_call_validity=round(sum(c.valid for c in calls) / len(calls), 4) if calls else 1.0,
        tool_call_success_rate=(
            round(sum(c.valid and c.succeeded for c in organic) / len(organic), 4)
            if organic
            else 1.0
        ),
        critical_failures=sum(
            any(c.severity is Severity.CRITICAL for c in r.failures) for r in results
        ),
        errored=sum(r.trace.error is not None for r in results),
        mean_latency_s=round(fmean(latencies), 4),
        p95_latency_s=round(percentile(latencies, 0.95), 4),
        input_tokens=sum(r.trace.usage.input_tokens for r in results),
        output_tokens=sum(r.trace.usage.output_tokens for r in results),
        failures_by_check=dict(failures.most_common()),
        pass_rate_by_mutator={m: _rate(rs) or 0.0 for m, rs in sorted(by_mutator.items())},
    )
