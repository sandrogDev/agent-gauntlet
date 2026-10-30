from __future__ import annotations

from datetime import UTC, datetime

from gauntlet.models import (
    CheckResult,
    RunResult,
    Scenario,
    ScenarioResult,
    Severity,
    Trace,
)
from gauntlet.regression import DiffStatus, compare
from gauntlet.scoring import compute_metrics, percentile, score_scenario
from gauntlet.storage import list_runs, resolve_run, save_run


def _check(
    passed: bool,
    weight: float = 1.0,
    severity: Severity = Severity.MAJOR,
    skipped: bool = False,
    label: str = "c",
):
    return CheckResult(
        check="x", label=label, passed=passed, weight=weight, severity=severity, skipped=skipped
    )


def _result(
    sid: str, passed: bool, score: float, failing: list[str] | None = None
) -> ScenarioResult:
    checks = [_check(False, label=f) for f in (failing or [])]
    return ScenarioResult(
        scenario=Scenario(id=sid, turns=["x"]),
        trace=Trace(scenario_id=sid),
        checks=checks,
        score=score,
        passed=passed,
    )


def _run(run_id: str, *results: ScenarioResult) -> RunResult:
    now = datetime.now(UTC)
    return RunResult(
        run_id=run_id,
        suite="s",
        provider="p",
        seed=0,
        started_at=now,
        finished_at=now,
        gauntlet_version="t",
        results=list(results),
        metrics=compute_metrics(results),
    )


def test_score_weights_and_skips() -> None:
    checks = [_check(True, 3), _check(False, 1), _check(False, 5, skipped=True)]
    score, passed = score_scenario(checks, Trace(scenario_id="t"), threshold=0.7)
    assert score == 0.75 and passed


def test_critical_failure_always_fails() -> None:
    checks = [_check(True, 10), _check(False, 1, Severity.CRITICAL)]
    score, passed = score_scenario(checks, Trace(scenario_id="t"), threshold=0.5)
    assert score > 0.9 and not passed


def test_trace_error_zeroes_score() -> None:
    assert score_scenario([_check(True)], Trace(scenario_id="t", error="boom"), 0.5) == (0.0, False)


def test_percentile() -> None:
    assert percentile([], 0.95) == 0.0
    assert percentile([float(i) for i in range(1, 101)], 0.95) == 95.0


def test_compare_classifies_every_status() -> None:
    base = _run(
        "b",
        _result("reg", True, 1.0),
        _result("fix", False, 0.4),
        _result("deg", True, 1.0),
        _result("same", True, 1.0),
        _result("gone", True, 1.0),
    )
    cand = _run(
        "c",
        _result("reg", False, 0.5, failing=["tool_called(x)"]),
        _result("fix", True, 0.9),
        _result("deg", True, 0.85),
        _result("same", True, 0.98),
        _result("new", True, 1.0),
    )
    cmp = compare(base, cand, tolerance=0.05)
    status = {d.scenario_id: d.status for d in cmp.diffs}
    assert status == {
        "reg": DiffStatus.REGRESSED,
        "fix": DiffStatus.FIXED,
        "deg": DiffStatus.DEGRADED,
        "same": DiffStatus.UNCHANGED,
        "gone": DiffStatus.REMOVED,
        "new": DiffStatus.NEW,
    }
    regressed = next(d for d in cmp.diffs if d.scenario_id == "reg")
    assert regressed.newly_failing == ["tool_called(x)"] and regressed.delta == -0.5
    assert cmp.has_regressions
    assert cmp.metric_deltas["pass_rate"] == 0.0
    assert cmp.metric_deltas["mean_score"] < 0


def test_storage_roundtrip_and_resolution(tmp_path) -> None:
    first = _run("20260101-000000-aaaaaa", _result("a", True, 1.0))
    second = _run("20260102-000000-bbbbbb", _result("a", False, 0.0))
    save_run(first, tmp_path)
    path = save_run(second, tmp_path)

    assert [p.stem for p in list_runs(tmp_path)] == [first.run_id, second.run_id]
    assert resolve_run("latest", tmp_path).run_id == second.run_id
    assert resolve_run("previous", tmp_path).run_id == first.run_id
    assert resolve_run(first.run_id, tmp_path).run_id == first.run_id
    assert resolve_run(str(path), tmp_path).results[0].passed is False
