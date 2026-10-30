"""Compare two runs scenario-by-scenario to gate releases in CI."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from gauntlet.models import RunResult


class DiffStatus(StrEnum):
    REGRESSED = "regressed"  # pass -> fail
    DEGRADED = "degraded"  # still passing but score dropped beyond tolerance
    FIXED = "fixed"  # fail -> pass
    IMPROVED = "improved"  # still failing but score rose beyond tolerance
    UNCHANGED = "unchanged"
    NEW = "new"
    REMOVED = "removed"


class ScenarioDiff(BaseModel):
    scenario_id: str
    status: DiffStatus
    baseline_score: float | None = None
    candidate_score: float | None = None
    newly_failing: list[str] = Field(default_factory=list)

    @property
    def delta(self) -> float | None:
        if self.baseline_score is None or self.candidate_score is None:
            return None
        return round(self.candidate_score - self.baseline_score, 4)


class RunComparison(BaseModel):
    baseline_id: str
    candidate_id: str
    diffs: list[ScenarioDiff]
    metric_deltas: dict[str, float]

    def with_status(self, *statuses: DiffStatus) -> list[ScenarioDiff]:
        return [d for d in self.diffs if d.status in statuses]

    @property
    def has_regressions(self) -> bool:
        return bool(self.with_status(DiffStatus.REGRESSED, DiffStatus.DEGRADED))


_COMPARED_METRICS = (
    "pass_rate",
    "mean_score",
    "baseline_pass_rate",
    "adversarial_pass_rate",
    "tool_call_validity",
    "tool_call_success_rate",
    "p95_latency_s",
)


def compare(baseline: RunResult, candidate: RunResult, *, tolerance: float = 0.05) -> RunComparison:
    base, cand = baseline.by_id(), candidate.by_id()
    diffs: list[ScenarioDiff] = []

    for sid in [*base, *(k for k in cand if k not in base)]:
        b, c = base.get(sid), cand.get(sid)
        if b is None:
            assert c is not None
            diffs.append(
                ScenarioDiff(scenario_id=sid, status=DiffStatus.NEW, candidate_score=c.score)
            )
            continue
        if c is None:
            diffs.append(
                ScenarioDiff(scenario_id=sid, status=DiffStatus.REMOVED, baseline_score=b.score)
            )
            continue

        delta = c.score - b.score
        if b.passed and not c.passed:
            status = DiffStatus.REGRESSED
        elif not b.passed and c.passed:
            status = DiffStatus.FIXED
        elif delta < -tolerance:
            status = DiffStatus.DEGRADED if c.passed else DiffStatus.UNCHANGED
        elif delta > tolerance and not c.passed:
            status = DiffStatus.IMPROVED
        else:
            status = DiffStatus.UNCHANGED

        before = {x.label for x in b.failures}
        newly = [x.label for x in c.failures if x.label not in before]
        diffs.append(
            ScenarioDiff(
                scenario_id=sid,
                status=status,
                baseline_score=b.score,
                candidate_score=c.score,
                newly_failing=newly,
            )
        )

    deltas: dict[str, float] = {}
    for name in _COMPARED_METRICS:
        bv, cv = getattr(baseline.metrics, name), getattr(candidate.metrics, name)
        if bv is not None and cv is not None:
            deltas[name] = round(cv - bv, 4)

    return RunComparison(
        baseline_id=baseline.run_id,
        candidate_id=candidate.run_id,
        diffs=diffs,
        metric_deltas=deltas,
    )
