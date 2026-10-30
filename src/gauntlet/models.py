"""Core data model.

Everything that crosses a module boundary is a pydantic model, so a run can be
serialised to JSON, stored, diffed against another run and re-rendered as a
report without re-executing a single model call.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Conversation primitives (provider-neutral)
# --------------------------------------------------------------------------- #


class Severity(StrEnum):
    """How bad a failed check is. Any CRITICAL failure fails the scenario."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class ToolSpec(BaseModel):
    """What the agent is told about a tool: name, description, JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One entry in the provider-neutral conversation.

    Providers translate this into their own wire format, so adding a new
    provider never touches the runner.
    """

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class AgentStep(BaseModel):
    """A single model response: optional text plus zero or more tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


# --------------------------------------------------------------------------- #
# Execution trace
# --------------------------------------------------------------------------- #


class ToolCallRecord(BaseModel):
    """Everything that happened when the agent attempted one tool call."""

    turn: int
    step: int
    call: ToolCall
    result: Any = None
    error: str | None = None
    schema_errors: list[str] = Field(default_factory=list)
    unknown_tool: bool = False
    injected_fault: bool = False

    @property
    def valid(self) -> bool:
        """The call named a real tool and its arguments matched the schema."""
        return not self.unknown_tool and not self.schema_errors

    @property
    def succeeded(self) -> bool:
        return self.error is None


class Trace(BaseModel):
    scenario_id: str
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    final_responses: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    steps: int = 0
    hit_step_limit: bool = False
    error: str | None = None

    def calls_to(self, name: str) -> list[ToolCallRecord]:
        return [r for r in self.tool_calls if r.call.name == name]

    @property
    def last_response(self) -> str:
        return self.final_responses[-1] if self.final_responses else ""


# --------------------------------------------------------------------------- #
# Suite definition (what users write in YAML)
# --------------------------------------------------------------------------- #


class CheckSpec(BaseModel):
    """A check to run against a trace. Unknown keys become check parameters."""

    model_config = ConfigDict(extra="allow")

    check: str
    weight: float = Field(default=1.0, ge=0)
    severity: Severity = Severity.MAJOR
    description: str | None = None

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    @property
    def label(self) -> str:
        """Human-readable identifier, e.g. ``tool_called(cancel_booking)``."""
        if self.description:
            return self.description
        key = next(
            (self.params[k] for k in ("tool", "name", "text", "pattern") if k in self.params),
            None,
        )
        return f"{self.check}({key})" if key is not None else self.check


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    turns: list[str] = Field(min_length=1)
    expect: list[CheckSpec] = Field(default_factory=list)
    # tool name -> error message returned instead of executing the first call
    tool_faults: dict[str, str] = Field(default_factory=dict)
    variant_of: str | None = None
    mutator: str | None = None

    @property
    def is_adversarial(self) -> bool:
        return self.variant_of is not None


class MutatorSpec(BaseModel):
    """An adversarial transformation applied to (a subset of) the scenarios."""

    model_config = ConfigDict(extra="allow")

    mutator: str
    tags: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    extra_expect: list[CheckSpec] = Field(default_factory=list)

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    def applies_to(self, scenario: Scenario) -> bool:
        if not self.tags and not self.scenarios:
            return True
        return scenario.id in self.scenarios or bool(set(self.tags) & set(scenario.tags))


def _default_global_checks() -> list[CheckSpec]:
    return [
        CheckSpec(check="schema_valid", severity=Severity.CRITICAL),
        CheckSpec(check="completed", severity=Severity.CRITICAL),
    ]


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    system_prompt: str
    tools_module: str
    max_steps: int = Field(default=8, ge=1, le=100)
    pass_threshold: float = Field(default=0.8, ge=0, le=1)
    global_checks: list[CheckSpec] = Field(default_factory=_default_global_checks)
    scenarios: list[Scenario] = Field(min_length=1)
    adversarial: list[MutatorSpec] = Field(default_factory=list)
    base_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _unique_ids(self) -> Suite:
        counts = Counter(s.id for s in self.scenarios)
        dupes = [sid for sid, n in counts.items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate scenario ids: {sorted(dupes)}")
        return self

    @property
    def tools_path(self) -> Path:
        return (self.base_dir / self.tools_module).resolve()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


class CheckResult(BaseModel):
    check: str
    label: str
    passed: bool
    weight: float
    severity: Severity
    message: str = ""
    skipped: bool = False


class ScenarioResult(BaseModel):
    scenario: Scenario
    trace: Trace
    checks: list[CheckResult]
    score: float
    passed: bool

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and not c.skipped]


class RunMetrics(BaseModel):
    scenarios: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    mean_score: float = 0.0
    baseline_pass_rate: float | None = None
    adversarial_pass_rate: float | None = None
    tool_calls: int = 0
    tool_call_validity: float = 1.0
    tool_call_success_rate: float = 1.0
    critical_failures: int = 0
    errored: int = 0
    mean_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    failures_by_check: dict[str, int] = Field(default_factory=dict)
    pass_rate_by_mutator: dict[str, float] = Field(default_factory=dict)


class RunResult(BaseModel):
    run_id: str
    suite: str
    provider: str
    seed: int
    started_at: datetime
    finished_at: datetime
    gauntlet_version: str
    results: list[ScenarioResult]
    metrics: RunMetrics

    def by_id(self) -> dict[str, ScenarioResult]:
        return {r.scenario.id: r for r in self.results}
